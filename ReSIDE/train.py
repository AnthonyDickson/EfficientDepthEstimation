import argparse
import datetime
import math
import os
import time
import warnings
from typing import Callable, Union, Optional, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.nn.functional as F

import wandb

from ReSIDE import sobel, loaddata, util
from ReSIDE.models import modules, resnet, densenet, net, senet
from ReSIDE.models.lasinger2019 import BestNet, Encoder, Decoder, BottleneckBlock
from ReSIDE.util import MetricsTracker, BestMetricsTracker, Timer, AverageMeter


def define_model(is_resnet=False, is_densenet=False, is_senet=False, is_efficientnet=False,
                 efficientnet_variant="efficientnet-b0"):
    if is_resnet:
        original_model = resnet.resnet50(pretrained=True)
        Encoder = modules.E_resnet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    elif is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel=[192, 384, 1056, 2208])
    elif is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    elif is_efficientnet:
        Encoder = modules.E_efficientnet(name=efficientnet_variant, pretrained=True)
        model = net.model(Encoder, num_features=Encoder.num_features, block_channel=Encoder.block_out_channels)

    return model


def main(args: Optional[List[str]] = None):
    """
    The main train/test loop for training a depth estimation network on NYUv2.

    :param args: The list of arguments for the program. By default (when this parameter is set to None), arguments are
    passed in from the command line. If you are calling this code from within Python, you can specify the command line
    arguments with this parameter.
    """
    parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
    parser.add_argument('--encoder', default="resnet50", type=str,
                        help='The encoder to use for the depth estimation network.')
    parser.add_argument('--decoder', default="hu2018", type=str, choices={"hu2018", "lasinger2019"},
                        help='The decoder to use for the depth estimation network.')
    parser.add_argument('--epochs', default=20, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    args = parser.parse_args(args=args)

    encoder = args.encoder
    decoder = args.decoder

    training_start_time = datetime.datetime.now()

    if decoder == "hu2018":
        if "efficientnet" in encoder:
            model = define_model(is_efficientnet=True, efficientnet_variant=encoder)
        elif "resnet" in encoder:
            model = define_model(is_resnet=True)
        elif "densenet" in encoder:
            model = define_model(is_densenet=True)
        elif "senet" in encoder:
            model = define_model(is_senet=True)
        else:
            raise RuntimeError(f"Unrecognised encoder '{encoder}'.")
    else:
        encoder_ = Encoder(name=encoder, pretrained=True)
        model = BestNet(
            encoder_,
            Decoder(encoder_.block_out_channels, num_features='auto'),
            output_size=(152, 114), input_size=(304, 228)
        )

    if torch.cuda.device_count() == 8:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
        batch_size = 64
    elif torch.cuda.device_count() == 4:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
        batch_size = 32
    else:
        model = model.cuda()
        batch_size = 8

    cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1)

    # train_loader = loaddata.getTrainingData(batch_size)
    # os.makedirs("checkpoints", exist_ok=True)
    #
    # for epoch in range(args.start_epoch, args.epochs):
    #     train(train_loader, model, optimizer, epoch)
    #     lr_scheduler.step(epoch)
    #     torch.save(model.state_dict(), os.path.join("checkpoints", f"encoder_name-{epoch + 1:02d}.pth"))

    train_loader = loaddata.getTrainingData(batch_size)
    test_loader = loaddata.getTestingData(batch_size)
    min_loss = float('inf')

    wandb.init(
        project="deep-depth-estimation",
        config={
            "network": {
                "encoder": {
                    "name": encoder
                },
                "decoder": {
                    "decoder_type": decoder
                }
            }
        },
        config_exclude_keys=["checkpoint_scheduler", "dataset"]
    )

    wandb.run.name = f"{encoder}-{decoder}-{wandb.run.id}"
    wandb.run.save()

    checkpoint_path = os.path.join(wandb.run.dir, f"{wandb.run.name}.pth")

    # noinspection PyTypeChecker
    wandb.watch(model, log=None)
    wandb.summary['num_parameters'] = model.num_parameters

    best_metrics = BestMetricsTracker()
    training_timer = Timer()
    test_timer = Timer()
    inference_timer = Timer()

    # for epoch in range(args.start_epoch, args.epochs):
    #     train(train_loader, model, optimizer, epoch)
    #     metrics = test(test_loader, model)
    #     lr_scheduler.step(epoch)
    #
    #     if metrics.abs_rel.value < min_loss:
    #         min_loss = metrics.abs_rel.value
    #
    #         if decoder == "lasinger2019":
    #             model.save(checkpoint_path)
    #         else:
    #             torch.save(model.state_dict(), checkpoint_path)

    for epoch in range(args.start_epoch, args.epochs):
        elapsed_time = datetime.datetime.now() - training_start_time
        print(f"Epoch {epoch + 1:02d}/{args.epochs:02d} - Total Elapsed Time: {elapsed_time}")

        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()

        with training_timer:
            train(train_loader, model, optimizer, epoch)

        with test_timer:
            metrics = test(test_loader, model)

        if metrics.abs_rel.value < min_loss:
            min_loss = metrics.abs_rel.value

            if decoder == "lasinger2019":
                model.save(checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)

        # noinspection PyArgumentList
        lr_scheduler.step()

        with torch.no_grad():
            model.eval()
            images = next(iter(test_loader))["image"].cuda()

            with inference_timer:
                examples = model(images)

            examples = F.interpolate(examples, size=images.shape[-2:], mode="bilinear", align_corners=True)
            # WANDB expects will scale to [0, 255], so we need to make sure the output is scaled to [0.0, 1.0] so
            #  that pixel values don't get clipped. For the NYUv2 dataset the max depth is 10m, so we divide by that.
            examples *= 1 / 10
            # Have to convert to numpy array to avoid normalisation of depth values so the scale is preserved between
            # examples from different batches.
            examples = examples.cpu().numpy()
            examples = [wandb.Image(example) for example in examples]

        best_metrics.update(metrics)

        for metric_name, metric_value in best_metrics.to_dict().items():
            wandb.summary[metric_name] = metric_value

        wandb.log({
            **metrics.to_dict(),
            "examples": examples,
            "vram_usage": torch.cuda.max_memory_cached(),
            "training_frame_time": training_timer.elapsed.total_seconds() / len(train_loader.dataset),
            "test_frame_time": test_timer.elapsed.total_seconds() / len(test_loader.dataset),
            "inference_time": inference_timer.elapsed.total_seconds() / len(images)
        })

    print(f"Total Training Time: {datetime.datetime.now() - training_start_time}.")
    wandb.join()


def train(train_loader, model, optimizer, epoch):
    losses = AverageMeter()

    model.train()

    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = sobel.Sobel().cuda()

    metrics = MetricsTracker()

    epoch_start_time = datetime.datetime.now()
    epoch_progress = 0

    for i, sample_batched in enumerate(train_loader):
        image, depth = sample_batched['image'], sample_batched['depth']

        depth = depth.cuda()
        image = image.cuda()

        ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().cuda()
        optimizer.zero_grad()

        output = model(image)

        depth_grad = get_gradient(depth)
        output_grad = get_gradient(output)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()
        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()

        loss = loss_depth + loss_normal + (loss_dx + loss_dy)

        losses.update(loss.item(), image.size(0))
        loss.backward()
        optimizer.step()

        metrics.update(outputs=output, labels=depth)

        epoch_progress += image.shape[0]

        elapsed_time = datetime.datetime.now() - epoch_start_time
        time_per_instance = elapsed_time.total_seconds() / epoch_progress

        print(f"\rTrain [{epoch_progress:05d}/{len(train_loader.dataset):05d}]"
              f" - Elapsed Time: {elapsed_time} ({time_per_instance:.4f}s/image)"
              f" - Loss: {loss.item():.3f} (Avg.: {losses.value:.3f})"
              f" - {metrics}", end="")

    print()

    return losses.value


def test(test_loader, model):
    model.eval()

    metrics = MetricsTracker()
    epoch_progress = 0
    epoch_start_time = datetime.datetime.now()

    for i, sample_batched in enumerate(test_loader):
        image, depth = sample_batched['image'], sample_batched['depth']

        depth = depth.cuda()
        image = image.cuda()

        output = model(image)
        output = torch.nn.functional.interpolate(output,
                                                 size=[depth.size(2), depth.size(3)],
                                                 mode='bilinear',
                                                 align_corners=True)

        metrics.update(outputs=output, labels=depth)

        batch_size = image.shape[0]
        epoch_progress += batch_size

        elapsed_time = datetime.datetime.now() - epoch_start_time
        time_per_instance = elapsed_time.total_seconds() / epoch_progress

        print(f"\rTrain [{epoch_progress:05d}/{len(test_loader.dataset):05d}]"
              f" - Elapsed Time: {elapsed_time} ({time_per_instance:.4f}s/image)"
              f" - {metrics}", end="")

    print()

    return metrics


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


if __name__ == '__main__':
    main()
