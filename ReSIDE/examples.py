import math
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
from matplotlib.image import imsave
from torch.utils.data import DataLoader
from torchvision import transforms

from ReSIDE.loaddata import depthDataset
from ReSIDE.models import modules, resnet, densenet, net, senet
import numpy as np
from ReSIDE import sobel, loaddata, util
from ReSIDE.models.lasinger2019 import MidasNet
from ReSIDE.nyu_transform import Scale, CenterCrop, ToTensor
from ReSIDE.train import define_model
from ReSIDE.util import MetricsTracker


def main():
    checkpoint_dir = os.path.abspath("checkpoints")

    test_loader = loaddata.getTestingData(8)
    first_batch = next(iter(test_loader))
    images, depth = first_batch['image'], first_batch['depth']
    images, depth = images.cuda(), depth.cuda()

    checkpoints = os.listdir(checkpoint_dir)

    examples_dir = os.path.join("examples", "nyu")

    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)

    outputs = defaultdict(lambda: defaultdict(list))

    for checkpoint_file in checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        checkpoint_name = Path(checkpoint_file).stem
        parts = checkpoint_name.split("-")

        if len(parts) == 2:
            encoder_name, decoder_name = parts
        else:  # len(parts) == 3
            encoder_name = '-'.join(parts[:2])
            decoder_name = parts[2]

        print(checkpoint_path)

        if 'hu' in checkpoint_file.lower():
            if 'resnet' in encoder_name.lower() or 'rn' in encoder_name.lower():
                model = define_model(is_resnet=True).cuda()
            elif 'efficientnet' in encoder_name.lower() or 'en' in encoder_name.lower():
                if 'en' in encoder_name.lower():  # Handle name style 'ENB#', e.g., 'ENB0', 'ENB4' etc.
                    efficientnet_name = f"efficientnet-b{encoder_name[-1:].lower()}"
                else:
                    efficientnet_name = encoder_name.lower()

                model = define_model(is_efficientnet=True, efficientnet_variant=efficientnet_name)
            else:
                raise RuntimeError("Unsupported encoder: {}.".format(encoder_name))

            state_dict = torch.load(checkpoint_path)
            # state_dict = torch.load('checkpoint.pth.tar')['state_dict']
            # state_dict = {f"module.{key}": value for key, value in state_dict.items()}

            model.load_state_dict(state_dict)
            model.cuda()
        else:
            model = MidasNet.load(checkpoint_path).cuda()

        with torch.no_grad():
            for i, image in enumerate(images):
                output = model(image.unsqueeze(0)).squeeze().cpu().numpy()

                outputs[decoder_name][encoder_name].append(output)

    test_images = depthDataset(csv_file='./data/nyu2_test.csv',
                                       transform=transforms.Compose([
                                           Scale(240),
                                           CenterCrop([304, 228], [304, 228]),
                                           ToTensor(is_test=True)
                                       ]))

    test_images = DataLoader(test_images, 8, shuffle=False)

    for i, image in enumerate(next(iter(test_images))['image'].numpy()):
        image_name = "input_{:d}.png".format(i)

        imsave(os.path.join(examples_dir, image_name), image.transpose((1, 2, 0)))
        print("Writing {}...".format(image_name))

    max_depth = -float('inf')

    for decoder in sorted(outputs):
        for encoder in sorted(outputs[decoder]):
            for depth_map in outputs[decoder][encoder]:
                if depth_map.max() > max_depth:
                    max_depth = depth_map.max()

    for decoder in sorted(outputs):
        for encoder in sorted(outputs[decoder]):
            jobs = []

            for i, depth_map in enumerate(outputs[decoder][encoder]):
                depth_map_name = "{}-{}-{:d}.png".format(encoder, decoder, i)

                jobs.append((os.path.join(examples_dir, depth_map_name), depth_map))

            for path, img_data in jobs:
                img_data = (255 * img_data / max_depth).astype(np.uint8)

                imsave(path, img_data)
                print("Writing {}...".format(path))


if __name__ == '__main__':
    main()
