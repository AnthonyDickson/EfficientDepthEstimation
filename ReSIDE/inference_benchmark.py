import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import stdev, mean

import pandas as pd
import plac
import torch
import torch.nn.parallel
import torch.nn.parallel
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose

from ReSIDE.demo_transform import Scale, CenterCrop, ToTensor, Normalize
from ReSIDE.models.lasinger2019 import MidasNet
from ReSIDE.train import define_model
from ReSIDE.util import Timer


@plac.annotations(
    image_path=plac.Annotation('The path to an RGB image or a directory containing RGB images.', type=str,
                               kind='option', abbrev='i'),
    checkpoint_path=plac.Annotation('The path to the folder containing the trained model weights.', type=str,
                                    kind='option', abbrev='c'),
    output_path=plac.Annotation('The path to save the benchmark results.', type=str, kind='option', abbrev='o'),
    num_trials=plac.Annotation('How many times to repeat the benchmark for each model', type=str, kind='option',
                               abbrev='n'),
)
def main(image_path, checkpoint_path='checkpoints', output_path="benchmark_results.csv", num_trials=5):
    """
    Measure the memory usage and inference times of a set of models over the frames of a video.

    :param image_path: The path to the video frames.
    :param checkpoint_path: The folder containing the model checkpoints.
    :param output_path: The path of the file to save the results to.
    :param num_trials: How many times to run the benchmark to get average stats and account for variation in inference
    times due to other stuff happening on the pc.
    """
    print(f"Loading dataset from {image_path}...")
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transform = transforms.Compose([
        Scale([320, 240]),
        CenterCrop([304, 228]),
        ToTensor(),
        Normalize(__imagenet_stats['mean'],
                  __imagenet_stats['std'])
    ])

    dataset = VideoFrameDataset(image_path, transform)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=False)

    benchmark_results = defaultdict(lambda: {'load_time': [], 'inference_time': [], 'memory_usage': []})
    # load_time, inference_time, memory_usage = benchmark_video(dataloader,
    #                                                           os.path.join(checkpoint_path, "resnet50-lasinger.pth"))
    # print(load_time, inference_time, memory_usage)

    for model_name in sorted(os.listdir(checkpoint_path)):
        model_path = os.path.join(checkpoint_path, model_name)

        for n in range(num_trials):
            print(f"{model_name}, Trial {n + 1}")
            load_time, inference_time, memory_usage = benchmark_video(dataloader, model_path)
            benchmark_results[model_name]['load_time'].append(load_time.total_seconds())
            benchmark_results[model_name]['inference_time'].append(inference_time.total_seconds())
            benchmark_results[model_name]['memory_usage'].append(memory_usage / 1e9)  # convert from bytes to GB

    aggregate_results = dict()

    def formatted_stats(model_name, metric):
        avg = mean(benchmark_results[model_name][metric])
        std = stdev(benchmark_results[model_name][metric])

        return f"{avg:.1f} \\pm {std:.1f}"

    for model_name in benchmark_results:
        model_name_ = model_name.replace(".pth", "")
        aggregate_results[model_name_] = {'load_time': formatted_stats(model_name, 'load_time'),
                                          'inference_time': formatted_stats(model_name, 'inference_time'),
                                          'memory_usage': formatted_stats(model_name, 'memory_usage'), }

    df = pd.DataFrame(aggregate_results).T
    print(df.to_latex(escape=False))
    df.to_csv(output_path)


class VideoFrameDataset(Dataset):
    def __init__(self, frames_dir: str, to_tensor_transform: Compose):
        """Dataset of video frames.

        :param frames_dir: The path to the video frames.
        :param to_tensor_transform: The transform that takes a numpy array and converts it to a torch Tensor object.
        """
        self.files = list(map(lambda filename: os.path.join(frames_dir, filename), sorted(os.listdir(frames_dir))))
        self.to_tensor = to_tensor_transform

    def __getitem__(self, index):
        img = Image.open(self.files[index])

        return self.to_tensor(img)

    def __len__(self):
        return len(self.files)


def benchmark_video(dataloader, model_path):
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()

    loading_timer = Timer()

    with loading_timer:
        filename = os.path.basename(model_path)
        checkpoint_name = Path(filename).stem
        parts = checkpoint_name.split("-")

        if len(parts) == 2:
            encoder_name, decoder_name = parts
        else:  # len(parts) == 3
            encoder_name = '-'.join(parts[:2])
            decoder_name = parts[2]

        print(model_path)

        if 'hu' in model_path:
            if 'resnet' in encoder_name.lower():
                model = define_model(is_resnet=True).cuda()
            elif 'efficientnet' in encoder_name.lower():
                model = define_model(is_efficientnet=True, efficientnet_variant=encoder_name.lower())
            else:
                raise RuntimeError("Unsupported encoder: {}.".format(encoder_name))

            try:
                state_dict = torch.load(model_path)

                model.load_state_dict(state_dict)
            except RuntimeError:
                # The pretrained weights are prefixed with 'module' but the model does not expect this so we need to strip
                # these from the statedict keys.
                state_dict = torch.load(model_path)
                state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

                model.load_state_dict(state_dict)

            # model.load_state_dict(state_dict)
            model.cuda()
        else:
            model = MidasNet.load(model_path).cuda()

    inference_timer = Timer()

    with torch.no_grad(), inference_timer:
        for i, batch in enumerate(dataloader):
            print(f"Batch {i + 1:03d}/{len(dataloader):03d}")
            height, width = batch.shape[-2:]
            depth = model(batch.cuda())

            _ = torch.nn.functional.interpolate(depth, size=(height, width), mode='bilinear', align_corners=True)

    peak_memory_usage = torch.cuda.max_memory_cached() + torch.cuda.max_memory_allocated()

    return loading_timer.elapsed, inference_timer.elapsed, peak_memory_usage


if __name__ == '__main__':
    plac.call(main)
