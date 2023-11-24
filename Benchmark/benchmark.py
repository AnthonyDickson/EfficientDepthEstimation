import argparse
import datetime
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Optional

import cv2
import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from monodepth.midas_custom_model import MidasModel
from ReSIDE.models.modules import E_resnet, E_senet, E_efficientnet
from ReSIDE.nyu_transform import ToTensor
from ReSIDE.util import Timer, MetricsTracker, AverageMeter
from monodepth.reside_model import ReSIDEModel
from DepthRenderer.DepthRenderer import animation, render, utils
from utils import image_io


def gaussian_kernel_1d(window_size, sigma):
    """
    Compute a 1-D Gaussian convolution kernel.

    A Pytorch implementation of a 1d Gaussian kernel based on the SciPy implementation
     https://github.com/scipy/scipy/blob/5ab7426247900db9de856e790b8bea1bd71aec49/scipy/ndimage/filters.py#L179.

    Args:
        window_size:
        sigma:

    Returns:

    """
    radius = window_size // 2
    x = torch.arange(-radius, radius + 1)
    sigma2 = sigma ** 2
    phi_x = torch.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    return phi_x


def gaussian_kernel_2d(kernel_size, num_channels):
    """
    Compute a 2-D Gaussian convolutional kernel to be used on a NCHW tensor.
    
    Args:
        kernel_size: The width of the kernel.
        num_channels: The number of channels in the NCHW tensor to be convolved over.

    Returns: The 2-D Gaussian kernel.
    """
    kernel_1d = gaussian_kernel_1d(kernel_size, 1.5).unsqueeze(1)
    kernel_2d = kernel_1d @ kernel_1d.T
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    kernel_2d = kernel_2d.expand(num_channels, 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d.contiguous()

    return kernel_2d


def _ssim(img1, img2, kernel, kernel_size, channel, return_batch_average=True):
    kernel_radius = kernel_size // 2

    mu_x = F.conv2d(img1, kernel, padding=kernel_radius, groups=channel)
    mu_y = F.conv2d(img2, kernel, padding=kernel_radius, groups=channel)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_x_mu_y = mu_x * mu_y

    var_x = F.conv2d(img1 * img1, kernel, padding=kernel_radius, groups=channel) - mu_x2
    var_y = F.conv2d(img2 * img2, kernel, padding=kernel_radius, groups=channel) - mu_y2
    covar_xy = F.conv2d(img1 * img2, kernel, padding=kernel_radius, groups=channel) - mu_x_mu_y

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_x_mu_y + C1) * (2 * covar_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (var_x + var_y + C2))

    if return_batch_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, kernel_size=11, return_batch_average=True):
        super(SSIM, self).__init__()
        self.kernel_size = kernel_size
        self.return_batch_average = return_batch_average
        self.num_channels = 1
        self.window = gaussian_kernel_2d(kernel_size, self.num_channels)

    def forward(self, img1, img2):
        _, num_channels, _, _ = img1.size()

        if num_channels == self.num_channels and self.window.data.type() == img1.data.type():
            kernel = self.window
        else:
            kernel = gaussian_kernel_2d(self.kernel_size, num_channels)

            if img1.is_cuda:
                kernel = kernel.cuda(img1.get_device())
            kernel = kernel.type_as(img1)

            self.window = kernel
            self.num_channels = num_channels

        return _ssim(img1, img2, kernel, self.kernel_size, num_channels, self.return_batch_average)


def ssim(img1, img2, kernel_size=11, return_batch_average=True):
    _, num_channels, _, _ = img1.shape
    kernel = gaussian_kernel_2d(kernel_size, num_channels)

    if img1.is_cuda:
        kernel = kernel.cuda(img1.get_device())

    kernel = kernel.type_as(img1)

    return _ssim(img1, img2, kernel, kernel_size, num_channels, return_batch_average)


def psnr(img1, img2, return_batch_average=True):
    """
    Compute the peak signal to noise ratio between two images.

    The images are expected to in floating point format with values in the range [0.0, 1.0].
    """
    assert len(img1.shape) == 4, f"Inputs must be in NCHW format. Got inputs with shapes {img1.shape} and {img2.shape}."
    assert img1.shape == img2.shape, f"The shape of the inputs do not match. Got {img1.shape} and {img2.shape}."
    assert img1.dtype == img2.dtype, f"The data type of the inputs do not match. Get {img1.dtype} and {img2.dtype}."
    assert img1.dtype != torch.uint8, f"Byte data type not supported for inputs."
    assert img1.min() >= 0.0 and img1.max() <= 1.0 and img2.min() >= 0.0 and img2.max() <= 1.0, \
        f"Inputs must be in the range [0.0, 1.0], got [{img1.min()}, {img1.max()}] and [{img2.min()}, {img2.max()}]."

    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])

    x = 10.0 * torch.log10(1.0 / mse)

    return x.mean() if return_batch_average else x


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
            self,
            width,
            height,
            resize_target=True,
            keep_aspect_ratio=False,
            ensure_multiple_of=1,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        sample = {sample_type: np.asarray(img) for sample_type, img in sample.items()}

        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                sample["mask"] = sample["mask"].astype(bool)

        sample = {sample_type: Image.fromarray(img) for sample_type, img in sample.items()}

        return sample


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

        assert all([isinstance(x, type(size[0])) for x in size]), "Sizes must all be the same type."

        if isinstance(size[0], float):
            assert all([0. < x < 1. for x in size]), "Float values must be a ratio between 0.0 and 1.0"

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        image = self.centerCrop(image, self.size)
        depth = self.centerCrop(depth, self.size)

        return {'image': image, 'depth': depth}

    def centerCrop(self, image, size):

        w1, h1 = image.size

        tw, th = size

        if isinstance(tw, float) or isinstance(th, float):
            tw = tw * w1
            th = th * h1

            tw = 2 * round(tw / 2)
            th = 2 * round(th / 2)

        if w1 == tw and h1 == th:
            return image

        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))

        image = image.crop((x1, y1, tw + x1, th + y1))

        return image


class DepthDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform
        self.root_dir = os.path.dirname(csv_file)

        try:
            with open(os.path.join(self.root_dir, "camera.json"), 'r') as f:
                camera_parms = json.load(f)

            self.camera_matrix = np.array([
                [camera_parms['fx'], 0., camera_parms['cx']],
                [0., camera_parms['fy'], camera_parms['cy']],
                [0., 0., 0.],
            ])

            self.camera_intrinsics = camera_parms
        except FileNotFoundError:
            warnings.warn(f"Could not find 'camera.json' in {self.root_dir}")

            self.camera_matrix = np.eye(3, dtype=np.float)
            self.camera_intrinsics = {
                'width': float('nan'),
                'height': float('nan'),
                'fx': float('nan'),
                'fy': float('nan'),
                'cx': float('nan'),
                'cy': float('nan')
            }

    def __getitem__(self, idx):
        image_name = self.frame.iloc[idx, 0]
        image_name = os.path.join(self.root_dir, image_name)
        depth_name = self.frame.iloc[idx, 1]
        depth_name = os.path.join(self.root_dir, depth_name)

        image = Image.open(image_name)
        depth = Image.open(depth_name)

        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.frame)

    @property
    def image_paths(self):
        return [os.path.join(self.root_dir, frame_path) for frame_path in sorted(self.frame.iloc[:, 0])]

    @property
    def depth_paths(self):
        return [os.path.join(self.root_dir, frame_path) for frame_path in sorted(self.frame.iloc[:, 1])]


class ImageFolderDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        assert os.path.isdir(base_dir), f"Could not find the folder: {base_dir}"

        self.base_dir = base_dir
        self.transform = transform

        filenames = list(sorted(os.listdir(base_dir)))
        assert len(filenames) > 0, f"No files found in the folder: {base_dir}"

        self.image_paths = [os.path.join(base_dir, filename) for filename in filenames]

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        if image_path.endswith('.raw'):
            image = image_io.load_raw_float32_image(image_path)
        else:
            image = Image.open(image_path)
            image = np.asarray(image)

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_paths)


class NestedImageFolderDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        assert os.path.isdir(base_dir), f"Could not find the folder: {base_dir}"

        self.base_dir = base_dir
        self.transform = transform

        self.image_paths = self._collect_image_paths(base_dir)

        assert len(self.image_paths) > 0, f"Found no images in the folder: {base_dir}"

    @staticmethod
    def _collect_image_paths(base_dir, allowed_extensions=None):
        if allowed_extensions is None:
            allowed_extensions = {'.png', '.jpeg', '.jpg'}

        sub_dirs = sorted(os.listdir(base_dir))

        image_paths = []

        for sub_dir in sub_dirs:
            path = os.path.join(base_dir, sub_dir)

            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                item_path = Path(item_path)

                if os.path.isfile(item_path) and item_path.suffix in allowed_extensions:
                    image_paths.append(item_path)

        return image_paths

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        if image_path.suffix == '.raw':
            image = image_io.load_raw_float32_image(image_path)
        else:
            image = Image.open(image_path)
            image = np.asarray(image)

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_paths)


class FlatDepthEstimator(nn.Module):
    """ Dummy neural networks that always predicts 0 depth."""

    def forward(self, image):
        N, _, H, W = image.shape
        C = 1

        return 3 * torch.zeros(size=(N, C, H, W), dtype=torch.float32, device=image.device)


def main(args):
    benchmark_results = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    labels_loader = get_nyu_loader(args.batch_size, csv_path=args.csv_path)
    num_frames = len(labels_loader.dataset)
    dataset = 'nyu'
    sample_output_base_dir = os.path.join(args.output_path, dataset)

    # TODO: Per-model dataloaders
    default_data_loader = get_nyu_loader(args.batch_size, args.csv_path, ensure_multiple_of=1)
    data_loader_power_of_two = get_nyu_loader(args.batch_size, args.csv_path, ensure_multiple_of=32)

    baseline_model = 'reside_senet'

    chapter_3_models = [
        (
            'reside_enb0',
            lambda: ReSIDEModel(
                model_path=args.reside_enb0_path,
                encoder_type=E_efficientnet,
                efficientnet_variant="efficientnet-b0"
            ),
            data_loader_power_of_two
        ),
        (
            'reside_enb4',
            lambda: ReSIDEModel(
                model_path=args.reside_enb4_path,
                encoder_type=E_efficientnet,
                efficientnet_variant="efficientnet-b4"
            ),
            data_loader_power_of_two
        ),
        ('reside_resnet50', lambda: ReSIDEModel(model_path=args.reside_resnet50_path, encoder_type=E_resnet),
         data_loader_power_of_two),
        ('midas_enb0', lambda: MidasModel(model_path=args.midas_enb0_path), data_loader_power_of_two),
        ('midas_enb4', lambda: MidasModel(model_path=args.midas_enb4_path), data_loader_power_of_two),
        ('midas_resnet50', lambda: MidasModel(model_path=args.midas_resnet50_path), data_loader_power_of_two),
    ]

    chapter_4_models = [
        (
            'reside_enb0',
            lambda: ReSIDEModel(
                model_path=args.reside_enb0_path,
                encoder_type=E_efficientnet,
                efficientnet_variant="efficientnet-b0"
            ),
            data_loader_power_of_two
        ),
        ('reside_senet', lambda: ReSIDEModel(model_path=args.reside_senet_path, encoder_type=E_senet),
         data_loader_power_of_two),
        ('reside_enb0-random_weights', lambda: ReSIDEModel(
            model_path="/home/anthony/PycharmProjects/consistent_depth/checkpoints/reside_small/hu-enb0_random-weights.pth",
            encoder_type=E_efficientnet), data_loader_power_of_two),
        ('flat', FlatDepthEstimator, data_loader_power_of_two),
    ]

    model_loaders = chapter_4_models

    print("Creating rendered images for dataset...")
    dataset_render_dir = os.path.join(args.output_path, dataset, 'ground_truth')
    dataset_render_time = create_rendered_images(dataset_render_dir, labels_loader, fps=args.renderer_fps)
    dataset_rendered_images_dir = os.path.join(dataset_render_dir, 'image')
    dataset_render_loader = DataLoader(
        NestedImageFolderDataset(dataset_rendered_images_dir),
        shuffle=False,
        batch_size=args.batch_size
    )
    print(f"Dataset Rendering Elapsed Time: {dataset_render_time}\n")

    print("Creating ground truth depth maps with added noise...")

    model_name = 'random'
    noisy_depth_dir = os.path.join(args.output_path, dataset, model_name)
    noisy_depth_map_dir = os.path.join(noisy_depth_dir, 'depth', 'png')
    noisy_depth_map_creation_time = create_noisy_depth_maps(noisy_depth_map_dir, labels_loader)
    noisy_depth_map_loader = DataLoader(ImageFolderDataset(noisy_depth_map_dir), shuffle=False,
                                        batch_size=args.batch_size)

    print(f"Time to create noisy depth maps: {noisy_depth_map_creation_time}\n")
    print(f"Creating rendered images for noisy depth maps...")
    noisy_depth_render_dir = os.path.join(noisy_depth_dir, 'rendered_images')
    noisy_depth_render_time = create_rendered_images(noisy_depth_render_dir, labels_loader,
                                                     depth_loader=noisy_depth_map_loader, fps=args.renderer_fps)

    noisy_depth_rendered_images_dir = os.path.join(noisy_depth_render_dir, 'image')
    noisy_depth_render_loader = DataLoader(
        NestedImageFolderDataset(noisy_depth_rendered_images_dir),
        shuffle=False,
        batch_size=args.batch_size
    )
    print(f"Noisy Depth Map Rendering Elapsed Time: {noisy_depth_render_time}\n")

    def run_benchmark(model_name, depth_labels_loader, depth_output_loader, render_labels_loader, render_output_loader):
        print("Standard Benchmark")

        cache_dir = os.path.join(args.output_path, dataset, model_name)
        benchmark_timer = Timer()

        with benchmark_timer:
            metrics = test(depth_labels_loader, depth_output_loader, cache_dir)

        benchmark_results[dataset][model_name].update(metrics)
        benchmark_results[dataset][model_name]['standard_benchmark_time'] = benchmark_timer.elapsed.total_seconds()

        print(f"Standard Benchmark Elapsed Time: {benchmark_timer.elapsed}\n")

        print("Visual Benchmark")
        benchmark_timer = Timer()

        with benchmark_timer:
            metrics = test_visual(render_labels_loader, render_output_loader, cache_dir)

        benchmark_results[dataset][model_name].update(metrics)
        benchmark_results[dataset][model_name]['visual_benchmark_time'] = benchmark_timer.elapsed.total_seconds()

        print(f"Visual Benchmark Elapsed Time: {benchmark_timer.elapsed}\n")

        save_benchmark_results(args, benchmark_results, relative_to=baseline_model)

    run_benchmark('random', labels_loader, noisy_depth_map_loader, dataset_render_loader, noisy_depth_render_loader)

    for model_name, model_loader, data_loader in model_loaders:
        print("=" * 80)
        print(dataset, model_name)
        print("=" * 80)

        print("Generating Depth Maps...")
        depth_output_dir = os.path.join(args.output_path, dataset, model_name, 'depth')

        peak_memory_usage, inference_time, io_time = \
            create_depth_maps(depth_output_dir, model_loader, data_loader)

        inference_less_io_time = inference_time - io_time
        print(f"Depth maps generated in {inference_time} ({inference_less_io_time} without IO).\n")

        benchmark_results[dataset][model_name]['peak_memory_usage'] = peak_memory_usage
        benchmark_results[dataset][model_name]['inference_time'] = inference_time.total_seconds()
        benchmark_results[dataset][model_name]['inference_time_no_io'] = inference_less_io_time.total_seconds()
        benchmark_results[dataset][model_name]['frame_time'] = inference_less_io_time.total_seconds() / num_frames

        outputs_loader = DataLoader(
            ImageFolderDataset(os.path.join(depth_output_dir, 'raw')),
            shuffle=False,
            batch_size=args.batch_size
        )

        print("Creating rendered images...")
        render_output_dir = os.path.join(args.output_path, dataset, model_name, 'rendered_images')
        render_time = create_rendered_images(render_output_dir, image_loader=labels_loader, depth_loader=outputs_loader)
        benchmark_results[dataset][model_name]['render_time'] = render_time.total_seconds()

        model_rendered_images_dir = os.path.join(render_output_dir, 'image')
        model_render_loader = DataLoader(
            NestedImageFolderDataset(model_rendered_images_dir),
            shuffle=False,
            batch_size=args.batch_size
        )

        print(f"Rendering Elapsed Time: {render_time}\n")

        run_benchmark(model_name, data_loader, outputs_loader, dataset_render_loader, model_render_loader)

    model_comparison_img = images_to_grid(data_loader_power_of_two.dataset, sample_output_base_dir, output_type='depth')
    Image.fromarray(model_comparison_img).save(os.path.join(sample_output_base_dir, "nyu-depth.png"))

    model_comparison_img = images_to_grid(data_loader_power_of_two.dataset, sample_output_base_dir,
                                          output_type='rendered_images')
    Image.fromarray(model_comparison_img).save(os.path.join(sample_output_base_dir, "nyu-rendered_images.png"))

    absolute_results_csv_path = os.path.join(args.output_path, f"{dataset}.csv")
    relative_results_csv_path = os.path.join(args.output_path, f"{dataset}-relative.csv")
    visualisation_output_path = os.path.join(args.output_path, "plots")
    os.makedirs(visualisation_output_path, exist_ok=True)

    assert os.path.isfile(absolute_results_csv_path) and os.path.isfile(relative_results_csv_path), \
        f"Could not find {absolute_results_csv_path} or {relative_results_csv_path}."

    visualise_results(visualisation_output_path, absolute_results_csv_path, relative_results_csv_path,
                      relative_to=baseline_model)


def get_nyu_loader(batch_size=64, csv_path='./data/nyu2_test.csv', ensure_multiple_of=1):
    transformed_testing = DepthDataset(csv_file=csv_path,
                                       transform=transforms.Compose([
                                           CenterCrop([.95, .95]),

                                           Resize(320, 240,
                                                  resize_target=True,
                                                  keep_aspect_ratio=True,
                                                  ensure_multiple_of=ensure_multiple_of,
                                                  resize_method="upper_bound",
                                                  image_interpolation_method=cv2.INTER_CUBIC,
                                                  ),
                                           ToTensor(is_test=True)
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=0, pin_memory=False)

    return dataloader_testing


def create_depth_maps(output_dir, model_loader, data_loader) -> Tuple[
    float, datetime.timedelta, datetime.timedelta]:
    """
    Load a depth estimation model to predict depth maps for a given dataset and write those results to disk.

    Args:
        output_dir: The directory to save the results to.
        model_loader: A function for loading the depth estimation model.
        data_loader: The dataloader for loading batches of samples from the dataset.

    Returns: 3-tuple containing the peak memory usage (GB), inference time (s), io time (s).
    """
    num_samples = len(data_loader.dataset)

    metadata_path = os.path.join(output_dir, 'metadata.json')
    raw_dir = os.path.join(output_dir, 'raw')
    png_dir = os.path.join(output_dir, 'png')

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    if len(os.listdir(raw_dir)) == num_samples and len(os.listdir(png_dir)) == num_samples and os.path.isfile(
            metadata_path):
        print(f"Found cached results.")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        peak_memory_usage = metadata['peak_memory_usage']
        inference_time = datetime.timedelta(seconds=metadata['inference_time'])
        io_time = datetime.timedelta(seconds=metadata['io_time'])
    else:
        inference_timer = Timer()
        io_time = datetime.timedelta()
        torch.cuda.reset_max_memory_allocated()

        model = model_loader()
        model.eval()

        with inference_timer:
            epoch_progress = 0

            with torch.no_grad():
                for i, sample_batched in enumerate(data_loader):
                    images = sample_batched['image']
                    images = images.cuda()

                    outputs = model(images)

                    if len(outputs.shape) == 3:
                        outputs = outputs.unsqueeze(dim=1)

                    outputs = torch.nn.functional.interpolate(outputs,
                                                              size=[images.size(2), images.size(3)],
                                                              mode='bilinear',
                                                              align_corners=True)

                    # The `.tiny` is here to ensure that the flat depth maps do no result in all NaNs.
                    normalised_outputs = (outputs - outputs.min()) / (
                                outputs.max() - outputs.min() + torch.finfo(outputs.dtype).tiny)
                    normalised_outputs = normalised_outputs.cpu().squeeze().numpy()
                    outputs = outputs.cpu().squeeze().numpy()

                    io_timer = Timer()

                    with io_timer:
                        for depth_map, normalised_depth_map in zip(outputs, normalised_outputs):
                            raw_path = os.path.join(raw_dir, f"{epoch_progress:06d}.raw")
                            png_path = os.path.join(png_dir, f"{epoch_progress:06d}.png")
                            image_io.save_image(raw_path, depth_map)
                            image_io.save_image(png_path, normalised_depth_map)

                            epoch_progress += 1

                    io_time += io_timer.elapsed

                    print(f"\rProgress: [{epoch_progress:02d}/{num_samples :02d}] {inference_timer.elapsed}",
                          end="")

                print()

        peak_memory_usage = torch.cuda.max_memory_allocated() / 1e9
        inference_time = inference_timer.elapsed

        metadata = {
            'peak_memory_usage': peak_memory_usage,
            'inference_time': inference_time.total_seconds(),
            'io_time': io_time.total_seconds()
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    return peak_memory_usage, inference_time, io_time


def create_noisy_depth_maps(output_path, labels_loader):
    def overlay_noise(image, **perlin_kwargs):
        height, width = image.shape[:2]

        noise = utils.perlin(width, height, **perlin_kwargs)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = 255 * noise
        noise = np.expand_dims(noise, -1)

        new_image = image.astype(np.float) + noise
        new_image = new_image / new_image.max()
        new_image = (255 * new_image).astype(np.uint8)

        return new_image

    noisy_depth_map_timer = Timer()

    with noisy_depth_map_timer:
        os.makedirs(output_path, exist_ok=True)

        if len(os.listdir(output_path)) == len(labels_loader.dataset):
            print(f"Found cached results, skipping...")
            return noisy_depth_map_timer.elapsed

        np.random.seed(42)

        for i in range(len(labels_loader.dataset)):
            depth_map = labels_loader.dataset[i]['depth']
            depth_map = depth_map.permute((1, 2, 0))
            depth_map = overlay_noise(depth_map.cpu().numpy(), scale=32)
            depth_map = overlay_noise(depth_map, scale=16)
            depth_map = overlay_noise(depth_map, scale=8)

            depth_map_path = os.path.join(output_path, f"{i:06d}.png")
            Image.fromarray(depth_map.squeeze()).save(depth_map_path)

            print(
                f"\rProgress: [{i:03,d}/{len(labels_loader.dataset):03,d}] - Elapsed Time: {noisy_depth_map_timer.elapsed}",
                end="")

        print()
        np.random.seed(None)

    return noisy_depth_map_timer.elapsed


def create_rendered_images(output_dir, image_loader, depth_loader=None, fps=60,
                           mesh_density=8, displacement_factor=4.0):
    timer = Timer()
    timer.start()
    num_frames = len(image_loader.dataset)

    os.makedirs(output_dir, exist_ok=True)

    video_output_dir = os.path.join(output_dir, 'video')
    os.makedirs(video_output_dir, exist_ok=True)

    frame_output_dir = os.path.join(output_dir, 'image')
    os.makedirs(frame_output_dir, exist_ok=True)

    if len(os.listdir(video_output_dir)) == num_frames and len(os.listdir(frame_output_dir)) == num_frames:
        print("Found cached results, skipping...")

        return timer.elapsed

    height, width = image_loader.dataset[0]['image'].shape[-2:]
    camera_position = utils.get_translation_matrix(dz=-10)
    camera = render.Camera(window_size=(width, height), fov_y=18)
    default_shader = render.ShaderProgram(
        vertex_shader_path='third_party/DepthRenderer/DepthRenderer/shaders/shader.vert',
        fragment_shader_path='third_party/DepthRenderer/DepthRenderer/shaders/shader.frag')
    renderer = render.MeshRenderer(default_shader_program=default_shader,
                                   fps=fps, camera=camera,
                                   unlimited_frame_works=True)

    rotation_angle = 2.5
    loops_per_second = 0.5 / rotation_angle

    camera_animation = animation.Compose([
        animation.RotateAxisBounce(np.deg2rad(rotation_angle), axis=utils.Axis.Y, offset=0.5, speed=-loops_per_second),
        animation.RotateAxisBounce(np.deg2rad(rotation_angle / 5.0), axis=utils.Axis.X, offset=0.5,
                                   speed=-loops_per_second),
        animation.Translate(distance=0.30, speed=loops_per_second),
        animation.Translate(distance=0.15, axis=utils.Axis.Y, offset=0.25, speed=loops_per_second)
    ])

    initial_delay = 3
    animation_length_in_frames = fps / loops_per_second
    image_writer = utils.AsyncImageWriter()

    write_frame = utils.RecurringTask(image_writer.write, frequency=fps)

    def data_loader():
        if depth_loader is None:
            for i, batch in enumerate(image_loader):
                images_ = batch['image']
                depth_maps_ = batch['depth']

                yield images_, depth_maps_
        else:
            for i, (images_, depth_maps_) in enumerate(zip(image_loader, depth_loader)):
                if isinstance(images_, dict):
                    images_ = images_['image']

                if isinstance(depth_maps_, dict):
                    depth_maps_ = depth_maps_['depth']

                yield images_, depth_maps_

    def generate_samples():
        i = 0

        for images, depth_maps in data_loader():
            N, C, H, W = images.shape
            images = images.permute((0, 2, 3, 1))  # NCHW -> NHWC format
            images = (255 * images).to(torch.uint8)
            alpha = 255 * torch.ones(N, H, W, 1, dtype=torch.uint8)
            images = torch.cat((images, alpha), dim=3)
            images = images.flip(dims=(1,))
            images = images.cpu().numpy()

            depth_maps = depth_maps.cuda()

            if len(depth_maps.shape) == 3:
                depth_maps = depth_maps.unsqueeze(1)

            depth_maps = torch.nn.functional.interpolate(depth_maps.to(torch.float),
                                                         size=(H, W), mode='bicubic', align_corners=True)
            depth_maps = depth_maps.permute((0, 2, 3, 1))
            depth_maps = depth_maps.flip(dims=(1,))
            depth_maps = depth_maps.cpu().numpy()

            for image, depth in zip(images, depth_maps):
                depth = (depth - depth.min()) / (depth.max() - depth.min() + np.finfo(depth.dtype).tiny)
                depth = (255 * depth).astype(np.uint8)
                yield i, image, depth
                i += 1

    class Context:
        def __init__(self, i, image, depth_map):
            self.i = i
            self.image = image
            self.depth_map = depth_map

            self.texture = render.Texture(image)
            self.mesh = render.Mesh.from_texture(self.texture, depth_map, density=mesh_density)
            self.mesh.vertices[:, 2] *= displacement_factor

            self.video_writer = utils.AsyncVideoWriter(
                os.path.join(video_output_dir, f"{self.i:06d}.avi"),
                size=renderer.frame_buffer_shape, fps=fps)
            self.frame_output_path = os.path.join(frame_output_dir, f"{self.i:06d}")
            os.makedirs(self.frame_output_path, exist_ok=True)

        def cleanup(self):
            if self.video_writer is not None:
                # NOTE: This may be slow depending on how many frames are waiting to be written to disk.
                self.video_writer.cleanup()

            if self.texture is not None:
                self.texture.cleanup()

            if self.mesh is not None:
                self.mesh.cleanup()

    class ContextSwitcher:
        def __init__(self, samples):
            self.samples_generator = samples

            self.current_context: Optional[Context] = None

        def next_context(self):
            self.cleanup()

            i, image, depth_map = next(self.samples_generator)
            context = Context(i, image, depth_map)

            self.current_context = context

            return self.current_context

        def cleanup(self):
            if self.current_context is not None:
                self.current_context.cleanup()
                self.current_context = None

    context_switcher = ContextSwitcher(generate_samples())

    def record_data_func():
        frame = renderer.get_frame()

        if frame is None:
            return

        context = context_switcher.current_context

        if context is not None:
            image_output_path = os.path.join(context.frame_output_path, f"{write_frame.call_count:06d}.png")

            write_frame(frame, image_output_path)
            video_writer = context.video_writer

            if video_writer is not None:
                video_writer.write(frame)

    record_data = utils.DelayedTask(record_data_func, delay=initial_delay)

    def next_mesh_func():
        context = context_switcher.next_context()

        renderer.mesh = context.mesh
        camera_animation.reset()
        record_data.reset()

    next_mesh = utils.RecurringTask(next_mesh_func, frequency=animation_length_in_frames + initial_delay + 1)

    def update_callback(delta):
        try:
            next_mesh()
            print(f"\rProgress: [{context_switcher.current_context.i:02d}/{num_frames:02d}] {timer.elapsed}", end="")
        except StopIteration:
            renderer.close()
            print()
            return

        camera_animation.update(delta)
        camera.view = camera_position @ camera_animation.transform
        record_data()

    def exit_callback():
        context_switcher.cleanup()
        default_shader.cleanup()
        renderer.cleanup()

    renderer.on_update = update_callback
    renderer.on_exit = exit_callback

    with timer:
        renderer.run()

    return timer.elapsed


def visualise_results(output_path, absolute_results_csv_path, relative_results_csv_path, relative_to):
    # TODO: Fix title not being fully visible.
    # TODO: Fix x axis label not being visible.
    # TODO: Make model names and series name in legend more human friendly (snake case to capitalised, space-separated words)
    df_abs = pd.read_csv(absolute_results_csv_path, index_col=0)
    df_rel = pd.read_csv(relative_results_csv_path, index_col=0)

    (1000 * df_abs['frame_time']).sort_values(ascending=False).plot(kind='barh',
                                                                    title='Frame Time During Inference of Different Models (Lower is Better)',
                                                                    ylabel='Frame Time (ms)', xlabel='Model')
    baseline = 1000 * df_abs['frame_time'].loc[relative_to]
    plt.axvline(x=baseline, label='Baseline', color='black', linestyle='--')
    plt.axvline(x=1000 / 30, label='30 fps', color='orange', linestyle='--')
    plt.axvline(x=1000 / 60, label='60 fps', color='green', linestyle='--')
    plt.legend()
    plt.savefig(os.path.join(output_path, "frame_time.png"))
    plt.close()

    df_abs['peak_memory_usage'].sort_values(ascending=False).plot(kind='barh',
                                                                  title='Peak Memory Usage During Inference of Different Models (Lower is Better)',
                                                                  ylabel='Memory Usage (GB)', xlabel='Model')
    baseline = df_abs['peak_memory_usage'].loc[relative_to]
    plt.axvline(x=baseline, label='Baseline', color='black', linestyle='--')
    plt.legend()
    plt.savefig(os.path.join(output_path, "memory_usage.png"))
    plt.close()

    (1. + df_rel['abs_rel'].sort_values(ascending=False)).plot(kind='barh',
                                                               title='Relative Absolute Relative Error (ABS_REL) of Different Models (Lower is Better)',
                                                               ylabel='ABS_REL', xlabel='Model')
    plt.axvline(x=1.0, label='Baseline', color='black', linestyle='--')
    plt.legend()
    plt.savefig(os.path.join(output_path, "abs_rel.png"))
    plt.close()

    (1. + df_rel['delta1'].sort_values()).plot(kind='barh',
                                               title='Relative Thresholded Accuracy (DELTA1) of Different Models (Higher is Better)',
                                               ylabel='DELTA1', xlabel='Model')
    plt.axvline(x=1.0, label='Baseline', color='black', linestyle='--')
    plt.legend()
    plt.savefig(os.path.join(output_path, "delta1.png"))
    plt.close()

    df_abs['ssim'].sort_values().plot(kind='barh',
                                      title='Structural Similarity (SSIM) for Different Models (Higher is Better)',
                                      ylabel='SSIM', xlabel='Model')
    baseline = df_abs['ssim'].loc[relative_to]
    plt.axvline(x=baseline, label='Baseline', color='black', linestyle='--')
    plt.legend()
    plt.savefig(os.path.join(output_path, "ssim.png"))
    plt.close()

    df_abs['psnr'].sort_values().plot(kind='barh',
                                      title='Peak Signal to Noise Ratio (PSNR) for Different Models (Higher is Better)',
                                      ylabel='PSNR', xlabel='Model')
    baseline = df_abs['psnr'].loc[relative_to]
    plt.axvline(x=baseline, label='Baseline', color='black', linestyle='--')
    plt.legend()
    plt.savefig(os.path.join(output_path, "psnr.png"))
    plt.close()

    df_abs['lpips'].sort_values().plot(kind='barh',
                                       title='Perceptual Similarity (LPIPS) for Different Models (Higher is Better)',
                                       ylabel='LPIPS', xlabel='Model')
    baseline = df_abs['lpips'].loc[relative_to]
    plt.axvline(x=baseline, label='Baseline', color='black', linestyle='--')
    plt.legend()
    plt.savefig(os.path.join(output_path, "lpips.png"))
    plt.close()


def get_sample_output(model, data_loader, num_samples=8):
    samples = None

    for i, sample_batched in enumerate(data_loader):
        image, depth = sample_batched['image'], sample_batched['depth']

        depth = depth.cuda()
        image = image.cuda()

        output = model(image)

        if len(output.shape) == 3:
            output = output.unsqueeze(dim=1)

        output = torch.nn.functional.interpolate(output,
                                                 size=[depth.size(2), depth.size(3)],
                                                 mode='bilinear',
                                                 align_corners=True)

        if samples is None:
            samples = output.detach().cpu().numpy()
        else:
            samples = np.concatenate((samples, output.detach().cpu().numpy()), axis=0)

        if len(samples) >= num_samples:
            break

    samples = samples[:num_samples]

    samples = (samples - samples.min()) / (samples.max() - samples.min())

    return samples


def test(test_loader, estimated_depth_loader, cache_dir):
    processed_count = 0
    metrics_dict_path = os.path.join(cache_dir, 'standard_benchmark_metadata.json')
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.isfile(metrics_dict_path):
        print("Found cached results, skipping...")

        with open(metrics_dict_path, 'r') as f:
            metrics_dict = json.load(f)

        return metrics_dict

    metrics = MetricsTracker()

    with torch.no_grad():
        for i, (labels_batch, output_batch) in enumerate(zip(test_loader, estimated_depth_loader)):
            labels = labels_batch['depth']

            labels = labels.cuda()
            outputs = output_batch.cuda()
            # The predicted depth maps may be read in NWH format, whereas the dataset depth maps may be read in NCWH.
            # Make sure that both the labels and outputs are the same format.
            outputs = outputs.reshape(labels.shape)

            # Sometimes we are dealing with images in 8-bit space, but for the below normalisation to work correct
            # properly we need to explicitly convert the tensors to a floating point type.
            # outputs = (outputs.float() - outputs.min()) / (outputs.max() - outputs.min() + torch.finfo(torch.float32).tiny)
            # labels = (labels.float() - labels.min()) / (labels.max() - labels.min() + torch.finfo(torch.float32).tiny)

            metrics.update(outputs=outputs, labels=labels)
            processed_count += labels.shape[0]

            print(f"\rProgress: [{processed_count:02d}/{len(test_loader.dataset):02d}] {metrics}", end="")

        print()

    metrics_dict = metrics.to_dict()

    with open(metrics_dict_path, 'w') as f:
        json.dump(metrics_dict, f)

    return metrics_dict


def test_visual(labels_loader, outputs_loader, cache_dir):
    metrics_dict_path = os.path.join(cache_dir, 'visual_benchmark_metadata.json')
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.isfile(metrics_dict_path):
        print("Found cached results, skipping...")

        with open(metrics_dict_path, 'r') as f:
            metrics_dict = json.load(f)
    else:
        processed_count = 0
        num_frames = len(labels_loader.dataset)

        metrics = VisualMetricsTracker()

        with torch.no_grad():
            for i, (labels_batch, output_batch) in enumerate(zip(labels_loader, outputs_loader)):
                labels = labels_batch.cuda()
                outputs = output_batch.cuda()

                # Ensure outputs and labels do not contain an alpha num_channels since the benchmark does not use it or
                #  compatible with it.
                if outputs.shape[1] == 4:
                    outputs = outputs[:, :3, :, :]
                elif outputs.shape[3] == 4:
                    outputs = outputs[:, :, :, :3]

                if labels.shape[1] == 4:
                    labels = labels[:, :3, :, :]
                elif labels.shape[3] == 4:
                    labels = labels[:, :, :, :3]

                metrics.update(outputs=outputs, labels=labels)
                processed_count += labels_batch.shape[0]

                print(f"\rProgress: [{processed_count:02d}/{num_frames :02d}] {metrics}", end="")

            print()

        metrics_dict = metrics.to_dict()

        with open(metrics_dict_path, 'w') as f:
            json.dump(metrics_dict, f)

    return metrics_dict


def save_benchmark_results(args, benchmark_results, relative_to=None):
    def save_csv_and_tex(df, path):
        df.to_csv(f"{path}.csv")
        df.to_latex(f"{path}.tex", float_format="%.2f")

    for dataset in benchmark_results:
        df = pd.DataFrame.from_dict(benchmark_results[dataset], orient='index')
        df = df.drop('log10', axis='columns')

        path = os.path.join(args.output_path, dataset)
        save_csv_and_tex(df, path)

        if relative_to is not None and relative_to in df.index:
            path = os.path.join(args.output_path, f"{dataset}-relative")
            save_csv_and_tex((df - df.loc[relative_to]) / df.loc[relative_to], path)


def save_sample_output(model, model_name, data_loader, sample_output_base_dir):
    samples = get_sample_output(model, data_loader, num_samples=8)
    sample_output_dir = os.path.join(sample_output_base_dir, model_name, 'depth')
    os.makedirs(sample_output_dir, exist_ok=True)

    for i, sample in enumerate(samples.squeeze(1)):
        path = os.path.join(sample_output_dir, f"{i:06d}")
        image_io.save_image(f"{path}.png", sample)
        image_io.save_image(f"{path}.raw", sample)


def images_to_grid(dataset: DepthDataset, output_dir, output_type='depth', arrange_column=True, file_extension='.raw',
                   num_examples=8):
    """
    Create a single image that visualises the outputs of multiple depth estimation models and the corresponding inputs
     side by side from the output of the benchmark suite.

    Args:
        dataset: The dataset containing the input images.
        output_dir: The path to the folder containing the folders of the output images.
        output_type: The type of output to compare - either 'depth' or 'rendered_images'.
        arrange_column: Whether to place the inputs vertically in the first column (`arrange_column == True`) or
            horizontally in the first row (`arrange_column == False`).
        file_extension: The file extension of the output images. This is used to distinguish between copies of the
        outputs in different file formats that are saved to the same folder.

    Returns: The image containing all of the inputs and outputs for multiple models.
    """
    assert output_type in ('depth', 'rendered_images')

    target_height, target_width = dataset[0]['depth'].shape[-2:]

    def add_row_to_collage(paths, collage=None, reshape_to=None):
        row = None

        for image_path in paths:
            if file_extension == '.raw' and image_path.endswith('.raw'):
                img = image_io.load_raw_float32_image(image_path)
            else:
                img = Image.open(image_path)
                img = np.array(img)

            img = (img - img.min()) / (img.max() - img.min() + np.finfo(np.float32).tiny)
            img = 255 * img
            img = img.astype(np.uint8)

            if reshape_to is not None:
                img = cv2.resize(img, dsize=reshape_to, interpolation=cv2.INTER_LINEAR)

            if row is None:
                row = img
            else:
                row = np.concatenate((row, img), axis=1 if arrange_column else 0)

        if len(row.shape) == 3:
            row = cv2.cvtColor(row, cv2.COLOR_RGB2RGBA)
        elif len(row.shape) == 2 and len(collage.shape) == 3:
            row = cv2.cvtColor(row, cv2.COLOR_GRAY2RGBA)

        if collage is None:
            collage = row
        else:
            collage = np.concatenate((collage, row), axis=0 if arrange_column else 1)

        return collage

    collage = add_row_to_collage(dataset.image_paths[:num_examples], reshape_to=(target_width, target_height))

    if output_type == 'depth':
        gt_paths = dataset.depth_paths[:num_examples]
    else:
        gt_dir = os.path.join(output_dir, 'ground_truth', 'image')
        input_images = list(sorted(os.listdir(gt_dir)))
        input_images = input_images[:num_examples]
        gt_paths = []

        for input_image in input_images:
            output_images = list(sorted(os.listdir(os.path.join(gt_dir, input_image))))

            for output_image in output_images:
                gt_paths.append(os.path.join(gt_dir, input_image, output_image))
                break  # Only want the first image.

    collage = add_row_to_collage(gt_paths, collage, reshape_to=(target_width, target_height))

    model_names = filter(lambda x: os.path.isdir(os.path.join(output_dir, x)), os.listdir(output_dir))
    # TODO: Include noised depth maps? Would need to create raw depth map files.
    model_names = filter(lambda x: x != "ground_truth" and x != "random", model_names)
    model_names = list(sorted(model_names))
    assert len(model_names) > 0, f"Did not find any folders in {output_dir}."

    for model_name in model_names:
        model_dir = os.path.join(output_dir, model_name, output_type)

        if output_type == 'depth':
            if file_extension.endswith('.raw'):
                model_dir = os.path.join(model_dir, 'raw')
            else:
                model_dir = os.path.join(model_dir, 'png')

            frame_filenames = filter(lambda filename: filename.endswith(file_extension), os.listdir(model_dir))
            frame_filenames = list(sorted(frame_filenames))
            frame_filenames = frame_filenames[:num_examples]
            assert len(frame_filenames) == num_examples

            frame_paths = [os.path.join(model_dir, filename) for filename in frame_filenames]
        else:
            model_dir = os.path.join(model_dir, 'image')
            input_images = list(sorted(os.listdir(model_dir)))
            input_images = input_images[:num_examples]
            frame_paths = []

            for input_image in input_images:
                output_images = list(sorted(os.listdir(os.path.join(model_dir, input_image))))

                for output_image in output_images:
                    frame_paths.append(os.path.join(model_dir, input_image, output_image))
                    break  # Only want the first image.
            assert len(frame_paths) == num_examples

        collage = add_row_to_collage(frame_paths, collage, reshape_to=(target_width, target_height))

    return collage


class VisualMetricsTracker:
    def __init__(self):
        self.ssim = AverageMeter()
        self.psnr = AverageMeter()
        self.lpips = AverageMeter()
        # Mean Image Feature Distance (MIFD)
        self.mifd = AverageMeter()

        self.lpips_fn = lpips.LPIPS(net='alex')

        if torch.cuda.is_available():
            self.lpips_fn = self.lpips_fn.cuda()

    def __getitem__(self, item):
        self.__getattribute__(item.lower())

    def to_dict(self):
        result = dict()

        for key, metric in self.__dict__.items():
            if isinstance(metric, AverageMeter):
                result[key] = metric.value

        return result

    @staticmethod
    def calculate_reprojection_error(label, output, ratio_threshold=0.7, k=2, min_matches=1, log_residual=False):
        img1 = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)

        # -- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        detector = cv2.xfeatures2d_SIFT.create()
        key_points1, descriptors1 = detector.detectAndCompute(img1, None)
        key_points2, descriptors2 = detector.detectAndCompute(img2, None)
        # -- Step 2: Matching descriptor vectors with a FLANN based matcher
        if descriptors1 is None or descriptors2 is None:
            warnings.warn(
                f"Could not extract any features for at least one image in the pair.")
            return float('nan')

        if len(descriptors1) < k or len(descriptors2) < k:
            warnings.warn(
                f"Not enough descriptors for k={k:d}, only got {len(descriptors1):,d} and {len(descriptors2):,d}.")
            return float('nan')

        # Since SURF is a floating-point descriptor NORM_L2 is used
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(descriptors1, descriptors2, k)
        # -- Filter matched key points using the Lowe's ratio test
        points1 = []
        points2 = []

        for m, n in knn_matches:
            if m.distance < ratio_threshold * n.distance:
                points1.append(key_points1[m.queryIdx].pt)
                points2.append(key_points2[m.trainIdx].pt)

        if len(points1) < min_matches:
            warnings.warn(f"Not enough matches for `min_matches={min_matches}`, only got {len(points1)}.")
            return float('nan')

        if log_residual:
            residuals = np.log10(points1) - np.log10(points2)
        else:
            residuals = np.asarray(points1) - np.asarray(points2)

        try:
            return np.mean(np.sqrt(np.sum(np.square(residuals), axis=1)))
        except np.AxisError:  # No matches means axis 1 will be out of bounds.
            return float('nan')

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        Update the running averages.

        :param outputs: The neural network predictions.
        :param labels: The ground truth depth maps
        """
        outputs_, labels_ = outputs.detach(), labels.detach()
        outputs_np = outputs_.cpu().numpy()
        labels_np = labels_.cpu().numpy()

        for label, output in zip(labels_np, outputs_np):
            self.mifd.update(self.calculate_reprojection_error(label, output))

        labels_batch_size = labels_.shape[0]
        outputs_batch_size = outputs_.shape[0]

        assert outputs_batch_size == labels_batch_size, \
            f"Batch sizes for labels and predictions do not match, " \
            f"got {labels_batch_size} and {outputs_batch_size} respectively."

        if not labels_.shape[1] == 3 or not outputs_.shape[1] == 3:
            if labels_.shape[3] == 3 or outputs_.shape[3] == 3:
                labels_ = labels_.permute((0, 3, 1, 2))
                outputs_ = outputs_.permute((0, 3, 1, 2))

                observed_format = 'NHWC'
            else:
                raise RuntimeError(f"Unrecognised format: Expected NCHW or NHWC where C=3, but got {labels_.shape}.")

            warnings.warn(f"Expected labels and outputs to be in NCHW, got {observed_format}. "
                          f"Reshaped inputs to {labels_.shape}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lpips_labels = labels_.to(device)
        lpips_outputs = outputs_.to(device)

        def normalise(tensor, low=0.0, high=1.0, dtype=torch.float32):
            range = abs(high - low)
            normalised_tensor = (tensor.to(dtype) - tensor.min()) / (tensor.max() - tensor.min())
            normalised_tensor = range * normalised_tensor + low

            return normalised_tensor

        normalised_labels = normalise(labels_.to(device))
        normalised_outputs = normalise(outputs_.to(device))
        lpips_labels = normalise(lpips_labels, low=-1.0, high=1.0)
        lpips_outputs = normalise(lpips_outputs, low=-1.0, high=1.0)

        self.lpips.update(self.lpips_fn(lpips_labels, lpips_outputs).mean().item())
        self.ssim.update(ssim(normalised_labels, normalised_outputs).item())
        self.psnr.update(psnr(normalised_labels, normalised_outputs).item())

    def __str__(self):
        return f"SSIM: {self.ssim:.3f} - PSNR: {self.psnr:.3f} - LPIPS: {self.lpips:.3f} - Reproj.: {self.mifd:.3f}      "


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.output_path = "benchmark"

    args.renderer_fps = 60

    args.reside_senet_path = "checkpoints/reside_original/model_senet"
    args.reside_resnet_path = "checkpoints/reside_original/model_resnet"

    args.reside_enb0_path = "checkpoints/reside/efficientnet-b0-hu.pth"
    args.reside_enb4_path = "checkpoints/reside/efficientnet-b4-hu.pth"
    args.reside_resnet50_path = "checkpoints/reside/resnet50-hu.pth"

    args.midas_enb0_path = "checkpoints/reside/efficientnet-b0-lasinger.pth"
    args.midas_enb4_path = "checkpoints/reside/efficientnet-b4-lasinger.pth"
    args.midas_resnet50_path = "checkpoints/reside/resnet50-lasinger.pth"

    args.csv_path = "data/datasets/nyuv2/nyu2_test.csv"
    args.batch_size = 4

    main(args)
