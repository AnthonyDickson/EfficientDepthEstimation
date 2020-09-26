import math
import os
from pathlib import Path

import plac
import torch
import torch.nn as nn
import torch.nn.parallel
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ReSIDE.colmap_io import read_model
from ReSIDE.models import modules, resnet, densenet, net, senet
import numpy as np
from ReSIDE import sobel, loaddata, util
from ReSIDE.models.lasinger2019 import MidasNet
from ReSIDE.demo_transform import Scale, CenterCrop, ToTensor, Normalize
from ReSIDE.train import define_model
from ReSIDE.util import MetricsTracker

from open3d.open3d_pybind.camera import PinholeCameraIntrinsic
from open3d.open3d_pybind.geometry import RGBDImage, PointCloud, TriangleMesh, Image
from open3d.open3d_pybind.io import write_point_cloud, write_triangle_mesh

@plac.annotations(
    images_path=plac.Annotation(
        'The path to the source video frames.',
        type=str, kind='option', abbrev='i'
    ),
    model_path=plac.Annotation(
        'The path to the model checkpoint.',
        type=str, kind='option', abbrev='m'
    ),
    output_path=plac.Annotation(
        'The path to save the point clouds to.',
        type=str, kind='option', abbrev='o'
    ),
    mirror_z_axis=plac.Annotation(
        "Flag indicating whether to mirror the depth map along the Z axis.",
        type=bool, kind="flag"
    ),
)
def main(images_path, model_path, output_path, mirror_z_axis):
    filename = os.path.basename(model_path)
    checkpoint_name = Path(filename).stem
    parts = checkpoint_name.split("-")

    if len(parts) == 2:
        encoder_name, decoder_name = parts
    else:  # len(parts) == 3
        encoder_name = '-'.join(parts[:2])
        decoder_name = parts[2]

    print(model_path)

    os.makedirs(output_path, exist_ok=True)

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

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    transform = transforms.Compose([
        Scale([320, 240]),
        CenterCrop([304, 228]),
        ToTensor(),
        Normalize(__imagenet_stats['mean'],
                  __imagenet_stats['std'])
    ])

    mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]))
    std = torch.from_numpy(np.array([0.229, 0.224, 0.225]))

    def denormalise(tensor):
        """Undo the normalisation applied to images."""
        tensor = tensor.clone()

        tensor.mul_(std[None, :, None, None]).add_(mean[None, :, None, None])

        return tensor

    f = 5.2921508098293293e+02 / 2
    cx = 3.2894272028759258e+02 / 2
    cy = 2.6748068171871557e+02 / 2
    camera_intrinsics = PinholeCameraIntrinsic(width=640 // 2, height=480 // 2, fx=f, fy=f, cx=cx, cy=cy)


    with torch.no_grad():
        for i, image_file in enumerate(sorted(os.listdir(images_path))):
            raw_img = PILImage.open(os.path.join(images_path, image_file))
            img = transform(raw_img).unsqueeze(0)
            depth = model(img.cuda())

            height, width = img.shape[-2:]
            depth = torch.nn.functional.interpolate(depth, size=(height, width), mode='bilinear', align_corners=True)

            color = Image(
                np.ascontiguousarray((255 * denormalise(img).squeeze().permute((1, 2, 0)).cpu().numpy()).astype(np.uint8))
            ).flip_vertical()
            depth = Image(depth.squeeze().cpu().numpy()).flip_vertical()

            rgbd = RGBDImage.create_from_color_and_depth(color=color,
                                                         depth=depth,
                                                         depth_scale=1, depth_trunc=float('inf'),
                                                         convert_rgb_to_intensity=False)

            point_cloud = PointCloud.create_from_rgbd_image(image=rgbd, intrinsic=camera_intrinsics)

            if mirror_z_axis:
                reflection_along_z_axis = np.array([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ])
                point_cloud = point_cloud.transform(reflection_along_z_axis)

            point_cloud_filename = "{:04d}.ply".format(i)
            point_cloud_path = os.path.join(output_path, point_cloud_filename)
            write_point_cloud(filename=point_cloud_path, pointcloud=point_cloud)
            print("Wrote frame {:d} to {}".format(i + 1, point_cloud_path))

            if i >= 60:
                break


if __name__ == '__main__':
    plac.call(main)
