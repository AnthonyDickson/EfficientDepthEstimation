# This script runs all models in the folder 'checkpoints' and outputs the raw and colorized depth maps to examples/nyu.
import os
from pathlib import Path

import imageio.v3
from matplotlib.image import imsave
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
from tqdm import tqdm

from DepthRenderer.DepthRenderer.utils import AsyncImageWriter as AsyncImageWriterBase
from ReSIDE import loaddata
from ReSIDE.models.lasinger2019 import MidasNet
from ReSIDE.train import define_model


class AsyncImageWriter(AsyncImageWriterBase):
    def write(self, frame, path, *args, **kwargs):
        self.pool.apply_async(imageio.v3.imwrite, (f"{path}.png", frame))
        self.pool.apply_async(imsave, (f"{path}.jpg", frame.astype(float) / 10_000.0))  # normalise s.t. max depth = 1.0

def load_model(checkpoint_dir, checkpoint_file):
    print(f"Loading checkpoint {checkpoint_file}...")
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
        return model, decoder_name, encoder_name
    else:
        return MidasNet.load(checkpoint_path), decoder_name, encoder_name

def main():
    checkpoint_dir = os.path.abspath("checkpoints")
    checkpoints = os.listdir(checkpoint_dir)

    examples_dir = os.path.join("examples", "nyu")

    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)

    test_loader = loaddata.getTestingData(4)

    image_writer = AsyncImageWriter()

    for checkpoint_file in checkpoints:
        model, decoder_name, encoder_name = load_model(checkpoint_dir, checkpoint_file)
        model = model.cuda().eval()

        print(f"Estimating depth maps for {encoder_name}-{decoder_name}...")
        image_index = 0
        output_path = os.path.join(examples_dir, f"{encoder_name}-{decoder_name}")
        os.makedirs(output_path, exist_ok=True)

        with torch.no_grad():
            for batch in tqdm(test_loader):
                outputs = model(batch['image'].cuda())
                outputs = F.interpolate(outputs, size=(480, 640), mode='bilinear', align_corners=True)
                outputs = outputs.squeeze().cpu().numpy()
                outputs[outputs > 10.0] = 0.0  # Cap at 10m
                outputs = (outputs * 1000.0).astype(np.uint16)

                for depth_map in outputs:
                    path = os.path.join(output_path, f"{image_index:06d}")
                    image_writer.write(depth_map, path)

                    image_index += 1

    print(f"Waiting for images to finish writing...")
    image_writer.cleanup()


if __name__ == '__main__':
    main()
