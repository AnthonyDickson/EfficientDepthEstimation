import os
from pathlib import Path

import cv2
import numpy as np
import plac
import torch
import torch.nn.parallel
import torch.nn.parallel
from PIL import Image
from torchvision import transforms

from ReSIDE.demo_transform import Scale, CenterCrop, ToTensor, Normalize
from ReSIDE.models.lasinger2019 import MidasNet
from ReSIDE.train import define_model


@plac.annotations(
    image_path=plac.Annotation('The path to an RGB image or a directory containing RGB images.', type=str,
                               kind='option', abbrev='i'),
    checkpoint_path=plac.Annotation('The path to the folder containing the trained model weights.', type=str,
                                    kind='option', abbrev='c'),
    output_path=plac.Annotation('The path to save the model output to.', type=str, kind='option', abbrev='o'),
)
def main(image_path, checkpoint_path='checkpoints', output_path="output"):
    os.makedirs(output_path, exist_ok=True)
    create_video(image_path, os.path.join(checkpoint_path, "resnet50-lasinger.pth"), output_path)

    # for model_name in sorted(os.listdir(checkpoint_path)):
    #     create_video(image_path, os.path.join(checkpoint_path, model_name), output_path)


def create_video(image_path, model_path, output_path):
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

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    # TODO: Detect input dimensions automatically.
    width, height = 1920, 1440
    # TODO: Add options defining border sizes or the aspect ratio of the original, unpadded input.
    # The size of the border (pillar/letterboxing) created from resizing a 16:9 video to 4:3 (in pixels).
    border_width = 180
    crop_rect = (0, border_width, width, height - border_width)

    # TODO: Make input size configurable via arguments.
    transform = transforms.Compose([
        Scale([640, 480]),
        CenterCrop([int(640 * 0.95), int(480 * 0.95)]),
        ToTensor(),
        Normalize(__imagenet_stats['mean'],
                  __imagenet_stats['std'])
    ])

    output_video_path = os.path.join(output_path, f"{checkpoint_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    video = cv2.VideoWriter(output_video_path, fourcc, 24.0, (2 * width, height - 2 * border_width))

    print(output_video_path, video)

    with torch.no_grad():
        for i, image_file in enumerate(sorted(os.listdir(image_path))):
            print(f"Frame {i + 1:03d}")
            raw_img = Image.open(os.path.join(image_path, image_file))
            img = transform(raw_img).unsqueeze(0) / 255.0
            depth = model(img.cuda())

            depth = torch.nn.functional.interpolate(depth, size=(height, width),
                                                    mode='bilinear', align_corners=True)

            color = raw_img.resize((width, height), Image.ANTIALIAS)
            color = color.crop(crop_rect)
            color = np.asarray(color)
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            depth = depth.cpu().squeeze(0).numpy()
            depth = (255.0 / (1.0 + depth)).astype(np.uint8)

            assert depth.max() <= 255
            depth = np.concatenate(3 * [depth], axis=0)
            depth = depth.transpose((1, 2, 0))
            depth = depth[border_width:height - border_width, :, :]

            stacked_frame = np.hstack((color, depth))

            assert stacked_frame.shape[0] == (height - 2 * border_width) and stacked_frame.shape[1] == 2 * width
            video.write(stacked_frame)

    video.release()


if __name__ == '__main__':
    plac.call(main)
