import os

import matplotlib.image
import matplotlib.pyplot as plt
import plac
import torch
import torch.nn.parallel

from ReSIDE import loaddata_demo as loaddata
from ReSIDE.models import modules, resnet, densenet, net, senet

plt.set_cmap("gray")


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained=True)
        Encoder = modules.E_resnet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel=[192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])

    return model


@plac.annotations(
    image_path=plac.Annotation('The path to an RGB image or a directory containing RGB images.', type=str, kind='option', abbrev='i'),
    model_path=plac.Annotation('The path to the pre-trained model weights.', type=str, kind='option', abbrev='m'),
    output_path=plac.Annotation('The path to save the model output to.', type=str, kind='option', abbrev='o'),
)
def main(image_path, model_path='pretrained_model/model_resnet', output_path=None):
    print("Loading model...")

    is_resnet = 'resnet' in model_path.lower()
    is_densenet = 'densenet' in model_path.lower()
    is_senet = 'senet' in model_path.lower()

    model = define_model(is_resnet=is_resnet, is_densenet=is_densenet, is_senet=is_senet)
    model = torch.nn.DataParallel(model).cuda()
    state_dict = torch.load(os.path.abspath(model_path))
    model.load_state_dict(state_dict)
    model.eval()

    print("Creating depth maps...")
    rgb_path = os.path.abspath(image_path)

    if os.path.isdir(rgb_path):
        for file in os.listdir(rgb_path):
            test(model, os.path.join(rgb_path, file), output_path)
    else:
        test(model, rgb_path, output_path)

    print("Done.")


def test(model, rgb_path, output_path):
    nyu2_loader = loaddata.readNyu2(rgb_path)

    path, file = os.path.split(rgb_path)
    file = f"{file.split('.')[0]}.png"
    depth_path = os.path.join(output_path, file) if output_path else os.path.join(path, f"out_{file}")

    print(f"{rgb_path} -> {depth_path}")

    for i, image in enumerate(nyu2_loader):
        image = image.cuda()
        out = model(image)

        matplotlib.image.imsave(depth_path, out.view(out.size(2), out.size(3)).data.cpu().numpy())


if __name__ == '__main__':
    plac.call(main)
