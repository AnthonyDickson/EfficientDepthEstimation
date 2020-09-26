import math
import os

import torch
import torch.nn as nn
import torch.nn.parallel

from ReSIDE.models import modules, resnet, densenet, net, senet
import numpy as np
from ReSIDE import sobel, loaddata, util
from ReSIDE.models.lasinger2019 import MidasNet
from ReSIDE.train import define_model
from ReSIDE.util import MetricsTracker


def main():
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()

    # model = define_model(is_resnet=True).cuda()
    # state_dict = torch.load('checkpoints/resnet50.pth')
    # # state_dict = torch.load('checkpoint.pth.tar')['state_dict']
    # # state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    #
    # model.load_state_dict(state_dict)

    model = MidasNet.load("checkpoints/resnet50.pth").cuda()

    test_loader = loaddata.getTestingData(8)
    test(test_loader, model)

    peak_memory_usage = torch.cuda.max_memory_cached()

    print(peak_memory_usage,)


def test(test_loader, model):
    model.eval()

    totalNumber = 0

    for i, sample_batched in enumerate(test_loader):
        image, depth = sample_batched['image'], sample_batched['depth']

        image = image.cuda()
        model(image)

        print(f"\rProgress: [{totalNumber:02d}/{len(test_loader.dataset):02d}]", end="")

    print()


if __name__ == '__main__':
    main()
