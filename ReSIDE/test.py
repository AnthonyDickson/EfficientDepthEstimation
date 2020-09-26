import math

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
    model = define_model(is_resnet=True)
    model = torch.nn.DataParallel(model).cuda()
    state_dict = torch.load('pretrained_model/model_resnet')
    # state_dict = torch.load('checkpoint.pth.tar')['state_dict']
    # state_dict = {f"module.{key}": value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)

    # model = MidasNet.load(r"checkpoints/bestnet-resnet50.pth").cuda()

    test_loader = loaddata.getTestingData(1)
    test(test_loader, model, 0.25)


def test(test_loader, model, thre):
    model.eval()

    totalNumber = 0

    Ae = 0
    Pe = 0
    Re = 0
    Fe = 0

    metrics = MetricsTracker()

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

        depth_edge = edge_detection(depth)
        output_edge = edge_detection(output)

        totalNumber += image.shape[0]

        edge1_valid = (depth_edge > thre)
        edge2_valid = (output_edge > thre)

        nvalid = np.sum(torch.eq(edge1_valid, edge2_valid).float().data.cpu().numpy())
        A = nvalid / (depth.size(2) * depth.size(3))

        nvalid2 = np.sum(((edge1_valid + edge2_valid) == 2).float().data.cpu().numpy())
        P = nvalid2 / (np.sum(edge2_valid.data.cpu().numpy()))
        R = nvalid2 / (np.sum(edge1_valid.data.cpu().numpy()))

        F = (2 * P * R) / (P + R)

        Ae += A
        Pe += P
        Re += R
        Fe += F

        print(f"\rProgress: [{totalNumber:02d}/{len(test_loader.dataset):02d}] {metrics}", end="")

    print()

    Av = Ae / totalNumber
    Pv = Pe / totalNumber
    Rv = Re / totalNumber
    Fv = Fe / totalNumber
    print('AV', Av)
    print('PV', Pv)
    print('RV', Rv)
    print('FV', Fv)

    return metrics


def edge_detection(depth):
    get_edge = sobel.Sobel().cuda()

    edge_xy = get_edge(depth)
    edge_sobel = torch.pow(edge_xy[:, 0, :, :], 2) + \
                 torch.pow(edge_xy[:, 1, :, :], 2)
    edge_sobel = torch.sqrt(edge_sobel)

    return edge_sobel


if __name__ == '__main__':
    main()
