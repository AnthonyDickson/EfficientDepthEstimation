import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils import model_zoo
from torchvision import utils

from ReSIDE.models import densenet
from ReSIDE.models import modules
from ReSIDE.models import resnet
from ReSIDE.models import senet


class model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):
        super(model, self).__init__()

        self.E = Encoder
        self.D = modules.D(num_features)
        self.MFF = modules.MFF(block_channel)
        self.R = modules.R(block_channel)

    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4, [x_decoder.size(2), x_decoder.size(3)])
        out = self.R(torch.cat((x_decoder, x_mff), 1))

        return out

    @property
    def num_parameters(self):
        """
        :return: The total number of parameters in the model and any sub-modules.
        """
        return sum(params.nelement() for params in self.parameters())
