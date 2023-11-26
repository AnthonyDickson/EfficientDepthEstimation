import warnings

import torch

from Benchmark.depth_model import DepthModel
from ReSIDE.models.lasinger2019 import MidasNet


class MidasModel(DepthModel):
    # Requirements and default settings
    align = 1
    learning_rate = 0.0001
    lambda_view_baseline = 0.0001

    def __init__(self, support_cpu: bool = False, pretrained: bool = True,
                 model_path="checkpoints/model_densenet"):
        super().__init__()

        if support_cpu:
            # Allow the model to run on CPU when GPU is not available.
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            # Rather raise an error when GPU is not available.
            self.device = torch.device("cuda")

        # Load pretrained checkpoint
        self.model = MidasNet.load(model_path)

        if not pretrained:
            warnings.warn(f"This custom Midas model does not support untrained models.")

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)

        self.norm_mean = torch.Tensor(
            [0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        self.norm_stdev = torch.Tensor(
            [0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

    def estimate_depth(self, images):
        # Reshape ...CHW -> XCHW
        shape = images.shape
        C, H, W = shape[-3:]
        input_ = images.reshape(-1, C, H, W).to(self.device)

        input_ = (input_ - self.norm_mean.to(self.device)) / \
                 self.norm_stdev.to(self.device)

        output = self.model(input_)

        # Reshape X1HW -> BNHW
        depth = output.reshape(shape[:-3] + output.shape[-2:])

        return depth

    def save(self, file_name):
        state_dict = self.model.state_dict()
        torch.save(state_dict, file_name)
