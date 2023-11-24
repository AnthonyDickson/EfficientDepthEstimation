import torch

from utils.url_helpers import get_model_from_url

from benchmark.depth_model import DepthModel
from ReSIDE.models.modules import E_resnet, E_densenet, E_senet, E_efficientnet
from ReSIDE.train import define_model


class ReSIDEModel(DepthModel):
    # Requirements and default settings
    align = 1
    learning_rate = 0.0001
    lambda_view_baseline = 0.0001

    def __init__(self, support_cpu: bool = False, pretrained: bool = True,
                 model_path="checkpoints/model_densenet",
                 encoder_type=E_resnet, efficientnet_variant='efficientnet-b0'):
        super().__init__()

        if support_cpu:
            # Allow the model to run on CPU when GPU is not available.
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            # Rather raise an error when GPU is not available.
            self.device = torch.device("cuda")

        self.model = define_model(is_resnet=encoder_type == E_resnet,
                                  is_densenet=encoder_type == E_densenet,
                                  is_senet=encoder_type == E_senet,
                                  is_efficientnet=encoder_type == E_efficientnet,
                                  efficientnet_variant=efficientnet_variant)

        # Load pretrained checkpoint
        if pretrained:
            state_dict = torch.load(model_path)

            if encoder_type != E_efficientnet:
                state_dict = {f"{key.replace('module.', '')}": value for key, value in state_dict.items()}

            if encoder_type == E_senet:
                state_dict = {f"{key.replace('se_', 'se_module.')}": value for key, value in state_dict.items()}

            self.model.load_state_dict(state_dict)

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
