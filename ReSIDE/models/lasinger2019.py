from typing import List, Union, Optional
from warnings import warn

import plac
import torch
import torchvision
from efficientnet_pytorch import EfficientNet
from torch import nn
# noinspection PyPep8Naming
from torch.nn import functional as F


__version__ = '0.2.0'


class Encoder(nn.Module):
    """An encoder that uses a pre-trained network."""

    def __init__(self, name="efficientnet-b0", pretrained=False, freeze_weights=False):
        """
        :param name: The name of the encoder to use, e.g. 'resnet50', 'efficientnet-b0'.
        :param pretrained: Whether to use a pre-trained version of the encoder.
        :param freeze_weights: Whether to use the encoder as a fixed (non-trainable) feature extractor.
        """
        super().__init__()

        self.name = name.lower()
        self.pretrained = pretrained
        self.freeze_weights = freeze_weights

        if name.startswith("efficientnet"):
            self.blocks, self._block_out_channels = Encoder._make_efficientnet_encoder(self.name, pretrained)
        elif name.startswith("resnet"):
            self.blocks, self._block_out_channels = Encoder._make_resnet_encoder(self.name, pretrained)
        else:
            raise RuntimeError(f"Unsupported encoder network '{self.name}'.")

        if freeze_weights:
            # Use the encoder as a fixed feature extractor.
            self.blocks.requires_grad_(False)

    @property
    def block_out_channels(self) -> List[int]:
        return self._block_out_channels

    def forward(self, x):
        output = x
        outputs = []

        for block in self.blocks:
            output = block(output)
            outputs.append(output)

        return outputs

    # noinspection PyProtectedMember
    @staticmethod
    def _make_efficientnet_encoder(encoder_name, pretrained):
        net = EfficientNet.from_pretrained(encoder_name) if pretrained else EfficientNet.from_name(encoder_name)

        if encoder_name == "efficientnet-b0":
            block_indices = [0, 3, 5, 8, len(net._blocks)]
        elif encoder_name in {"efficientnet-b1", "efficient-b2"}:
            block_indices = [0, 5, 8, 16, len(net._blocks)]
        elif encoder_name == "efficientnet-b3":
            block_indices = [0, 5, 8, 18, len(net._blocks)]
        elif encoder_name == "efficientnet-b4":
            block_indices = [0, 6, 10, 22, len(net._blocks)]
        elif encoder_name == "efficientnet-b5":
            block_indices = [0, 8, 13, 27, len(net._blocks)]
        elif encoder_name == "efficientnet-b6":
            block_indices = [0, 9, 15, 31, len(net._blocks)]
        elif encoder_name == "efficientnet-b7":
            block_indices = [0, 11, 18, 38, len(net._blocks)]
        else:
            raise RuntimeError(f"Unsupported encoder network '{encoder_name}'.")

        input_layers = list(net.children())[:2]

        blocks = nn.ModuleList([
            nn.Sequential(*input_layers, *net._blocks[block_indices[0]:block_indices[1]])
        ])

        for i in range(1, len(block_indices) - 1):
            start, end = block_indices[i], block_indices[i + 1]

            blocks.append(
                nn.Sequential(*net._blocks[start:end])
            )

        block_out_channels = [list(block.modules())[-2].num_features for block in blocks]

        return blocks, block_out_channels

    @staticmethod
    def _make_resnet_encoder(encoder_name, pretrained):
        if encoder_name == "resnet18":
            net = torchvision.models.resnet18(pretrained)
        elif encoder_name == "resnet50":
            net = torchvision.models.resnet50(pretrained)
        elif encoder_name == "resnet101":
            net = torchvision.models.resnet101(pretrained)
        elif encoder_name == "resnet152":
            net = torchvision.models.resnet152(pretrained)
        else:
            raise RuntimeError(f"Unsupported encoder network '{encoder_name}'.")

        # Remove the last two layers (pooling and fully connected layers) since we don't need them.
        layers = list(net.children())[:-2]

        blocks = nn.ModuleList(
            [nn.Sequential(*[layer for layer in layers[:5]])] + [nn.Sequential(layer) for layer in layers[5:]]
        )

        if encoder_name == "resnet18":
            block_out_channels = [list(block.modules())[-1].num_features for block in blocks]
        elif encoder_name in {"resnet50", "resnet101", "resnet152"}:
            block_out_channels = [list(block.modules())[-2].num_features for block in blocks]
        else:
            raise RuntimeError(f"Unsupported encoder network '{encoder_name}'.")

        return blocks, block_out_channels


class ResidualBlock(nn.Module):
    """A simple convolutional block with a skip connection."""

    def __init__(self, in_channels, out_channels=None, stride=1):
        """
        :param in_channels: The number of channels expected in the input.
        :param out_channels: The number of channels that should be in the output of this block.
        :param stride: The stride for the cross-correlation. For example, a stride of one will retain the original
        output resolution, whereas a stride of two will halve the output resolution.
        """
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down_sample = None

    def forward(self, x):
        # Cannot do inplace operation here since it will directly modify x, which is needed for the skip connection.
        output = F.relu(x, inplace=False)
        output = F.relu(self.bn1(self.conv1(output)), inplace=True)
        # Do not apply ReLU here since the output of this block may be added with a raw output in the decoder.
        output = self.bn2(self.conv2(output))

        if self.down_sample is not None:
            output += self.down_sample(x)
        else:
            output += x

        return output


class BottleneckBlock(nn.Module):
    """The bottleneck version of the residual block."""

    def __init__(self, in_channels, out_channels=None):
        """
        :param in_channels: The number of channels expected in the input.
        :param out_channels: The number of channels that should be in the output of this block.
        """
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        # 1/4 is the ratio that the input channels are reduced by in the PyTorch reference implementation.
        # See: https://github.com/pytorch/vision/blob/227027d5abc8eacb110c93b5b5c2f4ea5dd401d6/torchvision/models/resnet.py#L77
        bottleneck_channels = max(1, out_channels // 4)

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down_sample = None

    def forward(self, x):
        # Cannot do inplace operation here since it will directly modify x, which is needed for the skip connection.
        output = F.relu(x, inplace=False)
        output = F.relu(self.bn1(self.conv1(output)), inplace=True)
        output = F.relu(self.bn2(self.conv2(output)), inplace=True)
        # Do not apply ReLU here since the output of this block may be added with a raw output in the decoder.
        output = self.bn3(self.conv3(output))

        if self.down_sample is not None:
            output += self.down_sample(x)
        else:
            output += x

        return output


class DecoderBlock(nn.Module):
    """A block that transforms the outputs from the previous decoder block and the corresponding encoder block."""

    def __init__(self, in_channels, out_channels, block_type=Union[ResidualBlock, BottleneckBlock]):
        """
        :param in_channels: The number of input channels in the input.
        :param out_channels: The number of output channels for this block.
        :param block_type: The type of residual block to use.
        """
        super().__init__()

        self.res_block1 = block_type(in_channels, out_channels)
        self.res_block2 = block_type(out_channels)

    def forward(self, input_from_encoder, input_from_decoder, output_size):
        """
        Perform a forward pass of this block.

        :param input_from_encoder: The output from the corresponding encoder block.
        :param input_from_decoder: The output from the previous decoder block, or None if this is the first block.
        :param output_size: The output resolution (in WH format) that the block's output should be rescaled to.

        :return: The output of this block.
        """
        if input_from_decoder is None:
            output = self.res_block1(input_from_encoder)
        else:
            output = input_from_decoder + self.res_block1(input_from_encoder)
            output = self.res_block2(output)

        output = F.interpolate(output, size=output_size, mode="bilinear", align_corners=True)

        return output


class Decoder(nn.Module):
    """The decoder sub-network in an Encoder-Decoder network architecture."""

    def __init__(self, encoder_block_channels, num_features: Union[int, str] = 'auto', non_negative=False,
                 block_type=ResidualBlock):
        """
        :param encoder_block_channels: A list of the number of output channels for each block in the encoder.
        See `Encoder.block_out_channels`.
        :param num_features: The number of output channels for each decoder block. If 'auto', the number of channels
        defaults to the number of output channels in the first block of the encoder.
        :param non_negative: Whether to use the ReLU activation function on the output to clip negative values.
        :param block_type: The type of residual block to use in the decoder blocks.
        """
        super().__init__()

        self.num_features = encoder_block_channels[0] if num_features == 'auto' else int(num_features)
        self.non_negative = non_negative

        self.blocks = nn.ModuleList([
            DecoderBlock(in_channels, self.num_features, block_type) for in_channels in reversed(encoder_block_channels)
        ])

        self.conv1 = nn.Conv2d(self.num_features, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, encoder_outputs, output_size):
        output_sizes = [encoder_output.shape[-2:] for encoder_output in reversed(encoder_outputs[:-1])]
        output_sizes.append(tuple(map(lambda x: 2 * x, output_sizes[-1])))

        output = None

        for decoder_block, encoder_output, size in zip(self.blocks, reversed(encoder_outputs), output_sizes):
            output = decoder_block(encoder_output, output, size)

        output = F.relu(self.bn1(self.conv1(output)), inplace=True)
        output = F.interpolate(output, size=output_size, mode="bilinear", align_corners=True)

        output = F.relu(self.bn2(self.conv2(output)), inplace=True)
        # TODO: Fix output of conv3 becoming all negative, resulting in a black depth map and no gradient for training.
        output = F.relu(self.conv3(output), inplace=True) if self.non_negative else self.conv3(output)

        return output


class BestNet(nn.Module):
    """
    A re-implementation of MiDaS from [1].

    References:

    [1] K. Lasinger, R. Ranftl, K. Schindler, and V. Koltun, “Towards Robust Monocular Depth Estimation: Mixing
    Datasets for Zero-Shot Cross-Dataset Transfer,” 2019.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 output_size: tuple, input_size: Optional[tuple] = None,
                 adversarial_training=False):
        """
        :param encoder: The encoder sub-network to use.
        :param decoder: The decoder sub-network to use.
        :param output_size: The resolution of the output depth maps in WH format (width, height).
        :param input_size: (Optional) The resolution of the input rgb images in WH format (width, height). If None,
        `output_size` is used as the input resolution. Note that this parameter has no effect on the actual network, and
        is only here for reproducing results.
        :param adversarial_training: Metadata indicating that this network was trained with adversarial training.
        Important for knowing which normalisation scheme to apply to the input (e.g. ImageNet mean and standard
        deviation vs. [-1.0, 1.0]). Note: only set this to true if the network was trained on images normalised to the
        range [-1.0, 1.0].
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.input_size = input_size if input_size else output_size
        # TODO: Allow for dynamic output size for testing on random images.
        #  See https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Resize with int
        #  argument.
        self.output_size = output_size
        self.adversarial_training = adversarial_training

    @property
    def input_size(self):
        """
        :return: The resolution of the input rgb images that the network was trained on in WH format (width, height).
        """
        height, width = self._input_size
        return width, height

    @input_size.setter
    def input_size(self, size):
        """
        Set the resolution of the depth maps produced by the network.
        :param size: A 2-tuple containing the desired resolution in WH format (width, height).
        """
        width, height = size
        self._input_size = (height, width)

    @property
    def output_size(self):
        """
        :return: The resolution of the depth maps produced by the network in WH format (width, height).
        """
        height, width = self._output_size
        return width, height

    @output_size.setter
    def output_size(self, size):
        """
        Set the resolution of the depth maps produced by the network.
        :param size: A 2-tuple containing the desired resolution in WH format (width, height).
        """
        width, height = size
        self._output_size = (height, width)

    def forward(self, x):
        return self.decoder(self.encoder(x), self._output_size)

    def save(self, f):
        """
        Save the model.

        :param f: A file-like object or string path to save the model to.
        """
        state = {
            'encoder': {
                'name': self.encoder.name,
                'freeze_weights': self.encoder.freeze_weights
            },
            'decoder': {
                'num_features': self.decoder.num_features,
                'non_negative': self.decoder.non_negative
            },
            'input_size': self.input_size,
            'output_size': self.output_size,
            'adversarial_training': self.adversarial_training,
            'weights': self.state_dict(),
            'version': __version__
        }

        torch.save(state, f)

    @staticmethod
    def load(f):
        """
        Load a model from a checkpoint file.

        :param f: A file-like object or string path to save the model from.
        :return: The loaded network.
        """
        state = torch.load(f)

        if state['version'] != __version__:
            warn(f"Version mismatch: the loaded weights are for version {state['version']}, however the "
                 f"current version is {__version__}. This may cause issues loading the model.")

        encoder = Encoder(**state['encoder'])
        decoder = Decoder(encoder.block_out_channels, **state['decoder'])
        net = BestNet(encoder, decoder, state['output_size'], state['input_size'], state['adversarial_training'])
        net.load_state_dict(state['weights'])

        return net


class Discriminator(nn.Module):
    """A discriminator network to use for adversarial training."""

    def __init__(self, in_channels=4, adversarial_training=False):
        """
        :param in_channels: How many channels the input will have.
        :param adversarial_training: Metadata indicating that this network was trained with adversarial training.
        Important for knowing which normalisation scheme to apply to the input (e.g. ImageNet mean and standard
        deviation vs. [-1.0, 1.0]). Note: only set this to true if the network was trained on images normalised to the
        range [-1.0, 1.0].
        """
        super().__init__()

        self.in_channels = in_channels
        self.adversarial_training = adversarial_training

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7),
            nn.BatchNorm2d(32),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 1024, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        return self.net(x)

    def save(self, f):
        """
        Save the model.

        :param f: A file-like object or string path to save the model to.
        """
        state = {
            'weights': self.state_dict(),
            'options': {
                'in_channels': self.in_channels,
                'adversarial_training': self.adversarial_training
            },
            'version': __version__
        }

        torch.save(state, f)

    @staticmethod
    def load(f):
        """
        Load a model from a checkpoint file.

        :param f: A file-like object or string path to save the model from.
        :return: The loaded network.
        """
        state = torch.load(f)

        if state['version'] != __version__:
            warn(f"Version mismatch: the loaded weights are for version {state['version']}, however the "
                 f"current version is {__version__}. This may cause issues loading the model.")

        net = Discriminator(**state['options'])
        net.load_state_dict(state['weights'])

        return net


