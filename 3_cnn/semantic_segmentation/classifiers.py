import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from .backbones import FeatureDict, ChannelDict
from typing import Tuple, List, Dict


class BaseClassifier(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, features: FeatureDict) -> torch.Tensor:
        pass

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.atrous_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.atrous_conv(x)))


def determine_atrous_rates(output_stride: int) -> Tuple[int]:
    if output_stride == 16:
        atrous_rates = (6, 12, 18)
    elif output_stride == 8:
        atrous_rates = (12, 24, 36)
    else:
        raise NotImplementedError(f"Output stride of {output_stride} is not supported")
    return atrous_rates


class DeepLabV3PlusClassifier(BaseClassifier):
    """
    Classifer with similar architecture to DeepLabV3+
    In DeepLabV3+, output_stride should be set to 16
    """

    def __init__(self, backbone_channels: ChannelDict, config):
        """
        Variables:
            (Known) hidden_size: the output size of the final 3x3 conv layer
            (Known) low_in_channels: channels in the low level features
            (Known) high_in_channels: channels in the high level features
            (?) low_out_channels: output channels 1x1 conv (low)
            (?) aspp_out_channels: output channels after ASPP
            (?) high_out_channels: output channels after 1x1 conv (high) after ASPP

        "<-" means is determined by:
            hidden_size <- classifier_hidden_size
            (low_in_channels, high_in_channels) <- (backbone_channels, output_stride)
            low_out_channels, aspp_out_channels, high_out_channels <- ????

        We choose the following values to make the model as close to the original as possible while being flexible in size:
            low_out_channels = low_in_channels // 4
            aspp_out_channels = hidden_size
            high_out_channels = hidden_size
        """
        super().__init__()

        self.img_shape = (config.img_size, config.img_size)
        self.output_stride = config.output_stride

        num_classes = config.num_classes
        hidden_size = config.classifier_hidden_size

        atrous_rates = determine_atrous_rates(self.output_stride)

        if self.output_stride == 16:
            self.high_feature = "out16"
            self.low_feature = "out4"
        elif self.output_stride == 8:
            self.high_feature = "out8"
            self.low_feature = "out4"
        else:
            raise NotImplementedError(
                f"Output stride of {self.output_stride} is not supported"
            )

        high_in_channels = backbone_channels[self.high_feature]
        low_in_channels = backbone_channels[self.low_feature]

        low_out_channels = low_in_channels // 4
        aspp_out_channels = hidden_size
        high_out_channels = hidden_size

        
        self.aspp = ASPP(high_in_channels, atrous_rates, aspp_out_channels)

        self.high_project = nn.Sequential(
            nn.Conv2d(aspp_out_channels, high_out_channels, 1, bias=False),
            nn.BatchNorm2d(high_out_channels),
            nn.ReLU(inplace=True),
        )

        self.low_project = nn.Sequential(
            nn.Conv2d(low_in_channels, low_out_channels, 1, bias=False),
            nn.BatchNorm2d(low_out_channels),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(
                low_out_channels + high_out_channels,
                hidden_size,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, num_classes, 1),
        )

        self._init_weights()

    def forward(self, features: FeatureDict) -> torch.Tensor:
        low = self.low_project(features[self.low_feature])
        high = self.aspp(features[self.high_feature])
        high = self.high_project(high)
        high = F.interpolate(
            high, size=low.shape[-2:], mode="bilinear", align_corners=False
        )
        x = torch.cat([low, high], dim=1)
        x = self.classifier(x)
        x = F.interpolate(x, size=self.img_shape, mode="bilinear", align_corners=False)
        return x
