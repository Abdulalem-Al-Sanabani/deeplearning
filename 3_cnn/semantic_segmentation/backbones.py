from typing import TypedDict, Optional
import torch
import torch.nn as nn
import torchvision
from abc import ABC, abstractmethod


class FeatureDict(TypedDict, total=False):
    """
    A dictionary containing the output feature maps of the backbone (encoder) at different scales.
    If config.output_stride = 8, there will be no out16
    """

    out2: torch.Tensor  # output feature map of size H//2, W//2
    out4: torch.Tensor  # output feature map of size H//4, W//4
    out8: torch.Tensor  # output feature map of size H//8, W//8
    out16: Optional[torch.Tensor]  # output feature map of size H//16, W//16


class ChannelDict(TypedDict, total=False):
    """
    A dictionary containing the number of channels in the output feature maps of the backbone (encoder) at different scales.
    If config.output_stride = 8, there will be no out16
    """

    out2: int  # the C in tensor of shape (batch_size, C, H//2, W//2)
    out4: int  # the C in tensor of shape (batch_size, C, H//4, W//4)
    out8: int  # the C in tensor of shape (batch_size, C, H//8, W//8)
    out16: Optional[int]  # the C in tensor of shape (batch_size, C, H//16, W//16)


class BaseBackbone(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.features = {}
        self.hooks = []
        self.model = None

    @abstractmethod
    def get_channels(self) -> ChannelDict:
        """
        Return a ChannelDict containing the number of channels in the output feature maps of the backbone at different scales.
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> FeatureDict:
        pass

    def _register_feature_hooks(self, layer_names):
        """
        Register forward hooks on the specified layers of the model to extract the feature maps at those layers.
        layer_names: a list of strings containing the names of the layers to extract the feature maps from.
        """

        def hook_fn(name):
            def fn(_, __, output):
                self.features[name] = output

            return fn

        # check if the layer names are valid
        model_layers = [name for name, _ in self.model.named_children()]
        for layer_name in layer_names:
            assert layer_name in model_layers, f"{layer_name} not found in the model"

        # register hooks
        for name, layer in self.model.named_children():
            if name in layer_names:
                self.hooks.append(layer.register_forward_hook(hook_fn(name)))

        # check if all the layers are hooked
        assert len(self.hooks) == len(
            layer_names
        ), "Some layers are not found in the model"


class ResNetBackbone(BaseBackbone):
    """
    A ResNet backbone that extracts feature maps at different scales.
    Note that if output_stride=8, there will be no out16 feature map.
    For segmentation tasks, it is recommended to use resnet50 or resnet101 because there are customed DeepLabV3 models (different architecture, pre-trained) available for them.
    """

    RESNET_18 = "resnet18"
    RESNET_34 = "resnet34"
    RESNET_50 = "resnet50"
    RESNET_101 = "resnet101"
    RESNET_152 = "resnet152"

    def __init__(self, config):
        super().__init__()
        self.backbone_name = config.backbone_name
        self.output_stride = config.output_stride
        self.pretrained_backbone = config.pretrained_backbone

        self.model, self.feature_names = self._load_resnet(
            self.backbone_name, self.pretrained_backbone, self.output_stride
        )

        # register hooks to extract the feature maps
        self._register_feature_hooks(self.feature_names)

        # remove the unused layers
        if (
            self.backbone_name in [ResNetBackbone.RESNET_50, ResNetBackbone.RESNET_101]
            and self.output_stride == 8
        ):
            pass  # do nothing, we are using the custom DeepLabV3 models
        elif self.feature_names[-1] == "layer4":
            self.model = nn.Sequential(
                *list(self.model.children())[:-2]
            )  # remove fc and avgpool
        elif self.feature_names[-1] == "layer3":
            self.model = nn.Sequential(
                *list(self.model.children())[:-3]
            )  # remove layer4, fc, and avgpool
        elif self.feature_names[-1] == "layer2":
            self.model = nn.Sequential(
                *list(self.model.children())[:-4]
            )  # remove layer3, layer4, fc, and avgpool
        else:
            raise ValueError(f"Invalid feature names: {self.feature_names}")

    def _load_resnet(self, backbone_name: str, pretrained: bool, output_stride: int):
        assert output_stride in [
            8,
            16,
        ], f"Invalid output stride, Got: {output_stride}, Expected: 8 or 16"
        assert backbone_name in [
            ResNetBackbone.RESNET_18,
            ResNetBackbone.RESNET_34,
            ResNetBackbone.RESNET_50,
            ResNetBackbone.RESNET_101,
            ResNetBackbone.RESNET_152,
        ], f"Invalid backbone name: {backbone_name}"

        if output_stride == 16:
            if backbone_name in [ResNetBackbone.RESNET_18, ResNetBackbone.RESNET_34]:
                feature_names = ["relu", "layer1", "layer2", "layer3"]
                replace_stride_with_dilation = [
                    False,
                    False,
                    False,
                ]  # resnet18 and resnet34 don't support dilation
            else:
                feature_names = ["relu", "layer1", "layer2", "layer4"]
                replace_stride_with_dilation = [False, False, True]

        elif output_stride == 8:
            if backbone_name in [ResNetBackbone.RESNET_18, ResNetBackbone.RESNET_34]:
                feature_names = ["relu", "layer1", "layer2"]
                replace_stride_with_dilation = [False, False, False]
            else:
                feature_names = ["relu", "layer1", "layer4"]
                replace_stride_with_dilation = [False, True, True]

        if backbone_name == ResNetBackbone.RESNET_18:
            weights = (
                torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            model = torchvision.models.resnet18(
                weights=weights,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )
        elif backbone_name == ResNetBackbone.RESNET_34:
            weights = (
                torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None
            )
            model = torchvision.models.resnet34(
                weights=weights,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )
        elif backbone_name == ResNetBackbone.RESNET_152:
            weights = (
                torchvision.models.ResNet152_Weights.DEFAULT if pretrained else None
            )
            model = torchvision.models.resnet152(
                weights=weights,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )

        # Since pytorch provide a custom DeepLabV3 model for resnet50 and resnet101, if we output_stride=8, we can use these custom models
        if output_stride == 8:
            if backbone_name == ResNetBackbone.RESNET_50:
                weights = (
                    torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
                    if pretrained
                    else None
                )
                model = torchvision.models.segmentation.deeplabv3_resnet50(
                    weights=weights
                ).backbone
            elif backbone_name == ResNetBackbone.RESNET_101:
                weights = (
                    torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
                    if pretrained
                    else None
                )
                model = torchvision.models.segmentation.deeplabv3_resnet101(
                    weights=weights
                ).backbone
        elif output_stride == 16:
            if backbone_name == ResNetBackbone.RESNET_50:
                weights = (
                    torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
                )
                model = torchvision.models.resnet50(
                    weights=weights,
                    replace_stride_with_dilation=replace_stride_with_dilation,
                )
            elif backbone_name == ResNetBackbone.RESNET_101:
                weights = (
                    torchvision.models.ResNet101_Weights.DEFAULT if pretrained else None
                )
                model = torchvision.models.resnet101(
                    weights=weights,
                    replace_stride_with_dilation=replace_stride_with_dilation,
                )

        return model, feature_names

    def get_channels(self) -> ChannelDict:
        if self.output_stride == 16:
            if self.backbone_name in [
                ResNetBackbone.RESNET_18,
                ResNetBackbone.RESNET_34,
            ]:
                channels = {
                    "out2": 64,
                    "out4": 64,
                    "out8": 128,
                    "out16": 256,
                }
            else:
                channels = {
                    "out2": 64,
                    "out4": 256,
                    "out8": 512,
                    "out16": 2048,
                }

        elif self.output_stride == 8:
            if self.backbone_name in [
                ResNetBackbone.RESNET_18,
                ResNetBackbone.RESNET_34,
            ]:
                channels = {
                    "out2": 64,
                    "out4": 64,
                    "out8": 128,
                }
            else:
                channels = {
                    "out2": 64,
                    "out4": 256,
                    "out8": 2048,
                }

        return channels

    def forward(self, x: torch.Tensor) -> FeatureDict:
        """
        Output looks like:
            {
                'out2': self.features['maxpool'],
                'out4': self.features['layer1'],
                'out8': self.features['layer2'],
                'out16': self.features['layer3'],
            }
        """
        self.model(x)  # ['out4', 'out8', ...]
        outs = list(FeatureDict.__annotations__.keys())
        return {
            out_i: self.features[name] for out_i, name in zip(outs, self.feature_names)
        }


class MobileNetV3Backbone(BaseBackbone):
    """
    A MobileNet V3 backbone that extracts feature maps at different scales.
    It's recommended to use the 'large' model for segmentation tasks because it has a customed architecture for segmentation tasks and the pre-trained weights are available.
    """

    LARGE_NAME = "mobilenetv3_large"
    SMALL_NAME = "mobilenetv3_small"

    def __init__(self, config):
        super().__init__()
        self.backbone_name = config.backbone_name
        self.output_stride = config.output_stride
        self.pretrained_backbone = config.pretrained_backbone
        self.model, self.feature_names = self._load_model(
            self.backbone_name, self.pretrained_backbone, self.output_stride
        )

        self._register_feature_hooks(self.feature_names)

        # remoe unused layers
        last_index = int(self.feature_names[-1])
        self.model = nn.Sequential(*list(self.model.children())[: last_index + 1])

    def _load_model(self, backbone_name: str, pretrained: bool, output_stride: int):
        assert backbone_name in [
            MobileNetV3Backbone.LARGE_NAME,
            MobileNetV3Backbone.SMALL_NAME,
        ], f"Invalid backbone name: {backbone_name}"
        assert output_stride in [8, 16], f"Invalid output stride: {output_stride}"

        if backbone_name == MobileNetV3Backbone.LARGE_NAME:
            weights = (
                torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
                if pretrained
                else None
            )
            model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
                weights=weights
            ).backbone
            feature_names = ["1", "3", "4", "16"]  # align with MobileNetV3 paper
        elif backbone_name == MobileNetV3Backbone.SMALL_NAME:
            weights = (
                torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
                if pretrained
                else None
            )
            model = torchvision.models.mobilenet_v3_small(weights=weights).features
            feature_names = ["0", "1", "3", "8"]

        if output_stride == 8:
            feature_names = feature_names[:-1]

        return model, feature_names

    def get_channels(self) -> ChannelDict:
        if self.backbone_name == MobileNetV3Backbone.LARGE_NAME:
            channels = {
                "out2": 16,
                "out4": 24,
                "out8": 40,
                "out16": 960,
            }
        if self.backbone_name == MobileNetV3Backbone.SMALL_NAME:
            channels = {
                "out2": 16,
                "out4": 16,
                "out8": 24,
                "out16": 48,
            }
        return channels

    def forward(self, x: torch.Tensor) -> FeatureDict:
        self.model(x)
        outs = list(FeatureDict.__annotations__.keys())
        return {
            out_i: self.features[name] for out_i, name in zip(outs, self.feature_names)
        }
