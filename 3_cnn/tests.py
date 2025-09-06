import torch
import torch.nn as nn

from semantic_segmentation.backbones import BaseBackbone
from semantic_segmentation.classifiers import BaseClassifier
from semantic_segmentation.backbones import ChannelDict


class TestResidualBlock:
    @staticmethod
    def test_no_downsampling(ResidualBlock):
        in_channels, out_channels = 64, 64
        block = ResidualBlock(in_channels, out_channels)
        x = torch.randn(1, in_channels, 32, 32)
        output = block(x)
        assert output.shape == (
            1,
            out_channels,
            32,
            32,
        ), f"Expected shape (1, {out_channels}, 32, 32), got {output.shape}"
        print("Forward without downsampling test passed!")

    @staticmethod
    def test_downsampling(ResidualBlock):
        in_channels, out_channels = 64, 128
        block = ResidualBlock(in_channels, out_channels, use_downsample=True)
        x = torch.randn(1, in_channels, 32, 32)
        output = block(x)
        assert output.shape == (
            1,
            out_channels,
            16,
            16,
        ), f"Expected shape (1, {out_channels}, 16, 16), got {output.shape}"
        print("Forward with downsampling test passed!")

    @staticmethod
    def test_structure(ResidualBlock):
        # Test case 1: No downsampling
        block = ResidualBlock(64, 64, use_downsample=False)
        TestResidualBlock._check_basic_structure(block)
        assert (
            block.downsample is None
        ), "Downsample should be None when use_downsample is False"

        # Test case 2: With downsampling
        block = ResidualBlock(64, 128, use_downsample=True)
        TestResidualBlock._check_basic_structure(block)
        TestResidualBlock._check_downsample_structure(block)

        print("All basic structure tests passed!")

    @staticmethod
    def _check_basic_structure(block):
        assert hasattr(block, "conv1"), "Missing conv1 layer"
        assert hasattr(block, "bn1"), "Missing bn1 layer"
        assert hasattr(block, "relu"), "Missing relu layer"
        assert hasattr(block, "conv2"), "Missing conv2 layer"
        assert hasattr(block, "bn2"), "Missing bn2 layer"

    @staticmethod
    def _check_downsample_structure(block):
        assert (
            block.downsample is not None
        ), "Downsample should not be None when use_downsample is True"
        assert isinstance(
            block.downsample, nn.Sequential
        ), "Downsample should be nn.Sequential"
        assert len(block.downsample) == 2, "Downsample should have exactly 2 layers"
        assert isinstance(
            block.downsample[0], nn.Conv2d
        ), "First layer of downsample should be Conv2d"
        assert isinstance(
            block.downsample[1], nn.BatchNorm2d
        ), "Second layer of downsample should be BatchNorm2d"


class TestResNet34:
    @staticmethod
    def test_structure(ResNet34):
        model = ResNet34()

        # Check for essential components
        assert hasattr(model, "conv1"), "Missing conv1 layer"
        assert hasattr(model, "bn1"), "Missing bn1 layer"
        assert hasattr(model, "relu"), "Missing relu layer"
        assert hasattr(model, "maxpool"), "Missing maxpool layer"
        assert hasattr(model, "layer1"), "Missing layer1"
        assert hasattr(model, "layer2"), "Missing layer2"
        assert hasattr(model, "layer3"), "Missing layer3"
        assert hasattr(model, "layer4"), "Missing layer4"
        assert hasattr(model, "avgpool"), "Missing avgpool layer"
        assert hasattr(model, "fc"), "Missing fc layer"

        print("Basic structure test passed!")

    @staticmethod
    def test_forward(ResNet34):
        model = ResNet34()
        x = torch.randn(1, 3, 224, 224)
        output = model(x)

        assert output.shape == (
            1,
            1000,
        ), f"Expected output shape (1, 1000), got {output.shape}"
        print("Forward pass test passed!")

    @staticmethod
    def test_layer_shapes(ResNet34):
        model = ResNet34()
        x = torch.randn(1, 3, 224, 224)

        x = model.conv1(x)
        assert x.shape == (1, 64, 112, 112), f"Wrong shape after conv1: {x.shape}"

        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        assert x.shape == (1, 64, 56, 56), f"Wrong shape after maxpool: {x.shape}"

        x = model.layer1(x)
        assert x.shape == (1, 64, 56, 56), f"Wrong shape after layer1: {x.shape}"

        x = model.layer2(x)
        assert x.shape == (1, 128, 28, 28), f"Wrong shape after layer2: {x.shape}"

        x = model.layer3(x)
        assert x.shape == (1, 256, 14, 14), f"Wrong shape after layer3: {x.shape}"

        x = model.layer4(x)
        assert x.shape == (1, 512, 7, 7), f"Wrong shape after layer4: {x.shape}"

        x = model.avgpool(x)
        assert x.shape == (1, 512, 1, 1), f"Wrong shape after avgpool: {x.shape}"

        x = torch.flatten(x, 1)
        x = model.fc(x)
        assert x.shape == (1, 1000), f"Wrong shape after fc: {x.shape}"

        print("Layer shapes test passed!")


class TestStepFn:
    @staticmethod
    def _create_dummy_data(batch_size=2, num_classes=21, height=32, width=32):
        outputs = torch.randn(batch_size, num_classes, height, width)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        return outputs, targets

    @staticmethod
    def test_step(step_fn):
        outputs, targets = TestStepFn._create_dummy_data()
        void_label = 255

        result = step_fn(outputs, targets, void_label)

        expected_keys = ["loss", "mIoU", "accuracy"]
        for key in expected_keys:
            assert key in result, f"Expected '{key}' in step_fn result"
            assert isinstance(
                result[key], torch.Tensor
            ), f"Expected {key} to be a tensor"
            assert result[key].ndim == 0, f"Expected {key} to be a scalar tensor"

        print("step_fn sanity check passed!")


class TestSemanticSegmentationModel:
    @staticmethod
    def test_forward(SemanticSegmentationModel):
        class DummyBackbone(BaseBackbone):
            def forward(self, x):
                return {"out16": x}

            def get_channels(self):
                return {"out16": 3}

        class DummyClassifier(BaseClassifier):
            def forward(self, features):
                return features["out16"]

        dummy_backbone = DummyBackbone()
        dummy_classifier = DummyClassifier()

        class DummyConfig:
            train_backbone = False

        model = SemanticSegmentationModel(
            dummy_backbone, dummy_classifier, DummyConfig()
        )

        input_tensor = torch.randn(2, 3, 32, 32)
        output = model(input_tensor)

        assert (
            output.shape == input_tensor.shape
        ), "Output shape should match input shape"
        print("SemanticSegmentationModel forward sanity check passed!")


class TestClassifiers:
    @staticmethod
    def _create_dummy_config():
        class DummyConfig:
            img_size = 32
            output_stride = 16
            num_classes = 21
            classifier_hidden_size = 64

        return DummyConfig()

    @staticmethod
    def _create_dummy_feature_dict():
        return {
            "out4": torch.randn(2, 24, 8, 8),
            "out8": torch.randn(2, 40, 4, 4),
            "out16": torch.randn(2, 960, 2, 2),
        }

    @staticmethod
    def test_lite_raspp_classifier(LiteRASPPClassifier):
        config = TestClassifiers._create_dummy_config()
        backbone_channels = ChannelDict(out4=24, out8=40, out16=960)

        classifier = LiteRASPPClassifier(backbone_channels, config)
        features = TestClassifiers._create_dummy_feature_dict()

        output = classifier(features)

        assert output.shape == (
            2,
            config.num_classes,
            config.img_size,
            config.img_size,
        ), f"Expected output shape {(2, config.num_classes, config.img_size, config.img_size)}, but got {output.shape}"
        print("LiteRASPPClassifier sanity check passed!")

    @staticmethod
    def test_aspp(ASPP):
        in_channels = 960
        atrous_rates = (6, 12, 18)
        out_channels = 256

        aspp = ASPP(in_channels, atrous_rates, out_channels)
        x = torch.randn(2, in_channels, 32, 32)

        output = aspp(x)

        assert output.shape == (
            2,
            out_channels,
            32,
            32,
        ), f"Expected output shape {(2, out_channels, 32, 32)}, but got {output.shape}"
        print("ASPP sanity check passed!")

    @staticmethod
    def test_deeplabv3_classifier(DeepLabV3Classifier):
        config = TestClassifiers._create_dummy_config()
        backbone_channels = ChannelDict(out4=24, out8=40, out16=960)

        classifier = DeepLabV3Classifier(backbone_channels, config)
        features = TestClassifiers._create_dummy_feature_dict()

        output = classifier(features)

        assert output.shape == (
            2,
            config.num_classes,
            config.img_size,
            config.img_size,
        ), f"Expected output shape {(2, config.num_classes, config.img_size, config.img_size)}, but got {output.shape}"
        print("DeepLabV3Classifier sanity check passed!")
