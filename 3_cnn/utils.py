import torch
import torch.nn.functional as F
import torchvision

from dataclasses import replace

from semantic_segmentation import *
from semantic_segmentation.backbones import BaseBackbone
from training import create_dataloaders
from metrics import calculate_miou
from config import ExperimentConfig


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def format_params(num):
    return f"{num / 1e6:.5f}M"


def verify_backbone(backbone: BaseBackbone):
    """
    Verify the backbone by passing random input through it and checking the output feature map sizes.
    """
    # verify if the get_channels method is correctly specified
    channels = backbone.get_channels()
    assert isinstance(channels, dict), "get_channels must return a dictionary"

    N = 2  # batch size
    H, W = 224, 224
    x = torch.randn(N, 3, H, W)
    features = backbone(x)

    shape_map = {
        "out": (H, W),
        "out2": (H // 2, W // 2),
        "out4": (H // 4, W // 4),
        "out8": (H // 8, W // 8),
        "out16": (H // 16, W // 16),
        "out32": (H // 32, W // 32),
        "out64": (H // 64, W // 64),
    }

    # feature map shapes specified by the backbone
    correct_shapes = {
        out_i: (N, C, shape_map[out_i][0], shape_map[out_i][1])
        for out_i, C in channels.items()
    }

    for name, feature in features.items():
        # print(f" -> {name} shape: {feature.shape}")
        assert (
            feature.shape == correct_shapes[name]
        ), f"Shape mismatch for {name}. Expected {correct_shapes[name]} but got {feature.shape}"

    print("Backbone features verified successfully!")


def add_param_to_config_name(config, model):
    """
    Add the number of parameters in the backbone and classifier to the config name.
    """
    backbone_params = count_params(model.backbone) / 1e6
    classifier_params = count_params(model.classifier) / 1e6
    return replace(
        config,
        config_name=f"{config.config_name}_{backbone_params:.2f}M_{classifier_params:.2f}M",
    )


@torch.no_grad()
def eval_model(model, config, model_name):
    """
    Evaluate the model on the VOC2012 val set.
    """
    _, val_loader = create_dataloaders(config)
    model.eval()
    model.to(config.device)
    total_loss = 0
    total_miou = 0
    total_acc = 0
    num_batches = 0
    for images, targets in val_loader:
        # drop last
        if images.size(0) < val_loader.batch_size:
            continue
        images = images.to(config.device)
        targets = targets.to(config.device)
        outputs = model(images)
        if isinstance(outputs, dict):
            outputs = outputs["out"]

        num_classes = outputs.size(1)

        outputs = (
            outputs.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        )  # (N * H * W, K)
        targets = targets.view(-1)  # (N * H * W,)

        # Use ignore_index in cross_entropy
        loss = F.cross_entropy(outputs, targets, ignore_index=config.void_label)

        preds = outputs.argmax(dim=1)
        miou = calculate_miou(preds, targets, num_classes, config.void_label)
        valid = (targets != config.void_label)
        accuracy = ((preds == targets) * valid).float().sum() / valid.sum().float()

        total_loss += loss.item()
        total_miou += miou.item()
        total_acc += accuracy.item()
        num_batches += 1
    loss = total_loss / num_batches
    miou = total_miou / num_batches
    acc = total_acc / num_batches

    torch.cuda.empty_cache()
    num_params = count_params(model) / 1e6
    print(
        f"{model_name} ({num_params:.2f}M) | Val loss: {loss:.4f} | val mIoU: {miou:.4f} | val acc: {acc:.4f} | Resolutions: {config.img_size}x{config.img_size}"
    )


def eval_official(resolutions=[65, 129, 225, 513]):
    """
    Evaluate the official off-the-shelf torchvision models on VOC2012 val set with different resolutions.
    """
    torch.manual_seed(0)
    torch_models = {}

    weights = torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights=weights)
    torch_models["MobileNetV3_Large"] = model

    weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
    torch_models["DeepLabV3_ResNet50"] = model

    weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights)
    torch_models["DeepLabV3_ResNet101"] = model

    for model_name, model in torch_models.items():
        for resolution in resolutions:
            config = ExperimentConfig("", "", "", "", img_size=resolution)
            eval_model(model, config, model_name)


def print_model_info(model):
    backbone_params = count_params(model.backbone)
    classifier_params = count_params(model.classifier)
    total_params = backbone_params + classifier_params

    line_width = 60
    print("\n" + "=" * line_width)
    print(f"{'Semantic Segmentation Model Summary':^50}")
    print("=" * line_width)
    print(f"{'Component':<20} {'Parameters':>15} {'Percentage':>15}")
    print("-" * line_width)
    print(
        f"{'Backbone':<20} {format_params(backbone_params):>15} {backbone_params / total_params * 100:>14.1f}%"
    )
    print(
        f"{'Classifier':<20} {format_params(classifier_params):>15} {classifier_params / total_params * 100:>14.1f}%"
    )
    print("-" * line_width)
    print(f"{'Total':<20} {format_params(total_params):>15}")
    print("\nConfiguration:")
    print(f"  Backbone: {model.backbone.__class__.__name__}")
    print(f"  Classifier: {model.classifier.__class__.__name__}")
    print(f"  Train Backbone: {'Yes' if model.config.train_backbone else 'No'}")
    print(f"  Learning Rate: {model.config.lr}")
    print(f"  Weight Decay: {model.config.weight_decay}")
    print("=" * line_width + "\n")
