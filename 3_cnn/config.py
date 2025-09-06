from dataclasses import dataclass, fields
from typing import Tuple, Literal, Optional
import torch

ABBREVIATIONS = {
    "exp_name": None,
    "config_name": None,
    "backbone_name": None,
    "classifier_name": None,
    "classifier_hidden_size": "hidden",
    "pretrained_backbone": "pretrained",
    "train_backbone": "full",
    "output_stride": "os",
    "batch_size": "batch",
    "lr": "lr",
    "lr_scheduler": None,
    "img_size": "size",
}


@dataclass
class ExperimentConfig:
    # Experiment Identification
    exp_name: str

    # Model Architecture
    backbone_name: Literal[
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "mobilenetv3_large",
        "mobilenetv3_small",
    ]
    classifier_name: Literal["deeplabv3", "lite_raspp", "deeplabv3plus"]
    classifier_hidden_size: Optional[int] = None
    pretrained_backbone: bool = True
    train_backbone: bool = (
        False  # if True, the backbone will be trained with lr = config.lr / 10
    )
    output_stride: Literal[
        16, 8
    ] = 16  # 16 is recommended for MobileNetV3, 8 for ResNet

    # Experiment Identification (auto-generated or manually set)
    config_name: Optional[str] = None

    # Training parameters
    max_steps: int = 3000  # 1464 / 16 = 91.5 => 3000 steps are roughly 32 epochs
    batch_size: int = 16
    lr: float = 1e-3
    lr_scheduler: Literal[None, "poly", "onecycle"] = "poly"
    weight_decay: float = 1e-4

    # Evaluation parameters
    eval_every_n_steps: int = 500

    # Data parameters
    img_size: int = 513
    num_classes: int = 21
    void_label: int = 255  # Label for pixels that should be ignored in loss calculation and evaluation

    # Preprocessing parameters (the ImageNet mean and std are used by default)
    preprocess_mean: Tuple = (0.485, 0.456, 0.406)
    preprocess_std: Tuple = (0.229, 0.224, 0.225)

    # visualization parameters
    visualization_samples: int = 5
    alpha: float = 0.8  # transparency of the mask

    # Misc
    device: torch.device = None
    tqdm: bool = True

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.config_name is None:
            self.config_name = self._generate_config_name()

    def _generate_config_name(self):
        """
        Generate the config name based on the fields of the config object.
        A value will be included if (value != default and value in abbreviations).
        """
        config_name = ""
        for item in fields(self):
            if item.name in ["exp_name", "config_name"]:
                continue
            value = getattr(self, item.name)
            if value == item.default:
                continue
            if item.name in ABBREVIATIONS:
                abbr = ABBREVIATIONS[item.name]
                if abbr is not None:
                    config_name += abbr + "_"
                config_name += str(value) + "|"
        return config_name[:-1]  # remove the last underscore
