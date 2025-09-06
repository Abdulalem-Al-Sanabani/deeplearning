import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataclasses import dataclass

from core import make_summary_writer, BaseTrainer


@dataclass
class ExperimentConfig:
    exp_name: str  # name of experiment (e.g. dropout-vs-no-dropout)
    config_name: str  # name of the configuration (e.g. dropout-0.9)
    batch_size: int = 128
    max_steps: int = 10000
    lr: float = 3e-4
    input_dim: int = 784
    output_dim: int = 10
    num_layers: int = 4
    hidden_dim: int = 512
    act_fn: str = "relu"
    optimizer: str = "adamw"
    weight_decay: float = 0.0
    use_layernorm: bool = True
    dropout_rate: float = 0.0
    init_fn: str = "kaiming"
    tqdm: bool = True  # whether using tqdm or not
    eval_every_n_steps: int = 500

    device: torch.device = None

    def __post_init__(self):
        # Automatically determine the device if not provided
        if self.device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")


def create_dataloader(config: ExperimentConfig):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = datasets.FashionMNIST(
        root="../datasets", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="../datasets", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader


def init_weights(model, init_function):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init_function(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    output_dim: int,
    act_fn: nn.Module,
):
    """return a MLP"""
    layers = [nn.Linear(input_dim, hidden_dim), act_fn]
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def create_network(config: ExperimentConfig):
    """
    Create a MLP with layernorm and dropout (if specified)
    It looks like Linear -> layernorm -> Activation -> Dropout
    """
    act_fn = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }

    init_fn = {
        "kaiming": nn.init.kaiming_normal_,
        "xavier": nn.init.xavier_uniform_,
    }

    assert config.act_fn in act_fn
    assert config.init_fn in init_fn

    base_model = build_mlp(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        output_dim=config.output_dim,
        act_fn=act_fn[config.act_fn],
    )

    layers = []
    for i, layer in enumerate(base_model):
        if isinstance(layer, nn.Linear):
            layers.append(layer)
            # Only add LayerNorm if it's not the last Linear layer
            if (
                hasattr(config, "use_layernorm")
                and config.use_layernorm
                and i < len(base_model) - 1
            ):
                layers.append(nn.LayerNorm(layer.out_features))
        elif isinstance(layer, nn.Module) and not isinstance(layer, nn.Linear):
            layers.append(layer)  # Add the activation layer

        # Only add Dropout if it's not the last layer
        if hasattr(config, "dropout_rate") and config.dropout_rate > 0:
            if (
                isinstance(layer, act_fn[config.act_fn].__class__)
                and i < len(base_model) - 1
            ):
                layers.append(nn.Dropout(p=config.dropout_rate))

    model = nn.Sequential(*layers)

    # Initialize weights
    init_weights(model, init_fn[config.init_fn])

    return model


def create_model(config: ExperimentConfig):
    class MLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.model = create_network(config)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.model(x)

    return MLP(config)


def create_optimizer(config: ExperimentConfig, model):
    optimizer = {
        "sgd": optim.SGD,
        "adamw": optim.AdamW,
    }
    assert config.optimizer in optimizer
    return optimizer[config.optimizer](
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )


class Trainer(BaseTrainer):
    def __init__(self, config: ExperimentConfig):
        model = create_model(config)
        optimizer = create_optimizer(config, model)
        train_loader, val_loader = create_dataloader(config)
        logger = make_summary_writer(config.exp_name, config.config_name)

        super().__init__(
            config=config,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=config.device,
            max_steps=config.max_steps,
            eval_every_n_steps=config.eval_every_n_steps,
            logger=logger,
        )

    def training_step(self, batch):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, targets)
        _, predicted = outputs.max(1)
        accuracy = predicted.eq(targets).float().mean()
        return {"loss": loss, "accuracy": accuracy}

    def validation_step(self, batch):
        return self.training_step(batch)

    def run_experiment(self):
        for _ in self.fit():
            pass
