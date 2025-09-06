import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from exercise_utils.nlp.lm.tokenizer import CharacterTokenizer


@dataclass
class TrainFile:
    path: Path = Path("../custom_datasets/tiny-shakespeare.txt")
    train_ratio: float = 0.9  # Ratio of data to use for training


@dataclass
class LMExperimentConfig:
    rnn_type: Literal["custom_rnn", "custom_lstm", "pytorch_rnn", "pytorch_lstm"]
    exp_name: str = "lm"
    config_name: str = None
    tokenizer: CharacterTokenizer = None
    train_file: TrainFile = field(default_factory=lambda: TrainFile())
    batch_size: int = 1024
    max_steps: int = 10000
    lr: float = 1e-3
    weight_decay: float = 1e-2
    device: torch.device = None

    seq_len: int = 64
    hidden_size: int = 64
    eval_every_n_steps = 200
    generate_text_length: int = 1000

    def __post_init__(self):
        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        self.tokenizer = _create_tokenizer(self)
        self.vocab_size = self.tokenizer.vocab_size

        if self.config_name is None:
            self.config_name = self.rnn_type


def _create_tokenizer(config):
    tokenizer = CharacterTokenizer()
    tokenizer.train([config.train_file.path])
    return tokenizer
