import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Literal

from exercise_utils.nlp.nmt.tokenizer import Tokenizer


@dataclass(frozen=True)
class FilePaths:
    _BASE_PATH: Path = Path("../custom_datasets/multi30k")

    train: Tuple[Path] = (
        _BASE_PATH / "multi30k_train.jsonl",
        _BASE_PATH / "multi30k_train_synthetic.jsonl",
    )
    val: Tuple[Path] = (_BASE_PATH / "multi30k_val.jsonl",)


@dataclass
class NMTExperimentConfig:
    rnn_type: Literal["rnn", "lstm"]
    exp_name: str = "nmt"
    config_name: str = None
    vocab_size: int = 4096
    embed_size: int = 256
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.5
    lr: float = 0.001
    batch_size: int = 128
    grad_max_norm: float = 1
    max_steps: int = 10000
    files: FilePaths = FilePaths()
    eval_every_n_steps: int = 500
    weight_decay: float = 1.0
    device: torch.device = None

    src_tokenizer: Tokenizer = None
    tgt_tokenizer: Tokenizer = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.config_name is None:
            self.config_name = self.rnn_type

        self.src_tokenizer, self.tgt_tokenizer = _create_tokenizer(self)


def _create_tokenizer(config):
    src_tokenizer = Tokenizer()
    src_tokenizer.train(config.files.train, config.vocab_size)

    tgt_tokenizer = Tokenizer()
    tgt_tokenizer.train(config.files.val, config.vocab_size)
    return src_tokenizer, tgt_tokenizer
