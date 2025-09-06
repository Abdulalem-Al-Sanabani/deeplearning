import torch

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from .transformer_config import TransformerConfig
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
    # Experiment Identification
    config_name: str  # This should be one of the model types in GPTConfig
    exp_name: str = "nmt"

    vocab_size: int = 4096
    lr: float = 5e-4
    scheduler: str = "cosine"
    batch_size: int = 32
    max_steps: int = 25000
    warmup_steps: int = 4000
    eval_every_n_steps: int = 1000
    weight_decay: float = 1e-2
    beam_width: int = 1  # Beam width for beam search decoding, 1 equals greedy decoding

    files: FilePaths = FilePaths()
    transformer_config: TransformerConfig = None
    src_tokenizer: Tokenizer = None
    tgt_tokenizer: Tokenizer = None
    device: torch.device = None
    log_to_tensorboard: bool = True
    save_model: bool = False

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transformer_config = TransformerConfig(
            model_type=self.config_name, vocab_size=self.vocab_size
        )
        self.src_tokenizer, self.tgt_tokenizer = _create_tokenizer(self)

        # context_length should be greater than the maximum sentence length in the data
        # If not, replace it with a new context length
        max_sent_len = _max_sent_len(self.files.train + self.files.val)
        if self.transformer_config.context_length <= max_sent_len:
            self.transformer_config.context_length = _new_context_len(max_sent_len)
            print(
                f"Context length replaced with new value: {self.transformer_config.context_length}"
            )


def _create_tokenizer(config):
    src_tokenizer = Tokenizer()
    src_tokenizer.train(config.files.train, config.vocab_size)

    tgt_tokenizer = Tokenizer()
    tgt_tokenizer.train(config.files.val, config.vocab_size)
    return src_tokenizer, tgt_tokenizer


def _max_sent_len(file_paths: list[Path]):
    """Compute the maximum sentence length in all source and target data."""
    max_sent_len = 0
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                json_obj = json.loads(line.strip())
                max_sent_len = max(max_sent_len, len(json_obj["de"]))
                max_sent_len = max(max_sent_len, len(json_obj["en"]))
    return max_sent_len


def _new_context_len(max_sent_len):
    """
    Compute a new context length based on the maximum sentence length.
    New context length should the nearest 2**k greater than the maximum sentence length.
    """
    pow = math.ceil(math.log(max_sent_len, 2))
    return int(2**pow)
