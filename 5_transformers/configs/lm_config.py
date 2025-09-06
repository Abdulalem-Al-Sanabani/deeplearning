import torch

from dataclasses import dataclass
from pathlib import Path

from .transformer_config import TransformerConfig
from exercise_utils.nlp.lm.tokenizer import CharacterTokenizer, LMTokenizer


@dataclass(frozen=True)
class TrainFile:
    path: Path = Path("../custom_datasets/tiny-shakespeare.txt")
    train_ratio: float = 0.9  # Ratio of data to use for training


@dataclass
class LMExperimentConfig:
    # Experiment Identification
    config_name: str  # This should be one of the model types in GPTConfig
    exp_name: str = "lm"

    vocab_size: int = 4096
    lr: float = 1e-3
    scheduler: str = "cosine"
    batch_size: int = 32
    max_steps: int = 10000
    warmup_steps: int = 500
    weight_decay: float = 1e-2
    eval_every_n_steps: int = 500
    generate_text_length: int = 1000

    char_tokenizer: bool = True  # Use character-level tokenizer
    train_file: TrainFile = TrainFile()
    transformer_config: TransformerConfig = None
    device: torch.device = None

    def __post_init__(self):
        # Decide device if not specified
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize GPTConfig
        self.transformer_config = TransformerConfig(
            model_type=self.config_name, vocab_size=self.vocab_size
        )

        # Initialize tokenizer and overwrite vocab_size if using character tokenizer
        if self.char_tokenizer and self.vocab_size is not None:
            print(
                "Character Tokenizer is used. Vocab size will be determined by the training data."
            )
        self.tokenizer, self.vocab_size = _create_tokenizer(self)
        self.transformer_config.vocab_size = self.vocab_size


def _create_tokenizer(config):
    """
    Since vocab_size is determined by the CharacterTokenizer (if we use it), it needs to be initialized in the config.
    """
    if config.char_tokenizer:
        tokenizer = CharacterTokenizer()
        tokenizer.train([config.train_file.path])
        vocab_size = tokenizer.vocab_size
    else:
        tokenizer = LMTokenizer()
        tokenizer.train(
            files=[config.train_file.path],
            vocab_size=config.vocab_size,
        )
        vocab_size = tokenizer.vocab_size
    return tokenizer, vocab_size
