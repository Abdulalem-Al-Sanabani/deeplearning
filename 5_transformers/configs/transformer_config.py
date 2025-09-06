from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    # Base model configuration
    vocab_size: int

    # Either specify model type or the following parameters
    n_layer: Optional[int] = None
    n_head: Optional[int] = None
    n_embd: Optional[int] = None

    model_type: Optional[str] = None

    # Dropout parameters
    dropout = 0.1
    attn_dropout = 0.1

    def __post_init__(self):
        if self.model_type is not None:
            # Dictionary of preset model configurations
            model_configs = {
                # GPT-1
                "openai-gpt": dict(
                    n_layer=12, n_head=12, n_embd=768, context_length=2048
                ),
                # GPT-2 configs
                "gpt2": dict(n_layer=12, n_head=12, n_embd=768, context_length=2048),
                "gpt2-medium": dict(
                    n_layer=24, n_head=16, n_embd=1024, context_length=2048
                ),
                "gpt2-large": dict(
                    n_layer=36, n_head=20, n_embd=1280, context_length=2048
                ),
                "gpt2-xl": dict(
                    n_layer=48, n_head=25, n_embd=1600, context_length=2048
                ),
                # Gophers
                "gopher-44m": dict(
                    n_layer=8, n_head=16, n_embd=512, context_length=2048
                ),
                # Smaller configs
                "gpt-mini": dict(n_layer=6, n_head=12, n_embd=384, context_length=256),
                "gpt-micro": dict(n_layer=2, n_head=12, n_embd=384, context_length=256),
                "gpt-nano": dict(n_layer=2, n_head=4, n_embd=128, context_length=256),
            }

            if self.model_type not in model_configs:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # Update configuration based on model type
            config = model_configs[self.model_type]
            self.n_layer = config["n_layer"]
            self.n_head = config["n_head"]
            self.n_embd = config["n_embd"]
            self.context_length = config["context_length"]

            assert (
                self.n_embd % self.n_head == 0
            ), f"n_embd must be divisible by n_head, got {self.n_embd} % {self.n_head} != 0"
