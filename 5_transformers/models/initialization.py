import torch.nn as nn
import math


class InitMixin:
    """Mixin class providing initialization methods for transformer models."""

    def _init_weights(self):
        """Apply initialization to the transformer model."""
        # self.apply(self._initialize_weights)
        self._initialize_output_projection()
        # self._initialize_attention_scales()

    def _initialize_weights(self, module):
        """Initialize weights for transformer modules following standard practices."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            d_model = module.embedding_dim
            std = math.sqrt(1.0 / d_model)
            nn.init.normal_(module.weight, mean=0.0, std=std)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _initialize_output_projection(self):
        """Initialize the final output projection layer with smaller weights."""
        if hasattr(self, "out_proj"):
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.out_proj.bias)

    def _initialize_attention_scales(self):
        """Initialize attention scaling factors for transformer blocks."""
        if not hasattr(self, "config"):
            return

        n_embd = self.config.transformer_config.n_embd
        n_head = self.config.transformer_config.n_head
        scale = math.sqrt(n_embd / n_head)

        # Initialize encoder attention scales
        if hasattr(self, "encoder"):
            if hasattr(self.encoder, "self_attn"):
                self.encoder.self_attn.scale = scale

        # Initialize decoder attention scales
        if hasattr(self, "decoder"):
            if hasattr(self.decoder, "self_attn"):
                self.decoder.self_attn.scale = scale
            if hasattr(self.decoder, "cross_attn"):
                self.decoder.cross_attn.scale = scale
