import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import FeedForwardBlock, Embedding


class GPT(nn.Module):
    def __init__(self, BlockCls, config):
        super().__init__()
        config = config.transformer_config
        self.config = config
        self.embedding = Embedding(
            config.vocab_size, config.context_length, config.n_embd, config.dropout
        )

        self.blocks = nn.ModuleList([BlockCls(config) for _ in range(config.n_layer)])
        self.norm = nn.LayerNorm(config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.out_proj(x)
        return x

    def compute_loss(self, input_data, target_data):
        logits = self.forward(input_data)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_data.view(-1))
        return loss

    @torch.no_grad()
    def generate(model, context_ids, max_new_tokens: int = 500, temperature=1.0):
        """
        Generate text using the model with proper temperature scaling

        Args:
            context_ids: tokens indices of shape (T, )
            max_new_tokens: maximum number of tokens to generate
            temperature: controls randomness (higher = more random, lower = more deterministic)
        """
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        T = context_ids.shape[0]
        context_ids = context_ids.view(1, -1).clone().to(device)

        for _ in range(max_new_tokens):
            # Get the last context_length tokens
            x = context_ids[:, -model.config.context_length :]

            # Get logits and apply temperature
            logits = model(x)[:, -1, :]
            logits = logits / temperature

            # Apply softmax and sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            context_ids = torch.cat([context_ids, next_token], dim=-1)

        if was_training:
            model.train(was_training)
        return context_ids.squeeze()[T:]
