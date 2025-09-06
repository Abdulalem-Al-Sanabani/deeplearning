import torch
import torch.nn as nn


class FeedForwardBlock(nn.Sequential):
    def __init__(self, hidden_size, dropout):
        super().__init__(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )


class Embedding(nn.Module):
    def __init__(self, vocab_size, context_length, embed_dim, dropout):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(context_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.context_length = context_length

    def forward(self, x):
        device = x.device
        _, T = x.shape
        assert T <= self.context_length, "Sequence length is longer than context length"

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        pos_embed = self.pos_embedding(pos)
        word_embed = self.word_embedding(x)
        return self.dropout(word_embed + pos_embed)
