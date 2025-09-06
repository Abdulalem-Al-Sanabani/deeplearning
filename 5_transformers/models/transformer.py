import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Embedding
from .beam_search import BeamSearchMixin
from .initialization import InitMixin

from exercise_utils.nlp.nmt.dataset import to_padded_tensor


class Transformer(nn.Module, InitMixin, BeamSearchMixin):
    """Transformer model for Neural Machine Translation."""

    def __init__(self, EncoderBlock, DecoderBlock, config):
        super().__init__()
        self.config = config

        # Tokenizers
        self.src_tokenizer = config.src_tokenizer
        self.tgt_tokenizer = config.tgt_tokenizer

        # Modules
        vocab_size = config.transformer_config.vocab_size
        n_embd = config.transformer_config.n_embd
        context_length = config.transformer_config.context_length
        dropout = config.transformer_config.dropout

        self.src_embedding = Embedding(vocab_size, context_length, n_embd, dropout)
        self.tgt_embedding = Embedding(vocab_size, context_length, n_embd, dropout)
        self.out_proj = nn.Linear(n_embd, vocab_size)

        n_layer = config.transformer_config.n_layer
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(config.transformer_config) for _ in range(n_layer)]
        )
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(config.transformer_config) for _ in range(n_layer)]
        )

        self._init_weights()

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, src):
        """
        Args:
            src: source sentence tensor of shape (B, S)

        Returns:
            memory: tensor of shape (B, S, C) where C is the embedding dimension
            src_len: int tensor stating the length of each source sentence in the batch. shape (B,)
        """
        src_len = compute_seq_len(src, self.src_tokenizer.pad_id)

        context_length = self.config.transformer_config.context_length
        assert torch.all(src_len <= context_length), "Source sentence is too long"

        src_emb = self.src_embedding(src)
        memory = src_emb
        for encoder_block in self.encoder_blocks:
            memory = encoder_block(memory, src_len)
        return memory, src_len

    def decode(self, tgt, memory, src_len):
        """
        Args:
            tgt: target sentence tensor of shape (B, T)
            memory: tensor of shape (B, S, C) where C is the embedding dimension
            src_len: int tensor stating the length of each source sentence in the batch. shape (B,)

        Returns:
            logits: tensor of shape (B, T, V) where V is the vocabulary size
        """
        tgt_len = compute_seq_len(tgt, self.tgt_tokenizer.pad_id)
        tgt_emb = self.tgt_embedding(tgt)

        output = tgt_emb
        for decoder_block in self.decoder_blocks:
            output = decoder_block(output, tgt_len, memory, src_len)

        logits = self.out_proj(output)
        return logits

    def forward(self, src, tgt):
        """
        Args:
            src: source sentence tensor of shape (B, S)
            tgt: target sentence tensor of shape (B, T)

        Returns:
            loss: scalar tensor
        """
        B, T = tgt.shape
        tgt_input = tgt[:, :-1].contiguous()
        tgt_true = tgt[:, 1:].contiguous()
        memory, src_len = self.encode(src)
        logits = self.decode(tgt_input, memory, src_len)
        V = logits.shape[2]

        loss = F.cross_entropy(
            logits.view(B * (T - 1), V),
            tgt_true.view(B * (T - 1)),
            ignore_index=self.tgt_tokenizer.pad_id,
        )
        return loss

    @torch.no_grad()
    def translate(self, src_sents, beam_width=None, max_len=128):
        """Translate a batch of sentences using beam search decoding.

        Args:
            src_sents (list[str]): list of source sentences
            beam_size (int): beam size
                beam_width=1 is equivalent to greedy decoding.
            max_len (int): maximum length of the generated target sentence

        Returns:
            tgt_sents (list[str]): list of translated sentences
        """
        was_training = self.training
        if was_training:
            self.eval()

        if beam_width is None:
            beam_width = self.config.beam_width

        # Clip max_len to context_length or vice versa
        context_length = self.config.transformer_config.context_length
        max_len = min(max_len, context_length)

        # Tokenize source sentences
        src = self.src_tokenizer.encode_batch(src_sents)
        src = to_padded_tensor(src, self.src_tokenizer.pad_id).to(self.device)

        # Encode source sentences
        memory, src_len = self.encode(src)

        if beam_width == 1:
            # We use greedy decoding explicitly when beam_width=1
            # although it is equivalent to beam search
            # but it is more efficient to use greedy decoding
            sequences = self._greedy_decoding(memory, src_len, max_len)
        else:
            sequences = self.beam_search_decoding(
                memory,
                src_len,
                beam_width,
                max_len,
                self.tgt_tokenizer.start_id,
                self.tgt_tokenizer.end_id,
            )

        # Decode sequences
        tgt_sents = self.tgt_tokenizer.decode_batch(sequences)

        self.training = was_training
        return tgt_sents

    @torch.no_grad()
    def _greedy_decoding(self, memory, src_len, max_len=512):
        B = memory.shape[0]
        context_length = self.config.transformer_config.context_length
        # Initialize target sentences

        tgt = torch.full((B, 1), self.tgt_tokenizer.start_id).long().to(self.device)

        has_ended = torch.zeros(B, dtype=torch.bool).to(self.device)

        for t in range(max_len):
            # Decode next token
            decoder_out = self.decode(tgt, memory, src_len)
            logits = decoder_out[:, -1:, :]  # (B, 1, C)
            next_token = logits.squeeze(1).argmax(dim=-1)  # (B,)

            # Update endness status
            has_ended = has_ended | (next_token == self.tgt_tokenizer.end_id)
            if torch.all(has_ended):
                break

            # Replace tokens in ended sentences with padding
            next_token[has_ended] = self.tgt_tokenizer.pad_id

            # Append next token to target sentences
            next_token = next_token.unsqueeze(1)
            tgt = torch.cat([tgt, next_token], dim=1)

        sequences = tgt.tolist()
        return sequences


def create_key_padding_mask(seq_len, seq_for_check=None):
    """
    Args:
        seq_len: int tensor of shape (B,) containing the length of each sequence in the batch
        seq_for_check: input sequence of shape (B, T, C)
            This is for checking purposes only.

    Returns:
        key_padding_mask: bool tensor of shape (B, T) where T is the maximum sequence length in the batch. False values indicate padding positions.
    """
    assert seq_len.dim() == 1
    B, T = seq_len.size(0), seq_len.max().item()

    if seq_for_check is not None:
        assert (
            seq_len.shape[0] == seq_for_check.shape[0]
        ), f"Batch sizes doesn't match: {seq_len.shape[0]} != {seq_for_check.shape[0]}"
        assert (
            T == seq_for_check.shape[1]
        ), f"The maximum length {T} does not match the actual sequence length {seq_for_check.shape[1]}"

    positions = torch.arange(T).repeat(B, 1).to(seq_len.device)
    key_padding_mask = positions < seq_len.unsqueeze(1)
    return key_padding_mask


def compute_seq_len(seq, pad_id):
    """
    Args:
        tensor: int tensor of shape (B, T)
        pad_id: int, padding token id

    Returns:
        seq_len: int tensor of shape (B,) containing the length of each sequence in the batch
    """
    return (seq != pad_id).sum(1).long()
