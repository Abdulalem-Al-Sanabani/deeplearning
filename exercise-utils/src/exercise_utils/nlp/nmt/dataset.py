"""
The shape of data is (batch_size, seq_len) for transformer models.
This is different from the convention we used for RNNs where the shape was (seq_len, batch_size).
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from pathlib import Path
import json
from .tokenizer import Tokenizer


def read_data(file_paths: List[Path]) -> tuple[List[str], List[str]]:
    src_data = []
    tgt_data = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                json_obj = json.loads(line.strip())
                src_data.append(json_obj["de"])
                tgt_data.append(json_obj["en"])

    return src_data, tgt_data


class Multi30kDataset(Dataset):
    def __init__(
        self, file_paths: List[Path], src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer
    ):
        self.src_data, self.tgt_data = read_data(file_paths)
        assert len(self.src_data) == len(
            self.tgt_data
        ), "Number of source and target examples do not match"

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        # special tokens in source language
        self.special_tokens = [
            self.src_tokenizer.pad_id,
            self.src_tokenizer.start_id,
            self.src_tokenizer.end_id,
            self.src_tokenizer.unk_id,
        ]

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        src_text = self.src_data[idx]
        tgt_text = self.tgt_data[idx]

        src_tokens = self.src_tokenizer.encode(src_text)
        tgt_tokens = self.tgt_tokenizer.encode(tgt_text)

        # Add start and end tokens to target
        tgt_tokens = (
            [self.tgt_tokenizer.start_id] + tgt_tokens + [self.tgt_tokenizer.end_id]
        )
        return src_tokens, tgt_tokens


def to_padded_tensor(
    sequences: List[List[int]], pad_id: int, batch_first=True
) -> torch.Tensor:
    """Convert a list of sequences to a padded tensor.

    Args:
        sequences: List of token sequences.
        pad_id: Token ID to use for padding.
        batch_first: If True, the output tensor shape will be (batch_size, seq_len).
                     If False, the shape will be (seq_len, batch_size).

    Returns:
        torch.Tensor: Padded tensor of the specified shape.
    """
    # Find maximum sequence length in the batch
    max_len = max(len(seq) for seq in sequences)

    # Pad each sequence to max_len
    padded_sequences = []
    for seq in sequences:
        # Calculate padding length
        pad_len = max_len - len(seq)
        # Add padding to the right
        padded_seq = seq + [pad_id] * pad_len
        padded_sequences.append(padded_seq)

    # Convert to tensor of shape (batch_size, seq_len)
    padded_tensor = torch.tensor(padded_sequences, dtype=torch.long)

    if not batch_first:
        # Transpose to shape (seq_len, batch_size)
        padded_tensor = padded_tensor.T

    return padded_tensor


class BatchCollator:
    """Handles the collation of batches with padding."""

    def __init__(self, src_pad_id: int, tgt_pad_id: int, batch_first: bool):
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.batch_first = batch_first

    def __call__(
        self, batch: List[Tuple[List[int], List[int]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a batch of sequences into padded tensors."""
        src_sequences, tgt_sequences = zip(*batch)
        return (
            to_padded_tensor(list(src_sequences), self.src_pad_id, self.batch_first),
            to_padded_tensor(list(tgt_sequences), self.tgt_pad_id, self.batch_first),
        )
