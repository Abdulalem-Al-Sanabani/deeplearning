import torch
from torch.utils.data import Dataset
from typing import Tuple
from .tokenizer import LMTokenizer


class LMDataset(Dataset):
    def __init__(self, text: str, context_length: int, tokenizer: LMTokenizer):
        self.context_length = context_length
        self.encoded_data = tokenizer.encode(text)

    def __len__(self) -> int:
        return len(self.encoded_data) - self.context_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.encoded_data[idx : idx + self.context_length + 1]
        input_data = data[:-1]
        target_data = data[1:]

        input_tensor = torch.tensor(input_data, dtype=torch.long)
        target_tensor = torch.tensor(target_data, dtype=torch.long)
        return input_tensor, target_tensor
