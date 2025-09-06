from tokenizers import SentencePieceBPETokenizer
from typing import List


class SpecialTokens:
    PAD = "<pad>"
    START = "<s>"  # <sos>
    END = "</s>"  # <eos>
    UNK = "<unk>"


class Tokenizer:
    def __init__(self):
        self.tokenizer = SentencePieceBPETokenizer()
        self.special_tokens = [
            SpecialTokens.PAD,
            SpecialTokens.START,
            SpecialTokens.END,
            SpecialTokens.UNK,
        ]
        self.pad_id = None
        self.unk_id = None
        self.start_id = None
        self.end_id = None

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size() + len(self.special_tokens)

    def train(self, files, vocab_size):
        self.tokenizer.train(
            files=[str(file) for file in files],
            vocab_size=vocab_size - len(self.special_tokens),
            special_tokens=self.special_tokens,
            show_progress=False,
        )
        # After training, get the IDs for special tokens
        self.pad_id = self.tokenizer.token_to_id(SpecialTokens.PAD)
        self.unk_id = self.tokenizer.token_to_id(SpecialTokens.UNK)
        self.start_id = self.tokenizer.token_to_id(SpecialTokens.START)
        self.end_id = self.tokenizer.token_to_id(SpecialTokens.END)

    def encode(self, text: str) -> List[int]:
        """Tokenize and convert tokens to IDs."""
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        """Convert IDs to tokens and join to form a string."""
        return self.tokenizer.decode(ids)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Batched version of encode."""
        return [self.encode(text) for text in texts]

    def decode_batch(self, sequences: List[List[int]]) -> List[str]:
        """Batched version of decode."""
        return [self.decode(seq) for seq in sequences]
