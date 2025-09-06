from tokenizers import SentencePieceBPETokenizer
from typing import List


class CharacterTokenizer:
    def __init__(self):
        self.chars = None
        self.char2id = None
        self.id2char = None
        self._vocab_size = None

    def train(self, files: list[str]):
        text = ""
        for file in files:
            with open(file, "r") as f:
                text += f.read()
        self.chars = sorted(set(text))
        self.char2id = {ch: i for i, ch in enumerate(self.chars)}
        self.id2char = {i: ch for ch, i in self.char2id.items()}
        self._vocab_size = len(self.chars)

    @property
    def vocab_size(self):
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        return [self.char2id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join([self.id2char[id] for id in ids])

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]

    def decode_batch(self, sequences: List[List[int]]) -> List[str]:
        return [self.decode(seq) for seq in sequences]


class LMTokenizer:
    def __init__(self):
        self.tokenizer = SentencePieceBPETokenizer()

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def train(self, files, vocab_size):
        self.tokenizer.train(
            files=[str(file) for file in files],
            vocab_size=vocab_size,
            show_progress=False,
        )

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
