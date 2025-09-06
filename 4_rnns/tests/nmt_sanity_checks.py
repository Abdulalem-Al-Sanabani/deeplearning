import torch
import torch.nn as nn


class TestNMTEncoder:
    @staticmethod
    def test_output_shape(
        NMTEncoder,
        batch_size=32,
        seq_len=20,
        vocab_size=1000,
        embed_size=256,
        hidden_size=512,
        num_layers=2,
    ):
        """Test if encoder outputs have correct shapes"""

        class Config:
            def __init__(self):
                self.vocab_size = vocab_size
                self.embed_size = embed_size
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.dropout = 0.1
                self.rnn_type = "lstm"

        config = Config()
        encoder = NMTEncoder(config)
        src = torch.randint(0, vocab_size, (batch_size, seq_len))
        pad_id = 0

        last_hidden = encoder(src, pad_id)
        h_n, c_n = last_hidden

        expected_hidden_shape = (num_layers, batch_size, hidden_size)
        if h_n.shape == expected_hidden_shape and c_n.shape == expected_hidden_shape:
            print("Test passed: Encoder output shapes are correct")
        else:
            print(
                f"Test failed: Expected shape {expected_hidden_shape}, got {h_n.shape}"
            )


class TestNMTDecoder:
    @staticmethod
    def test_output_shape(
        NMTDecoder,
        batch_size=32,
        seq_len=20,
        vocab_size=1000,
        embed_size=256,
        hidden_size=512,
        num_layers=2,
    ):
        """Test if decoder outputs have correct shapes"""

        class Config:
            def __init__(self):
                self.vocab_size = vocab_size
                self.embed_size = embed_size
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.dropout = 0.1
                self.rnn_type = "lstm"

        config = Config()
        decoder = NMTDecoder(config)

        tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
        h_0 = torch.randn(num_layers, batch_size, hidden_size)
        c_0 = torch.randn(num_layers, batch_size, hidden_size)
        init_hidden = (h_0, c_0)
        pad_id = 0

        logits, (h_n, c_n) = decoder(tgt, init_hidden, pad_id)

        expected_logits_shape = (batch_size, seq_len, vocab_size)
        expected_hidden_shape = (num_layers, batch_size, hidden_size)

        shapes_correct = (
            logits.shape == expected_logits_shape
            and h_n.shape == expected_hidden_shape
            and c_n.shape == expected_hidden_shape
        )

        if shapes_correct:
            print("Test passed: Decoder output shapes are correct")
        else:
            print("Test failed: Output shapes are incorrect")
            print(
                f"Expected logits shape: {expected_logits_shape}, got: {logits.shape}"
            )
            print(f"Expected hidden shape: {expected_hidden_shape}, got: {h_n.shape}")
