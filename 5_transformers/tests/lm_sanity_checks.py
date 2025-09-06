import torch


class TestGPTBlock:
    @staticmethod
    def test_basic(GPTBlock):
        """Basic forward pass test for GPTBlock."""

        class TransformerConfig:
            n_embd = 32
            n_head = 4
            dropout = 0.1
            attn_dropout = 0.1
            context_length = 32

        # Initialize config and block
        transformer_config = TransformerConfig()
        block = GPTBlock(transformer_config)

        # Create sample input
        batch_size = 8
        seq_len = 16
        x = torch.rand(batch_size, seq_len, transformer_config.n_embd)

        # Forward pass
        output = block(x)

        # Verify output shape matches input
        assert output.shape == (
            batch_size,
            seq_len,
            transformer_config.n_embd,
        ), f"Expected output shape {(batch_size, seq_len, transformer_config.n_embd)}, got {output.shape}"

        print("Test Passed!")
