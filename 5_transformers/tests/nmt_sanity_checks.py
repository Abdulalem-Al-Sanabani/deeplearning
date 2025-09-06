import torch


class TestMultiheadAttention:
    @staticmethod
    def test_basic(MultiheadAttention, use_mask=False):
        """Basic forward pass test for MultiheadAttention module."""
        embed_dim = 32
        num_heads = 4
        batch_size = 8
        len_q = 12
        len_k = 14

        if use_mask:
            attn_mask = torch.randint(0, 2, (len_q, len_k)).to(torch.bool)
            key_padding_mask = torch.randint(0, 2, (batch_size, len_k)).to(torch.bool)
        else:
            attn_mask = None
            key_padding_mask = None

        attention = MultiheadAttention(embed_dim, num_heads, 0.1, 0.1)
        query = torch.rand(batch_size, len_q, embed_dim)
        key = torch.rand(batch_size, len_k, embed_dim)
        value = torch.rand(batch_size, len_k, embed_dim)
        attn_output, attn_weights = attention(
            query, key, value, attn_mask, key_padding_mask
        )
        assert attn_output.shape == (batch_size, len_q, embed_dim)
        assert attn_weights.shape == (batch_size, num_heads, len_q, len_k)
        print("Test Passed!")

    def test_specific_mask(MultiheadAttention):
        """Test for a specific attn_mask and key_padding_mask for easier debugging."""
        # Set parameters for a simple test
        embed_dim = 8
        num_heads = 1
        batch_size = 1
        len_q = 3
        len_k = 4

        # Define a simple attention mask (attn_mask) and key padding mask (key_padding_mask)
        attn_mask = torch.tensor(
            [
                [
                    True,
                    False,
                    False,
                    False,
                ],  # Only first key is attended for the first query
                [
                    True,
                    True,
                    False,
                    False,
                ],  # First two keys are attended for the second query
                [True, True, True, True],  # All keys are attended for the third query
            ]
        )

        key_padding_mask = torch.tensor(
            [
                [True, True, True, False],  # The last key is masked (padding)
            ]
        )

        # Initialize the MultiheadAttention module
        attention = MultiheadAttention(
            embed_dim, num_heads, dropout=0.0, attn_dropout=0.0
        )

        query = (
            torch.randn(len_q * embed_dim).float().view(batch_size, len_q, embed_dim)
        )
        key = torch.randn(len_k * embed_dim).float().view(batch_size, len_k, embed_dim)
        value = (
            torch.randn(len_k * embed_dim).float().view(batch_size, len_k, embed_dim)
        )

        # Run the forward pass
        attn_output, attn_weights = attention(
            query, key, value, attn_mask, key_padding_mask
        )

        # Assert the shapes are correct
        assert attn_output.shape == (batch_size, len_q, embed_dim)
        assert attn_weights.shape == (batch_size, num_heads, len_q, len_k)

        # Additional checks for masking behavior
        # Ensure that masked positions have zero attention weights
        print("Masked Attention Weights:")
        print(attn_weights)

        # Check that masked positions are not contributing to the output
        assert torch.allclose(
            attn_weights[0, 0, 0, 1:], torch.tensor([0.0, 0.0, 0.0])
        ), "First query should only attend to the first key"
        assert torch.allclose(
            attn_weights[0, 0, 1, 2:], torch.tensor([0.0, 0.0])
        ), "Second query should only attend to the first two keys"
        assert (
            attn_weights[0, 0, :, 3].sum() == 0
        ), "Last key is masked by key_padding_mask"

        print("Test Passed: Masking operations are correctly applied!")


class TestTransformerEncoderBlock:
    @staticmethod
    def test_basic(TransformerEncoderBlock):
        class TransformerConfig:
            n_embd = 16
            n_head = 2
            dropout = 0.1
            attn_dropout = 0.1

        # Initialize TransformerEncoderBlock
        transformer_config = TransformerConfig()
        encoder_block = TransformerEncoderBlock(transformer_config)

        batch_size = 4
        seq_len = 10
        # Random input data
        src = torch.randn(batch_size, seq_len, transformer_config.n_embd)
        src_len = torch.randint(1, seq_len, (batch_size,))
        src_len[0] = seq_len  # Make sure one sentence is full length

        # Forward pass
        output = encoder_block(src, src_len)

        # Assertions
        assert output.shape == (
            batch_size,
            seq_len,
            transformer_config.n_embd,
        ), f"Expected output shape {(batch_size, seq_len, transformer_config.n_embd)}, but got {output.shape}"
        print("Passed!")


class TestTransformerDecoderBlock:
    @staticmethod
    def test_basic(TransformerDecoderBlock):
        class TransformerConfig:
            n_embd = 16
            n_head = 2
            dropout = 0.1
            attn_dropout = 0.1
            context_length = 256

        # Initialize TransformerDecoderBlock
        transformer_config = TransformerConfig()
        decoder_block = TransformerDecoderBlock(transformer_config)

        batch_size = 4
        tgt_len = 8
        memory_len = 10

        # Random input data for the decoder and encoder memory
        tgt = torch.randn(batch_size, tgt_len, transformer_config.n_embd)
        memory = torch.randn(batch_size, memory_len, transformer_config.n_embd)
        tgt_len_tensor = torch.randint(1, tgt_len, (batch_size,))
        memory_len_tensor = torch.randint(1, memory_len, (batch_size,))

        # Make sure one sentence is full length for testing
        tgt_len_tensor[0] = tgt_len
        memory_len_tensor[0] = memory_len

        # Forward pass
        output = decoder_block(tgt, tgt_len_tensor, memory, memory_len_tensor)

        # Assertions
        assert output.shape == (
            batch_size,
            tgt_len,
            transformer_config.n_embd,
        ), f"Expected output shape {(batch_size, tgt_len, transformer_config.n_embd)}, but got {output.shape}"
        print("Test Passed for TransformerDecoderBlock!")
