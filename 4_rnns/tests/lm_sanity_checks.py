import torch
import torch.nn as nn
import os


class TestCharacterTokenizer:
    @staticmethod
    def _test_cases():
        return [
            ("Hello World!", "Hello!", torch.tensor([2, 5, 6, 6, 7, 1])),
            ("abcdefg", "bed", torch.tensor([1, 4, 3])),
            ("12345", "54321", torch.tensor([4, 3, 2, 1, 0])),
        ]

    @staticmethod
    def test_encode(CharacterTokenizer):
        test_cases = TestCharacterTokenizer._test_cases()

        for corpus, text, tokens in test_cases:
            tokenizer = CharacterTokenizer(corpus)
            encoded = tokenizer.encode(text)
            assert torch.equal(encoded, tokens), f"Encode failed for '{text}'"
            assert tokenizer.vocab_size == len(
                set(corpus)
            ), f"Incorrect vocab size for '{corpus}'"
        print("Encode tests passed.")

    @staticmethod
    def test_decode(CharacterTokenizer):
        test_cases = TestCharacterTokenizer._test_cases()

        for corpus, text, tokens in test_cases:
            tokenizer = CharacterTokenizer(corpus)
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Decode failed for indices {tokens}"
        print("Decode tests passed.")


class TestDataset:
    # Constants
    SEQUENCE_LENGTH = 10
    DUMMY_VOCAB_SIZE = 100
    DUMMY_DATA_LENGTH = 1000
    TEST_INDEX = 3

    class DummyTokenizer:
        def encode(self, text):
            return torch.arange(len(text))

        @property
        def vocab_size(self):
            return TestDataset.DUMMY_VOCAB_SIZE

    class DummyConfig:
        def __init__(self, file_path, tokenizer):
            self.file_path = file_path
            self.tokenizer = tokenizer
            self.seq_len = TestDataset.SEQUENCE_LENGTH

    @staticmethod
    def test_getitem(dataset_class, file_path):
        tokenizer = TestDataset.DummyTokenizer()
        config = TestDataset.DummyConfig(file_path, tokenizer)

        # Ensure the file exists
        assert os.path.exists(file_path), f"File not found: {file_path}"

        dataset = dataset_class(config, split="train")

        # Override encoded_data for consistent testing
        dataset.encoded_data = torch.arange(TestDataset.DUMMY_DATA_LENGTH)

        # Test __getitem__
        x, y = dataset[TestDataset.TEST_INDEX]

        # Check shapes
        expected_shape = torch.Size([TestDataset.SEQUENCE_LENGTH])
        assert (
            x.shape == expected_shape
        ), f"Input shape mismatch: expected {expected_shape}, got {x.shape}"
        assert (
            y.shape == expected_shape
        ), f"Target shape mismatch: expected {expected_shape}, got {y.shape}"

        # Check if y is shifted by one position
        assert torch.equal(x[1:], y[:-1]), "Target not correctly shifted"

        print("ShakespeareDataset __getitem__ test passed.")


class TestRNN:
    @staticmethod
    def test_output_shape(
        RNN,
        input_size=10,
        hidden_size=20,
        batch_size=32,
        seq_len=10,
    ):
        """
        Test if the output shapes of the custom RNN are correct.
        """
        custom_rnn = RNN(input_size, hidden_size)

        # Generate input data
        x = torch.randn(batch_size, seq_len, input_size)
        h_0 = torch.randn(1, batch_size, hidden_size)

        # Forward pass
        with torch.no_grad():
            custom_output, custom_hn = custom_rnn(x, h_0)

        # Check output shapes
        expected_output_shape = (batch_size, seq_len, hidden_size)
        expected_hn_shape = (1, batch_size, hidden_size)

        shape_correct = (
            custom_output.shape == expected_output_shape
            and custom_hn.shape == expected_hn_shape
        )

        if shape_correct:
            print("Test passed: Output shapes are correct.")
        else:
            print("Test failed: Output shapes are incorrect.")
            print(
                f"Expected output shape: {expected_output_shape}, got: {custom_output.shape}"
            )
            print(
                f"Expected hidden state shape: {expected_hn_shape}, got: {custom_hn.shape}"
            )

    @staticmethod
    def test_output_equality(
        RNN,
        input_size=10,
        hidden_size=20,
        batch_size=32,
        seq_len=10,
        rtol=1e-5,
        atol=1e-6,
    ):
        """
        The function tests if the output of the custom RNN matches the output of the reference RNN.
        """
        ref_rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        custom_rnn = RNN(input_size, hidden_size)
        TestRNN._set_custom_params_to_ref(custom_rnn, ref_rnn)

        # Generate input data
        x = torch.randn(batch_size, seq_len, input_size)
        h_0 = torch.randn(1, batch_size, hidden_size)

        # Forward pass for both models
        with torch.no_grad():
            custom_output, custom_hn = custom_rnn(x, h_0)
            ref_output, ref_hn = ref_rnn(x, h_0)

        # Check output equality
        output_equal = torch.allclose(custom_output, ref_output, rtol=rtol, atol=atol)
        hn_equal = torch.allclose(custom_hn, ref_hn, rtol=rtol, atol=atol)

        if output_equal and hn_equal:
            print("Test passed: Custom RNN output matches reference RNN output.")
        else:
            print("Test failed: Custom RNN output differs from reference RNN output.")
            if not output_equal:
                print(
                    f"Max output difference: {(custom_output - ref_output).abs().max().item()}"
                )
            if not hn_equal:
                print(
                    f"Max hidden state difference: {(custom_hn - ref_hn).abs().max().item()}"
                )

    @staticmethod
    def _set_custom_params_to_ref(custom_rnn, ref_rnn):
        with torch.no_grad():
            custom_rnn.Wxh.copy_(ref_rnn.weight_ih_l0)
            custom_rnn.Whh.copy_(ref_rnn.weight_hh_l0)
            custom_rnn.bh.copy_(ref_rnn.bias_ih_l0 + ref_rnn.bias_hh_l0)


class TestLSTM:
    @staticmethod
    def test_output_shape(
        LSTM,
        input_size=10,
        hidden_size=20,
        batch_size=32,
        seq_len=10,
    ):
        """
        Test if the output shapes of the custom LSTM are correct.
        """
        custom_lstm = LSTM(input_size, hidden_size)

        # Generate input data
        x = torch.randn(batch_size, seq_len, input_size)
        h_0 = torch.randn(1, batch_size, hidden_size)
        c_0 = torch.randn(1, batch_size, hidden_size)

        # Forward pass
        with torch.no_grad():
            custom_output, (custom_hn, custom_cn) = custom_lstm(x, (h_0, c_0))

        # Check output shapes
        expected_output_shape = (batch_size, seq_len, hidden_size)
        expected_hn_shape = (1, batch_size, hidden_size)
        expected_cn_shape = (1, batch_size, hidden_size)

        shape_correct = (
            custom_output.shape == expected_output_shape
            and custom_hn.shape == expected_hn_shape
            and custom_cn.shape == expected_cn_shape
        )

        if shape_correct:
            print("Test passed: LSTM output shapes are correct.")
        else:
            print("Test failed: LSTM output shapes are incorrect.")
            print(
                f"Expected output shape: {expected_output_shape}, got: {custom_output.shape}"
            )
            print(
                f"Expected hidden state shape: {expected_hn_shape}, got: {custom_hn.shape}"
            )
            print(
                f"Expected cell state shape: {expected_cn_shape}, got: {custom_cn.shape}"
            )

    @staticmethod
    def test_output_equality(
        LSTM,
        input_size=10,
        hidden_size=20,
        batch_size=32,
        seq_len=10,
        rtol=1e-5,
        atol=1e-6,
    ):
        """
        The function tests if the output of the custom LSTM matches the output of the reference LSTM.
        """
        ref_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        custom_lstm = LSTM(input_size, hidden_size)
        TestLSTM._set_custom_params_to_ref(custom_lstm, ref_lstm)

        # Generate input data
        x = torch.randn(batch_size, seq_len, input_size)
        h_0 = torch.randn(1, batch_size, hidden_size)
        c_0 = torch.randn(1, batch_size, hidden_size)

        # Forward pass for both models
        with torch.no_grad():
            custom_output, (custom_hn, custom_cn) = custom_lstm(x, (h_0, c_0))
            ref_output, (ref_hn, ref_cn) = ref_lstm(x, (h_0, c_0))

        # Check output equality
        output_equal = torch.allclose(custom_output, ref_output, rtol=rtol, atol=atol)
        hn_equal = torch.allclose(custom_hn, ref_hn, rtol=rtol, atol=atol)
        cn_equal = torch.allclose(custom_cn, ref_cn, rtol=rtol, atol=atol)

        if output_equal and hn_equal and cn_equal:
            print("Test passed: Custom LSTM output matches reference LSTM output.")
        else:
            print("Test failed: Custom LSTM output differs from reference LSTM output.")
            if not output_equal:
                print(
                    f"Max output difference: {(custom_output - ref_output).abs().max().item()}"
                )
            if not hn_equal:
                print(
                    f"Max hidden state difference: {(custom_hn - ref_hn).abs().max().item()}"
                )
            if not cn_equal:
                print(
                    f"Max cell state difference: {(custom_cn - ref_cn).abs().max().item()}"
                )

    @staticmethod
    def _set_custom_params_to_ref(custom_lstm, ref_lstm):
        with torch.no_grad():
            # Weight matrices
            custom_lstm.Wih.copy_(ref_lstm.weight_ih_l0)
            custom_lstm.Whh.copy_(ref_lstm.weight_hh_l0)

            # Bias terms
            custom_lstm.bih.copy_(ref_lstm.bias_ih_l0)
            custom_lstm.bhh.copy_(ref_lstm.bias_hh_l0)
