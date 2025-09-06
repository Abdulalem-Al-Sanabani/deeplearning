"""
Test case generation example (with reference implementation):
    1. SanityChecks.generate_expected_log_sum_exp(log_sum_exp)
    2. SanityChecks.generate_expected_huber_loss(huber_loss)

Verification example:
    1. SanityChecks.verify_implementation(log_sum_exp)
    2. SanityChecks.verify_implementation(huber_loss)

Note:
    Currently, the output of the test functions should be a scalar value of numpy array.
"""

import numpy as np
import pickle
from pathlib import Path


class SanityChecks:
    OUTPUT_PATH = Path(__file__).resolve().parent / "expected_outputs"

    @classmethod
    def _get_file_path(cls, func, file_name):
        func_name = func.__name__
        if file_name is None:
            path = cls.OUTPUT_PATH / f"{func_name}.pkl"
        else:
            path = cls.OUTPUT_PATH / file_name
        return path

    @classmethod
    def _generate_ground_truth(cls, ref_func, inputs, file_name=None):
        """Generate and save ground truth data using reference implementation"""
        ref_output = ref_func(*inputs)
        test_data = {
            "inputs": inputs,
            "output": ref_output,
        }

        if not cls.OUTPUT_PATH.exists():
            cls.OUTPUT_PATH.mkdir()

        save_path = cls._get_file_path(ref_func, file_name)

        with open(save_path, "wb") as f:
            pickle.dump(test_data, f)

    @classmethod
    def generate_expected_log_sum_exp(cls, log_sum_exp, num_samples=100):
        np.random.seed(0)
        a, b, e = np.random.randn(3)
        alpha, beta = np.random.rand(2)
        N = np.exp(np.random.randn(num_samples))  # positive values
        D = np.exp(np.random.randn(num_samples))  # positive values
        inputs = (a, b, e, alpha, beta, N, D)
        cls._generate_ground_truth(log_sum_exp, inputs)

    @classmethod
    def generate_expected_huber_loss(cls, huber_loss, num_samples=100):
        np.random.seed(0)
        y_true = np.random.randn(num_samples)
        y_pred = np.random.randn(num_samples)
        inputs = (y_true, y_pred)
        cls._generate_ground_truth(huber_loss, inputs)

    @classmethod
    def verify_implementation(cls, test_func, file_name=None):
        """Verify an implementation against saved ground truth"""
        # Load ground truth
        ground_truth_path = cls._get_file_path(test_func, file_name)

        with open(ground_truth_path, "rb") as f:
            test_data = pickle.load(f)

        # Unpack inputs and run test function
        inputs = test_data["inputs"]
        test_output = test_func(*inputs)
        ref_output = test_data["output"]

        # Compare outputs
        is_close = np.allclose(test_output, ref_output)
        if not is_close:
            max_diff = np.max(np.abs(test_output - ref_output))
            raise AssertionError(
                f"{test_func.__name__} output mismatch! Max difference: {max_diff}"
            )
        else:
            print("Implementation is correct!")
