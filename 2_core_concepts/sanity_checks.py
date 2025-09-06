import unittest
import torch
import torch.nn as nn
import math


class TestDropout(unittest.TestCase):
    def initialize(self, Dropout, p=0.4):
        self.p = p
        self.dropout = Dropout(self.p)

    def test_shape(self):
        # input and output shape should be the same
        self.dropout.train()
        x = torch.ones(15, 16)
        y1 = self.dropout(x)
        self.dropout.eval()
        y2 = self.dropout(x)
        self.assertTrue(x.shape == y1.shape)
        self.assertTrue(x.shape == y2.shape)

    def test_train(self):
        # output should be roughly self.p proportion of zeros
        self.dropout.train()
        n = 100000
        x = torch.ones(n)
        y = self.dropout(x)
        mean = (y == 0).sum().item() / y.numel()
        std = math.sqrt(mean * (1 - mean) / n)
        # 99% confidence interval
        ci_lower = mean - 2.575 * std
        ci_upper = mean + 2.575 * std
        self.assertTrue(
            ci_lower < self.p < ci_upper,
            f"p: {self.p:.3f} not in ({ci_lower:.5f}, {ci_upper:.5f})",
        )

    def test_eval(self):
        # output should be the same as input
        self.dropout.eval()
        x = torch.ones(1001, 1020)
        y_eval = self.dropout(x)
        self.assertTrue(torch.all(y_eval == x))

    def test_backprop(self):
        # sanity check if module is backpropagatable
        self.dropout.train()
        x = torch.ones(1000, 1000, requires_grad=True)
        y = self.dropout(x)
        y.sum().backward()
        self.assertTrue(x.grad is not None)


class TestBatchNorm(unittest.TestCase):
    def initialize(self, BatchNorm, num_features=20):
        self.num_features = num_features
        self.bn = BatchNorm(self.num_features, affine=True)

    def test_shape(self):
        # input and output shape should be the same
        x = torch.ones(100, self.num_features)
        self.bn.train()
        y = self.bn(x)
        self.assertTrue(x.shape == y.shape)
        self.bn.eval()
        y = self.bn(x)
        self.assertTrue(x.shape == y.shape)

    def test_normalized(self):
        # output should have zero mean and unit variance
        self.bn.train()
        x = torch.randn(1000, self.num_features)
        y = self.bn(x)
        mean = y.mean(dim=0)
        var = y.var(dim=0)
        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-2))
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-2))

    def test_backprop(self):
        # sanity check if module is backpropagatable
        x = torch.ones(100, self.num_features, requires_grad=True)
        y = self.bn(x)
        y.sum().backward()
        self.assertTrue(x.grad is not None)


class TestLayerNorm(unittest.TestCase):
    def initialize(self, LayerNorm):
        self.ln = LayerNorm
        self.normalized_shape = [100, 21]
        self.x_shape = [16, 100, 21]

    def test_normalized(self):
        # Test if the output is normalized
        x = torch.randn(*self.x_shape)
        ln = self.ln(self.normalized_shape)
        y = ln(x)
        mean = y.mean(dim=tuple(range(-len(self.normalized_shape), 0)))
        var = y.var(dim=tuple(range(-len(self.normalized_shape), 0)), unbiased=False)
        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-6))
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-6))

    def test_reference_forward(self):
        # Compare to reference implementation: forward pass
        x = torch.randn(*self.x_shape)
        ln = self.ln(self.normalized_shape)
        y = ln(x)
        y_ref = nn.LayerNorm(self.normalized_shape, elementwise_affine=True)(x)
        self.assertTrue(torch.allclose(y, y_ref, atol=1e-6))

    def test_reference_backward(self):
        # Compare to reference implementation: backward pass
        x = torch.randn(*self.x_shape, requires_grad=True)
        ln = self.ln(self.normalized_shape)
        y = ln(x)
        y.sum().backward()

        x_ref = x.clone().detach()
        x_ref.requires_grad_(True)
        y_ref = nn.LayerNorm(self.normalized_shape, elementwise_affine=True)(x_ref)
        y_ref.sum().backward()

        self.assertTrue(torch.allclose(x.grad, x_ref.grad, atol=1e-6))

    def test_one_step_update(self):
        # Compare to reference implementation: one step of optimization
        x = torch.randn(*self.x_shape, requires_grad=True)
        x_ref = x.clone().detach()
        x_ref.requires_grad_(True)

        ln = self.ln(self.normalized_shape)
        ln_ref = nn.LayerNorm(self.normalized_shape, elementwise_affine=True)

        lr = 0.01
        optimizer_custom = torch.optim.SGD(ln.parameters(), lr=lr)
        optimizer_ref = torch.optim.SGD(ln_ref.parameters(), lr=lr)

        optimizer_custom.zero_grad()
        optimizer_ref.zero_grad()

        y = ln(x)
        loss = y.sum()

        y_ref = ln_ref(x_ref)
        loss_ref = y_ref.sum()

        loss.backward()
        loss_ref.backward()

        optimizer_custom.step()
        optimizer_ref.step()

        self.assertTrue(torch.allclose(y, y_ref, atol=1e-6))
        self.assertTrue(torch.allclose(x.grad, x_ref.grad, atol=1e-6))


class BaseTestOptimizer(unittest.TestCase):
    def initialize(self, optimizer_class, lr=1e-4):
        self.optimizer_class = optimizer_class
        self.lr = lr

    def setUp(self):
        torch.manual_seed(42)
        self.model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1))
        self.criterion = nn.MSELoss()
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr)
        self.x = torch.randn(5, 10)  # Move input to setUp
        self.target = torch.randn(5, 1)  # Move target to setUp

    def _run_forward_and_backward(self):
        output = self.model(self.x)
        loss = self.criterion(output, self.target)
        loss.backward()
        return loss

    def test_gradient_update(self):
        # test if the optimizer changes the parameters
        initial_params = [param.clone() for param in self.model.parameters()]
        self._run_forward_and_backward()
        self.optimizer.step()

        for param, initial_param in zip(self.model.parameters(), initial_params):
            self.assertFalse(
                torch.equal(param, initial_param),
                "Parameter did not change after optimizer step",
            )

    def test_training_loop(self):
        # test if the optimizer can reduce the loss for a simple model
        num_iterations = 1000
        initial_loss = None
        final_loss = None

        for i in range(num_iterations):
            self.optimizer.zero_grad()
            loss = self._run_forward_and_backward()

            if i == 0:
                initial_loss = loss.item()
            if i == num_iterations - 1:
                final_loss = loss.item()

            self.optimizer.step()

        print("========================")
        print(f"Initial Loss: {initial_loss:.6f}")
        print(f"Final Loss: {final_loss:.6f}")
        print("========================")

        self.assertLess(
            final_loss,
            initial_loss,
            "Final loss is not less than initial loss after training",
        )


def create_test_suite(test_class, setUpArgs):
    """Create a test suite for the given test class and initialization arguments."""
    suite = unittest.TestSuite()
    test_loader = unittest.TestLoader()

    for test_name in test_loader.getTestCaseNames(test_class):
        test_case = test_class(test_name)
        test_case.initialize(
            **setUpArgs
        )  # Manually call initialize with the necessary arguments
        suite.addTest(test_case)

    return suite


def run_tests(**kwargs):
    suite = unittest.TestSuite()

    if "Dropout" in kwargs:
        suite.addTests(create_test_suite(TestDropout, {"Dropout": kwargs["Dropout"]}))
    if "BatchNorm" in kwargs:
        suite.addTests(
            create_test_suite(TestBatchNorm, {"BatchNorm": kwargs["BatchNorm"]})
        )
    if "LayerNorm" in kwargs:
        suite.addTests(
            create_test_suite(TestLayerNorm, {"LayerNorm": kwargs["LayerNorm"]})
        )
    if "optimizer" in kwargs:
        suite.addTests(
            create_test_suite(
                BaseTestOptimizer, {"optimizer_class": kwargs["optimizer"]}
            )
        )

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
