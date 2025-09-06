import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from collections import OrderedDict
import random
from torch.utils.data import DataLoader, Subset
from sklearn.datasets import make_moons, make_circles


class ToyDataset:
    @staticmethod
    def _add_bias_column(X):
        return torch.cat((torch.ones(X.shape[0], 1), X), 1)

    @classmethod
    def sine(cls):
        """
        1D regression dataset with a sine wave.
        """
        X = torch.linspace(0, 1, 32).view(-1, 1)
        y = torch.sin(2 * torch.pi * 2 * X)
        return X, y

    @classmethod
    def sinc(cls):
        """
        1D regression dataset with a sinc function.
        """
        X = torch.linspace(-10, 10, 100).view(-1, 1)
        y = torch.where(X != 0, torch.sin(X) / X, torch.tensor(1.0))
        return X, y

    @classmethod
    def binary_blob(cls):
        """
        2D binary classification dataset with two blobs.
        """
        data1 = 0.4 * torch.randn(200, 2) + 3
        data2 = 0.8 * torch.randn(200, 2) - 1
        X = torch.cat((data1, data2), 0)
        X = cls._add_bias_column(X)
        y = torch.cat((torch.zeros(200), torch.ones(200)))
        return X, y

    @classmethod
    def two_moons(cls, n_samples=400, noise=0.1):
        """
        2D binary classification dataset with two moons.
        """
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=0)
        X = torch.tensor(X, dtype=torch.float32)
        X = cls._add_bias_column(X)
        y = torch.tensor(y, dtype=torch.float32)
        return X, y

    @classmethod
    def xor(cls, n_samples=400):
        """
        2D binary classification dataset with XOR pattern.
        """
        X = torch.rand(n_samples, 2) * 2 - 1
        y = ((X[:, 0] > 0) != (X[:, 1] > 0)).float()
        X = cls._add_bias_column(X)
        return X, y

    @classmethod
    def concentric_circles(cls, n_samples=400, noise=0.1, factor=0.3):
        """
        2D binary classification dataset with two concentric circles.
        """
        X, y = make_circles(
            n_samples=n_samples, noise=noise, factor=factor, random_state=0
        )
        X = torch.tensor(X, dtype=torch.float32)
        X = cls._add_bias_column(X)
        y = torch.tensor(y, dtype=torch.float32)
        return X, y

    @classmethod
    def list_datasets(cls):
        torch.manual_seed(0)
        return [
            method
            for method in dir(cls)
            if not method.startswith("__")
            and method != "get_dataset"
            and callable(getattr(cls, method))
        ]

    @classmethod
    def get_dataset(cls, name, **kwargs):
        if hasattr(cls, name):
            return getattr(cls, name)(**kwargs)
        else:
            raise ValueError(
                f"Dataset '{name}' not found. Available datasets: {cls.list_datasets()}"
            )


def get_dataset(name, **kwargs):
    return ToyDataset.get_dataset(name, **kwargs)


def plot_logistic_regression(data, labels, weights, feature_transform_fn=None):
    colors = ["#1f77b4", "#ff7f0e"]  # Blue for Class 0, Orange for Class 1

    plt.figure(figsize=(5, 4))
    plt.scatter(
        data[labels == 0, 1],
        data[labels == 0, 2],
        c=colors[0],
        label="Class 0",
        alpha=0.6,
    )
    plt.scatter(
        data[labels == 1, 1],
        data[labels == 1, 2],
        c=colors[1],
        label="Class 1",
        alpha=0.6,
    )

    x_min, x_max = data[:, 1].min().item() - 1, data[:, 1].max().item() + 1
    y_min, y_max = data[:, 2].min().item() - 1, data[:, 2].max().item() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = torch.tensor(
        np.c_[np.ones_like(xx.ravel()), xx.ravel(), yy.ravel()], dtype=torch.float32
    )

    if feature_transform_fn is not None:
        grid = feature_transform_fn(grid)

    with torch.no_grad():
        z = torch.sigmoid(grid @ weights).reshape(xx.shape)

    custom_cmap = plt.cm.colors.LinearSegmentedColormap.from_list("custom", colors)

    plt.contourf(xx, yy, z.numpy(), levels=8, cmap=custom_cmap, alpha=0.2)
    plt.contour(
        xx,
        yy,
        z.numpy(),
        levels=[0.5],
        colors="k",
        linestyles="--",
        linewidths=1,
        alpha=0.5,
    )

    plt.title("Logistic Regression Decision Boundary")
    plt.legend()
    plt.show()


def create_small_loader(loader, num_samples=1000, batch_size=1000, seed=42):
    assert isinstance(loader, DataLoader)
    random.seed(seed)
    torch.manual_seed(seed)

    dataset = loader.dataset
    total_samples = len(dataset)
    num_samples = min(num_samples, total_samples)

    subset_indices = random.sample(range(total_samples), num_samples)
    small_dataset = Subset(dataset, subset_indices)
    small_loader = DataLoader(
        small_dataset,
        batch_size=min(
            batch_size, num_samples
        ),  # Ensure batch_size doesn't exceed num_samples
        shuffle=False,
    )

    return small_loader


def extract_activations(model, data_loader):
    activations = OrderedDict()

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    # Register hooks for each layer
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(get_activation(name)))

    model.eval()
    all_activations = OrderedDict()
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.view(inputs.shape[0], -1)
            _ = model(inputs)
            for name, activation in activations.items():
                if name not in all_activations:
                    all_activations[name] = []
                all_activations[name].append(activation)
            all_labels.append(labels)

    for hook in hooks:
        hook.remove()

    # Combine activations from all batches
    combined_activations = OrderedDict()
    for name, activations_list in all_activations.items():
        combined_activations[name] = torch.cat(activations_list)

    labels = torch.cat(all_labels)

    return combined_activations, labels


def apply_tsne(activations, perplexity=70, n_components=2):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    tsne_results = OrderedDict()
    for name, activation in activations.items():
        tsne_results[name] = tsne.fit_transform(
            activation.reshape(activation.shape[0], -1).numpy()
        )
    return tsne_results


def map_to_custom_names(activations):
    layer_display_names = {
        "network.0": "Input Layer",
        "network.1": "Hidden Layer (ReLU)",
        "network.2": "Output Layer",
    }
    updated_activations = OrderedDict()
    for key, value in activations.items():
        if key in layer_display_names:
            updated_activations[layer_display_names[key]] = value
        else:
            raise ValueError(f"Unknown layer name: {key}")
    return updated_activations


def plot_tsne(tsne_results, labels, num_classes):
    num_plots = len(tsne_results)
    rows = (num_plots + 2) // 3  # Ceiling division
    cols = min(num_plots, 3)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (layer_name, tsne_result) in enumerate(tsne_results.items()):
        ax = axes[i]
        scatter = ax.scatter(
            tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap="tab10", s=5
        )
        ax.set_title(layer_name)
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove any unused subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.colorbar(scatter, ax=axes, label="Class", ticks=range(num_classes))

    return fig


def show_tsne_plot(model, loader, num_samples=1000):
    small_loader = create_small_loader(
        loader, num_samples=num_samples
    )  # sample 1000 images from the dataset
    activations, labels = extract_activations(model, small_loader)
    activations = map_to_custom_names(activations)
    tsne_results = apply_tsne(activations)
    fig = plot_tsne(tsne_results, labels, num_classes=10)
    plt.show()


def make_grid(images, predictions=None, labels=None):
    """
    Helper function for logging image prediction results
    """
    n = images.shape[0]
    grid_n = int(n**0.5)  # grid will be grid_n x grid_n
    fig, axes = plt.subplots(grid_n, grid_n, figsize=(grid_n * 2, grid_n * 2))

    for i in range(grid_n):
        for j in range(grid_n):
            idx = i * grid_n + j
            axes[i, j].imshow(images[idx].squeeze(), cmap="gray")
            # kids, don't code like this at home
            title = []
            if predictions is not None:
                title.append(f"Pred: {predictions[idx]}")
            if labels is not None:
                title.append(f"Label: {labels[idx]}")
            axes[i, j].set_title(" | ".join(title))
            axes[i, j].axis("off")

    plt.tight_layout()
    return fig
