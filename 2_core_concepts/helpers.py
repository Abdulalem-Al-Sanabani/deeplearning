import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


class InitMethods:
    @staticmethod
    def zero_init(m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def unit_gaussian_init(m):
        if isinstance(m, nn.Linear):
            m.weight.data = torch.randn_like(m.weight)
            m.bias.data = torch.randn_like(m.bias)

    @staticmethod
    def xavier_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def scaled_xavier_init(m, scale):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.weight.data *= scale
            nn.init.zeros_(m.bias)


def analyze_activations(model, loader):
    activations = {}

    def hook_fn(module, input, output):
        activations[module] = output.detach()

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Tanh):
            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        data, _ = next(iter(loader))
        _ = model(data)

    for hook in hooks:
        hook.remove()

    return activations


def plot_activations(ax, activations, init_name):
    for i, (module, activation) in enumerate(activations.items()):
        values = activation.numpy().flatten()
        ax.hist(
            values,
            bins=100,
            range=(-1, 1),
            alpha=0.5,
            label=f"Layer {i+1}",
            density=True,
        )

    ax.set_title(f"Activation Distributions - {init_name}")
    ax.set_xlabel("Activation Value")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")


def analyze_activations_combined(init_methods, SimpleNet, train_loader):
    num_methods = len(init_methods)

    # Calculate the number of rows and columns
    num_cols = min(3, num_methods)  # Max 3 columns
    num_rows = math.ceil(num_methods / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 4 * num_rows))
    fig.suptitle("Activation Distributions for Different Initializations", fontsize=16)

    # Ensure axes is always a 2D array
    if num_methods == 1:
        axes = np.array([[axes]])
    elif num_methods <= 3:
        axes = axes.reshape(1, -1)

    for (name, init_func), ax in zip(init_methods.items(), axes.flatten()):
        model = SimpleNet()
        model.apply(init_func)

        activations = analyze_activations(model, train_loader)
        plot_activations(ax, activations, name)

    # Hide any unused subplots
    for ax in axes.flatten()[num_methods:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def train_model(model, loader, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    losses = []

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                losses.append(loss.item())
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return losses

def evaluate_model(model, loader):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data, target in loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def train_init_methods(SimpleNet, init_methods, loader, test_loader, epochs=3):
    results = {}
    print()
    for name, init_func in init_methods.items():
        print(name.center(80, "-"))
        model = SimpleNet()
        model.apply(init_func)
        results[name] = train_model(model, loader, epochs)
        acc = evaluate_model(model, test_loader)
        print(f"Accuracy {acc:.2f}")

    plt.figure(figsize=(6, 4))
    for name, losses in results.items():
        plt.plot(losses, label=name)
    plt.xlabel("Iterations")
    plt.ylabel("Train Loss")
    plt.title("Impact of Weight Initialization on Training")
    plt.legend()
    plt.show()
