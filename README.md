# Deep Learning Exercises

Welcome to the **Deep Learning** exercises!
In these exercises, you'll implement fundamental concepts from scratch and recreate key ideas from influential papers.

## Getting Started

You'll need Python and several deep learning libraries, which are listed in `environment.yaml`.

You can use `conda` to set up your environment (we recommend to install it via [miniforge](https://github.com/conda-forge/miniforge)):

```bash
conda env create -f environment.yaml
conda activate deeplearning
```

## Course Overview

### Week 1: Getting Started with PyTorch
We'll begin with PyTorch fundamentals - tensors, automatic differentiation, and hardware-accelerated operations (if available).
You'll implement basic optimization tasks and build your first neural networks, culminating in a handwritten digit classifier.

### Week 2: Deep Learning Building Blocks
This week focuses on the essential components that make deep networks work.
You'll implement various initialization techniques, normalization layers, dropout, and modern optimizers.
Through experiments on Fashion-MNIST, you'll see how these details affect training dynamics.

### Week 3: Computer Vision and CNNs
You'll dive into modern computer vision by implementing ResNets and semantic segmentation models.
We'll explore different backbone architectures the tradeoffs between accuracy and computational efficiency.

### Week 4: Sequential Data and RNNs
We'll tackle sequence modeling by building RNNs from scratch.
You'll create a character-level language model to generate Shakespeare-like text and implement a German-English translation system.
This introduces key NLP concepts like tokenization and sequence handling.

### Week 5: The Transformer Architecture
You'll implement the fundamental building blocks of many modern neural networks - from attention mechanisms to complete Transformers.
The exercises cover both autoregressive text generation (GPT-style) and neural machine translation tasks.

### Week 6: Scaling and Multimodality
In the final week, we'll look at how models behave as they grow larger, examining scaling laws empirically.
You'll also build a simple system that can generate captions for images, bringing together concepts from vision and language processing.

Each week builds on the previous ones, taking you from fundamentals to current research topics. Good luck and happy learning!
