# MNIST Digit Classification

A convolutional neural network for handwritten digit recognition using Keras with a PyTorch backend.

## Model Architecture

- Conv2D(32, 3x3) → MaxPool(2x2)
- Conv2D(64, 3x3) → MaxPool(2x2)
- Flatten → Dropout(0.5) → Dense(10, softmax)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install keras torch torchvision
```

## Usage

```bash
python mnist.py
```

Trains for 5 epochs on the MNIST dataset (60k train / 10k test images) and prints test accuracy. Expects ~99% accuracy.
