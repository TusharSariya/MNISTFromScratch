# ReLU (Rectified Linear Unit)

## What It Actually Does

```
relu(x) = max(0, x)
```

If positive, pass through. If negative, output zero. Applied independently to every element in the tensor.

That's it. Everything else is why.

---

## Why It Exists

### The Problem: Linearity

Conv2D and Dense layers are linear operations — multiply by weights, add bias:

```
output = W * input + b
```

Stacking two linear operations without anything in between:

```
layer1: y = W1 * x + b1
layer2: z = W2 * y + b2

Substitute:
z = W2 * (W1 * x + b1) + b2
z = (W2 * W1) * x + (W2 * b1 + b2)
z = W3 * x + b3
```

Two layers collapse into one. The depth is an illusion — the network is no more powerful than a single layer. This holds for any number of layers: 100 linear layers stacked together = 1 linear layer.

A linear model can only learn straight decision boundaries. It cannot learn curves, edges, or any pattern that isn't a flat hyperplane. It would fail at MNIST because digit shapes are not linear.

### The Fix: Non-Linearity

Insert a non-linear function between layers. `max(0, x)` cannot be factored out:

```
z = W2 * relu(W1 * x + b1) + b2
```

This cannot be simplified to `W3 * x + b3`. Each layer now adds real representational power. The network can learn increasingly complex patterns as you add depth.

---

## Why ReLU Specifically

### Before ReLU: Sigmoid and Tanh

```
sigmoid(x) = 1 / (1 + e^(-x))     output range: (0, 1)
tanh(x) = (e^x - e^-x) / (e^x + e^-x)   output range: (-1, 1)
```

Both squash the input into a small range. They work, but they cause a critical problem in deep networks.

### The Vanishing Gradient Problem

During backpropagation, the gradient flows backward through each layer. At each activation function, it gets multiplied by the activation's derivative:

```
sigmoid derivative: max 0.25 (at x=0), approaches 0 at extremes
tanh derivative:    max 1.0 (at x=0), approaches 0 at extremes
```

For sigmoid, each layer multiplies the gradient by at most 0.25. After 10 layers:

```
0.25^10 = 0.000001
```

The gradient reaching the first layer is essentially zero. Early layers stop learning. This is the vanishing gradient problem — it killed deep networks for years.

### ReLU's Gradient

```
relu'(x) = 0  if x < 0
relu'(x) = 1  if x > 0
```

When the input is positive, the gradient passes through unchanged — multiplied by 1. No shrinking. Backpropagation can flow through many layers without vanishing.

When the input is negative, the gradient is 0. That neuron contributes nothing to the update. This is fine — not every neuron needs to fire for every input.

---

## Dead ReLU Problem

If a neuron's input becomes permanently negative (due to a large weight update), its output is always 0 and its gradient is always 0. It never recovers — it's "dead."

This happens when:
- Learning rate is too high (large weight updates overshoot)
- Bad weight initialization (many neurons start in negative territory)

In practice, it's rarely a serious issue for well-tuned networks. But variants exist to address it:

```
Leaky ReLU:       max(0.01 * x, x)    — small slope for negatives instead of 0
ELU:              x if x > 0, α(e^x - 1) if x ≤ 0
GELU:             x * Φ(x)            — smooth approximation, used in transformers
```

For MNIST with a small network, plain ReLU works perfectly.

---

## Where It Goes in the Network

ReLU is applied after the linear operation, before the next layer:

```
Conv2D(32, 3x3) → ReLU → MaxPool(2x2) → Conv2D(64, 3x3) → ReLU → MaxPool(2x2) → ...
```

In your Keras code, `activation="relu"` in Conv2D means the layer does the convolution AND the ReLU in one step. They're conceptually separate operations.

The final layer uses softmax instead of ReLU because the output needs to be probabilities (0 to 1, summing to 1). ReLU would just clip negatives to 0, which isn't a probability distribution.

---

## In Your Network

After your first conv2d forward pass produces a `(32, 26, 26)` tensor, ReLU is:

```
output = np.maximum(0, conv_output)
```

One numpy call. Every negative value becomes 0. Every positive value stays the same. The shape doesn't change — `(32, 26, 26)` in, `(32, 26, 26)` out.

Computationally trivial compared to the convolution. But without it, your network can't learn digit shapes.

---

## The Backward Pass

For backpropagation, you need the derivative of ReLU:

```
gradient_out = gradient_in * (input > 0)
```

`(input > 0)` produces a mask of 1s and 0s. Multiply element-wise. Where the original input was positive, the gradient passes through. Where it was negative, the gradient is blocked.

You need to store the original input (or the mask) during the forward pass to use during backprop. This is true for all activation functions — the forward pass saves values needed by the backward pass.
