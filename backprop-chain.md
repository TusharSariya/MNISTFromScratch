# Backpropagation and the Chain Rule

Backpropagation is the algorithm that computes **how much each weight in the network contributed to the error**, so we know how to update it. It's just the chain rule from calculus, applied systematically from the output layer back to the input.

---

## The Core Idea

You have a loss value (a single number: "how wrong was the prediction"). You need to compute:

```
dL/dW  for every weight W in the network
```

"How much does the loss change if I nudge this weight a tiny bit?"

Once you have these gradients, the optimizer (Adam, SGD, etc.) uses them to update the weights.

---

## The Chain Rule (Single Variable)

If `y = f(g(x))`, then:

```
dy/dx = dy/dg × dg/dx
```

Derivative of the whole thing = derivative of the outer × derivative of the inner.

**Example:**

```
y = (3x + 2)²

Let g = 3x + 2,  y = g²

dy/dg = 2g
dg/dx = 3
dy/dx = 2g × 3 = 2(3x + 2) × 3 = 6(3x + 2)
```

---

## The Chain Rule in a Neural Network

A neural network is a chain of composed functions:

```
input → layer1 → layer2 → layer3 → ... → loss
  x        h1       h2       h3             L
```

Where:

```
h1 = f1(x)
h2 = f2(h1)
h3 = f3(h2)
L  = loss(h3, y_true)
```

To get dL/dW1 (gradient for a weight in layer 1):

```
dL/dW1 = dL/dh3 × dh3/dh2 × dh2/dh1 × dh1/dW1
```

This is the chain rule applied through every layer between the loss and the weight. Each layer contributes one link in the chain.

---

## Forward Pass vs Backward Pass

### Forward pass (left to right)

Compute the output step by step:

```
x = input image
h1 = relu(W1 @ x + b1)       # layer 1
h2 = relu(W2 @ h1 + b2)      # layer 2
y_pred = softmax(W3 @ h2 + b3)  # output layer
L = cross_entropy(y_pred, y_true)  # loss
```

Each layer saves its input and output (needed for backward pass).

### Backward pass (right to left)

Start from the loss and work backward:

```
dL/dy_pred = ...              # derivative of loss (known formula)
dL/dh2 = W3ᵀ @ dL/dy_pred    # pass gradient through layer 3
dL/dh1 = W2ᵀ @ dL/dh2        # pass gradient through layer 2
```

And at each layer, also compute the weight gradients:

```
dL/dW3 = dL/dy_pred @ h2ᵀ
dL/dW2 = dL/dh2 @ h1ᵀ
dL/dW1 = dL/dh1 @ xᵀ
```

---

## How a Single Layer Backpropagates

Every layer implements two functions:

```
forward(input) → output             # store input for later
backward(dL/doutput) → dL/dinput    # also compute dL/dW, dL/db
```

### Dense layer: y = Wx + b

```
Forward:
    y = W @ x + b                   # save x

Backward (given dL/dy):
    dL/dW = dL/dy @ xᵀ             # weight gradient
    dL/db = dL/dy                   # bias gradient (just sum over batch)
    dL/dx = Wᵀ @ dL/dy             # pass to previous layer
```

### ReLU: y = max(0, x)

```
Forward:
    y = max(0, x)                   # save mask: where x > 0

Backward (given dL/dy):
    dL/dx = dL/dy * mask            # gradient passes through where x > 0
                                    # gradient is 0 where x ≤ 0
```

ReLU acts like a gate: it lets the gradient through where the input was positive and blocks it where the input was negative.

### MaxPooling: y = max of each 2×2 window

```
Forward:
    y = max of each pool window     # save which position had the max

Backward (given dL/dy):
    dL/dx = 0 everywhere except at the max positions
            where it equals dL/dy
```

Only the "winner" of each pool window gets the gradient. Everyone else gets zero.

### Softmax + Cross-Entropy Loss

These are usually combined because the combined gradient is simple:

```
dL/dz = y_pred - y_true
```

Where z is the input to softmax (logits). If the true label is class 3 and the model predicts [0.1, 0.05, 0.1, 0.7, 0.05]:

```
dL/dz = [0.1, 0.05, 0.1, 0.7, 0.05] - [0, 0, 0, 1, 0]
       = [0.1, 0.05, 0.1, -0.3, 0.05]
```

The gradient is negative for the correct class (push it higher) and positive for wrong classes (push them lower).

---

## Full Backward Pass Through Your MNIST Model

```
Forward:
  X (28,28,1)
  → Conv2D(32,3) + ReLU  → (26,26,32)
  → MaxPool(2)           → (13,13,32)
  → Conv2D(64,3) + ReLU  → (11,11,64)
  → MaxPool(2)           → (5,5,64)
  → Flatten              → (1600,)
  → Dropout(0.5)         → (1600,)
  → Dense(10) + Softmax  → (10,)
  → Cross-Entropy Loss   → scalar L

Backward:
  dL/dlogits = y_pred - y_true                          # (10,)
  ↓
  Dense backward: dL/dW_dense, dL/db_dense, dL/dh       # dL/dh is (1600,)
  ↓
  Dropout backward: zero out same positions as forward   # (1600,)
  ↓
  Flatten backward: reshape to (5,5,64)                  # no parameters
  ↓
  MaxPool backward: gradient to max positions only       # (11,11,64)
  ↓
  Conv2D(64) backward: dL/dW2, dL/db2, dL/dh            # (13,13,32)
  ↓
  MaxPool backward: gradient to max positions only       # (26,26,32)
  ↓
  Conv2D(32) backward: dL/dW1, dL/db1, dL/dX            # (28,28,1)
  ↓
  (stop — no need to compute gradient for the input image)
```

After this single backward pass, you have gradients for every weight in the network. Adam then uses these to update all the weights.

---

## Why Backprop Is Efficient

A naive approach would compute dL/dW for each weight independently — that would require a separate forward pass for each weight. With millions of weights, this is impossibly slow.

Backprop reuses intermediate results. Each layer's backward pass:
1. Receives dL/doutput from the layer above.
2. Computes dL/dinput (to pass down) and dL/dW (to store).

This means the entire backward pass has roughly the **same computational cost as one forward pass**. For N weights, you get all N gradients in one backward pass instead of N forward passes.

---

## What Frameworks Do For You

When you write `model.fit(...)` in Keras or `loss.backward()` in PyTorch, the framework:

1. Builds a **computation graph** during the forward pass (tracking which operations produced which tensors).
2. Walks the graph **in reverse** during the backward pass.
3. Calls each operation's backward function automatically.
4. Accumulates gradients for every parameter.

This is called **automatic differentiation** (autograd). You never write backward functions yourself — unless you're implementing everything in raw C/CUDA.

---

## Common Gradient Flow Problems

### Vanishing Gradients

The chain rule multiplies gradients through layers:

```
dL/dW1 = dL/dh3 × dh3/dh2 × dh2/dh1 × dh1/dW1
```

If each factor is < 1 (e.g., 0.2 × 0.3 × 0.1 = 0.006), gradients shrink exponentially. Early layers barely learn.

**Fixes:** ReLU (gradient is 0 or 1, no shrinking), residual connections (skip connections that add a shortcut for gradients), batch normalization.

### Exploding Gradients

If factors are > 1, gradients grow exponentially. Weights get enormous updates and training diverges.

**Fixes:** Gradient clipping (cap the gradient magnitude), careful weight initialization, lower learning rate.

### Dead ReLU

If a ReLU neuron's input is always negative, its gradient is always 0, and it never updates. It's permanently "dead."

**Fixes:** Leaky ReLU (small nonzero gradient for negative inputs), careful initialization, lower learning rate.

---

## Key Takeaways

1. Backprop is just the **chain rule applied layer by layer** from loss to input.
2. Every layer needs a **forward** and a **backward** function.
3. The backward pass computes **weight gradients** (for updates) and **input gradients** (to pass to the previous layer).
4. The whole backward pass costs roughly the **same as one forward pass** — this is what makes training practical.
5. Gradients can **vanish** (too small) or **explode** (too large) in deep networks — modern architectures use ReLU, skip connections, and normalization to manage this.
6. Frameworks like Keras and PyTorch handle all of this automatically via **autograd**.
