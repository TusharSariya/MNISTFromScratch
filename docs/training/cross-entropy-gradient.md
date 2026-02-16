# Cross-Entropy Gradient (Softmax + Cross-Entropy Combined)

## The Punchline

The gradient of cross-entropy loss with respect to the logits (input to softmax) is:

```
dL/d(logits) = predictions - targets
```

That's it. This is the starting gradient that flows backward through the entire network.

---

## Why This Matters

This is the first gradient you compute in backprop. Every other gradient in the network depends on this one flowing backward through the chain rule. If this is wrong, everything is wrong.

The fact that it simplifies to `predictions - targets` is not a coincidence — it's the reason cross-entropy is paired with softmax in classification networks.

---

## The Derivation

We need `dL/dz_i` where `z` are the logits (input to softmax) and `L` is the cross-entropy loss.

### Step 1: Write out both functions

Softmax:
```
p_i = exp(z_i) / Σ_j exp(z_j)
```

Cross-entropy (with one-hot target `y`):
```
L = -Σ_i y_i × log(p_i)
```

Since `y` is one-hot (only `y_k = 1` for the correct class `k`):
```
L = -log(p_k)
```

### Step 2: Chain rule

```
dL/dz_i = dL/dp × dp/dz_i
```

But softmax connects every output to every input, so we need:
```
dL/dz_i = Σ_j (dL/dp_j) × (dp_j/dz_i)
```

### Step 3: dL/dp_j

```
dL/dp_j = -y_j / p_j
```

(From differentiating `-Σ y_j × log(p_j)` with respect to `p_j`)

### Step 4: dp_j/dz_i (the softmax Jacobian)

This has two cases:

When `j = i` (how does `p_i` change when you nudge `z_i`):
```
dp_i/dz_i = p_i × (1 - p_i)
```

When `j ≠ i` (how does `p_j` change when you nudge `z_i`):
```
dp_j/dz_i = -p_j × p_i
```

### Step 5: Combine

```
dL/dz_i = Σ_j (-y_j / p_j) × (dp_j/dz_i)
```

Split into `j = i` and `j ≠ i`:

```
= (-y_i / p_i) × p_i(1 - p_i)  +  Σ_{j≠i} (-y_j / p_j) × (-p_j × p_i)
= -y_i(1 - p_i)                 +  Σ_{j≠i} y_j × p_i
= -y_i + y_i × p_i              +  p_i × Σ_{j≠i} y_j
= -y_i + p_i × (y_i + Σ_{j≠i} y_j)
= -y_i + p_i × Σ_j y_j
```

Since `y` is one-hot, `Σ_j y_j = 1`:

```
= -y_i + p_i × 1
= p_i - y_i
```

That's `predictions - targets`.

---

## Visual Example

```
logits:      [2.0,  1.0,  0.1]        ← raw output of dense layer
softmax:     [0.66, 0.24, 0.10]       ← predictions (probabilities)
target:      [0,    0,    1   ]        ← one-hot (correct class is 2)

gradient:    [0.66, 0.24, -0.90]      ← predictions - targets
              ^^^^  ^^^^  ^^^^^^
              push  push   push
              down  down    UP
```

The gradient is negative for the correct class → when backprop uses this, it increases that logit.
The gradient is positive for wrong classes → it decreases those logits.
The magnitudes tell you how much: the network was 90% wrong about class 2, so it gets a strong push.

### After training converges:

```
logits:      [0.1,  0.2,  5.0]
softmax:     [0.01, 0.01, 0.98]
target:      [0,    0,    1   ]

gradient:    [0.01, 0.01, -0.02]      ← tiny! almost no correction needed
```

When the network is already confident and correct, the gradient is near zero — it stops changing. This is exactly what you want.

---

## What Happens Next

This `(10,)` gradient vector flows backward through the network:

```
gradient (10,)
  → dense_backward        → gradient (1600,), plus dW and db for dense weights
  → dropout_backward      → gradient (1600,)
  → flatten_backward      → gradient (64, 5, 5)
  → maxpool_backward      → gradient (64, 11, 11)
  → relu_backward         → gradient (64, 11, 11)
  → conv2d_backward       → gradient (32, 13, 13), plus dW and db for conv2 weights
  → maxpool_backward      → gradient (32, 26, 26)
  → relu_backward         → gradient (32, 26, 26)
  → conv2d_backward       → plus dW and db for conv1 weights (don't need input grad)
```

At each layer, the gradient gets transformed and passed further back. The `dW` and `db` values at each layer tell the optimizer how to update that layer's weights.

---

## Implementation

```
Parameters:
    predictions — ndarray, shape (10,), output of softmax
    targets     — ndarray, shape (10,), one-hot encoded true label

Returns:
    gradient — ndarray, shape (10,), dL/d(logits)

Steps:
    1. Return: predictions - targets
```

One line. All the complexity is in the derivation, not the code.

---

## Common Confusion

**"Why don't I need to differentiate softmax separately?"**

You can, but then you'd compute the full softmax Jacobian (a 10×10 matrix) and multiply it with the cross-entropy gradient. The combined derivation above shows these cancel out to just `p - y`. Frameworks and from-scratch implementations always use the combined form.

**"Is this gradient with respect to the softmax output or the logits?"**

The logits (input to softmax). This is important: you pass this gradient into `dense_backward`, not into some softmax_backward function. Softmax doesn't have its own backward step — it's fused with cross-entropy.

**"What if I'm not using one-hot targets?"**

The formula `predictions - targets` still works as long as targets sum to 1. One-hot is just the most common case for classification.
