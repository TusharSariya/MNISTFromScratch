# Numerical Stability in Neural Networks

Computers use floating-point numbers with limited precision. This causes real problems in deep learning: values overflow to infinity, underflow to zero, or accumulate rounding errors that silently corrupt training. This doc covers the common pitfalls and how to fix them.

---

## Float32 Basics

Neural networks typically use 32-bit floats (float32):

```
Max value:     ~3.4 × 10³⁸
Min positive:  ~1.2 × 10⁻³⁸
Precision:     ~7 decimal digits
```

This means:
- Numbers larger than ~3.4e38 become **inf** (overflow).
- Numbers smaller than ~1.2e-38 become **0** (underflow).
- Adding 1e8 + 1e-1 gives 1e8 — the small value is lost (limited precision).

---

## Problem 1: Softmax Overflow

Softmax is defined as:

```
softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
```

If z_i = 1000, then exp(1000) = inf. Your output becomes NaN.

### The Fix: Subtract the Max

```
softmax(z_i) = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))
```

This is mathematically identical (the max cancels in numerator and denominator), but now the largest exponent is exp(0) = 1. No overflow.

```python
# Unstable
def softmax_bad(z):
    return np.exp(z) / np.sum(np.exp(z))

# Stable
def softmax_good(z):
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)
```

**Example:**

```
z = [1000, 1001, 999]

Bad:   exp([1000, 1001, 999]) = [inf, inf, inf] → NaN

Good:  z - max = [-1, 0, -2]
       exp([-1, 0, -2]) = [0.368, 1.0, 0.135]
       normalize → [0.245, 0.665, 0.090]  ✓
```

Every framework does this automatically. If you implement softmax yourself, you must do this.

---

## Problem 2: Log of Softmax (Log-Sum-Exp)

Cross-entropy loss needs log(softmax(z)):

```
log(softmax(z_i)) = log(exp(z_i) / Σ exp(z_j))
```

Computing softmax first, then log, is numerically bad: softmax can produce values very close to 0, and log(~0) = -inf.

### The Fix: Log-Sum-Exp Trick

```
log(softmax(z_i)) = z_i - log(Σ_j exp(z_j))
```

And the log-sum-exp itself is stabilized:

```
log(Σ exp(z_j)) = max(z) + log(Σ exp(z_j - max(z)))
```

Combined:

```
log_softmax(z_i) = (z_i - max(z)) - log(Σ_j exp(z_j - max(z)))
```

This is why frameworks provide `log_softmax` as a single fused operation instead of `log(softmax(x))`.

```python
# Unstable
log_probs = np.log(softmax(z))          # softmax can give 0 → log(0) = -inf

# Stable
def log_softmax(z):
    z_shifted = z - np.max(z)
    return z_shifted - np.log(np.sum(np.exp(z_shifted)))
```

In PyTorch, `F.log_softmax(x, dim=1)` and `nn.CrossEntropyLoss` (which takes raw logits) use this internally.

---

## Problem 3: Cross-Entropy with One-Hot Targets

Cross-entropy loss:

```
L = -Σ_i y_true_i × log(y_pred_i)
```

If y_pred for the correct class is 0.0 (or very close), log(0) = -inf → loss is inf → gradients explode.

### The Fix: Clamp Predictions

```python
# Clip to avoid log(0)
y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
loss = -np.sum(y_true * np.log(y_pred))
```

Or better: use log_softmax with raw logits (previous section) so you never compute probabilities that can hit exactly 0.

In your Keras model, `loss="categorical_crossentropy"` handles this internally.

---

## Problem 4: Division by Small Numbers

Adam's update rule divides by `sqrt(v) + ε`:

```
parameter -= lr * m_hat / (sqrt(v_hat) + ε)
```

If v_hat is very close to 0, without ε you'd divide by ~0 and get huge or inf updates.

### The Fix: Epsilon

The `ε` (typically 1e-7 or 1e-8) ensures the denominator is never too small:

```python
# Without epsilon — unstable
update = m_hat / np.sqrt(v_hat)          # can be inf if v_hat ≈ 0

# With epsilon — stable
update = m_hat / (np.sqrt(v_hat) + 1e-7) # bounded
```

This shows up everywhere: Adam, RMSProp, batch normalization, layer normalization — anywhere there's a division by a learned or computed quantity.

---

## Problem 5: Exp Overflow in Other Contexts

The sigmoid function:

```
sigmoid(x) = 1 / (1 + exp(-x))
```

If x = -1000, exp(1000) = inf → result = 0/inf = NaN.

### The Fix: Piecewise Computation

```python
def sigmoid_stable(x):
    # For x >= 0: 1 / (1 + exp(-x))    — exp(-x) is small, no overflow
    # For x < 0:  exp(x) / (1 + exp(x)) — exp(x) is small, no overflow
    pos = 1.0 / (1.0 + np.exp(-np.clip(x, 0, None)))
    neg = np.exp(np.clip(x, None, 0)) / (1.0 + np.exp(np.clip(x, None, 0)))
    return np.where(x >= 0, pos, neg)
```

Or equivalently, many frameworks just clip the input:

```python
def sigmoid_simple(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))
```

---

## Problem 6: Accumulation Errors in Summations

Adding many small numbers to a large running sum loses precision:

```python
total = 1e8
total += 1e-1    # 1e-1 is lost — float32 can't represent 100000000.1
```

### The Fix: Kahan Summation or Higher Precision

```python
# Bad: naive sum
total = 0.0
for x in values:
    total += x    # rounding error accumulates

# Better: use float64 for accumulation
total = np.float64(0.0)
for x in values:
    total += np.float64(x)
```

In practice, frameworks compute reductions (loss sums, batch norm statistics) carefully to avoid this.

---

## Problem 7: Gradient Scaling

During backprop, gradients are multiplied through many layers:

```
dL/dW1 = dL/dh_n × dh_n/dh_{n-1} × ... × dh2/dh1 × dh1/dW1
```

If each factor is ~0.5: after 20 layers, gradient ≈ 0.5²⁰ ≈ 0.000001 (vanishing).
If each factor is ~2.0: after 20 layers, gradient ≈ 2²⁰ ≈ 1,000,000 (exploding).

### The Fixes

| Problem | Solution |
|---------|----------|
| Vanishing gradients | ReLU (gradient is 0 or 1), residual connections, careful initialization |
| Exploding gradients | Gradient clipping, batch normalization, proper initialization |

**Gradient clipping** in practice:

```python
# Clip by global norm
max_norm = 1.0
total_norm = sqrt(sum(grad.norm()² for all grads))
if total_norm > max_norm:
    scale = max_norm / total_norm
    for grad in all_grads:
        grad *= scale
```

---

## Problem 8: Weight Initialization

If weights start too large, activations and gradients explode. Too small, and they vanish.

### The Fixes: Smart Initialization

**Glorot/Xavier** (used by Keras Conv2D by default):

```
W ~ Uniform(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
```

**He/Kaiming** (better for ReLU):

```
W ~ Normal(0, √(2/fan_in))
```

These keep the variance of activations roughly constant across layers, preventing both explosion and vanishing at the start of training.

---

## Summary: Where Instability Hides in Your MNIST Model

| Component | Risk | Protection |
|-----------|------|------------|
| **Softmax** (Dense output) | exp overflow | Subtract max (automatic in Keras) |
| **Cross-entropy loss** | log(0) = -inf | Log-sum-exp trick (automatic in Keras) |
| **Adam optimizer** | Division by ~0 | ε = 1e-7 |
| **ReLU activations** | Dead neurons (always 0) | Good initialization (Glorot) |
| **Dropout** | Not a stability issue | Scaling is handled automatically |
| **Backprop chain** | Vanishing/exploding gradients | ReLU + good init + Adam (adapts lr) |
| **Weight init** | Too large or small | Glorot uniform (Keras default) |

Most of this is handled automatically by Keras and PyTorch. It becomes your problem only when you implement from scratch in C/CUDA.

---

## Key Takeaways

1. **Always subtract the max** before exp (softmax, log-sum-exp).
2. **Never compute log(softmax(x))** in two steps — use log_softmax.
3. **Always add ε** when dividing by computed quantities (Adam, normalization).
4. **Use proper weight initialization** (Glorot or He) to start gradients in a safe range.
5. **Frameworks handle most of this**, but understanding it matters when debugging NaN losses, diverging training, or implementing custom layers.
6. If your loss suddenly becomes **NaN** or **inf**, the cause is almost always one of the problems above.
