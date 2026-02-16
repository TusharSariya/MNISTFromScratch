# Cross-Entropy Loss

## What It Does

Cross-entropy loss measures how far your predicted probabilities are from the actual answer. It's the single number that tells the network "how wrong you are" — and the entire point of training is to make this number smaller.

---

## The Setup

After softmax, you have a probability distribution over 10 digits:

```
prediction: [0.02, 0.01, 0.85, 0.03, 0.01, 0.02, 0.01, 0.03, 0.01, 0.01]
                                 ^
                              digit 2 (highest)
```

The true label is a one-hot vector — all zeros except a 1 at the correct class:

```
target:     [0,    0,    1,    0,    0,    0,    0,    0,    0,    0]
                         ^
                      digit 2 (correct answer)
```

---

## The Formula

```
L = -Σ target[i] × log(prediction[i])
```

Since target is one-hot, only one term survives — the correct class:

```
L = -log(prediction[correct_class])
```

That's it. Cross-entropy loss is just the negative log of the probability you assigned to the right answer.

---

## Visual: Why Negative Log?

```
prediction for correct class  →  loss
─────────────────────────────────────
1.0   (perfect)               →  0.0
0.9   (confident, right)      →  0.105
0.5   (coin flip)             →  0.693
0.1   (wrong guess)           →  2.303
0.01  (very wrong)            →  4.605
0.001 (terrible)              →  6.908

        loss
        ▲
   7.0  │                                          ·
        │
   6.0  │
        │
   5.0  │
        │                                    ·
   4.0  │
        │
   3.0  │
        │                              ·
   2.0  │
        │
   1.0  │                       ·
        │                 ·
   0.0  │──────────·─────────────────────────────▶ prediction
        0.0       1.0   0.9   0.5   0.1  0.01
```

Properties:
- **Perfect prediction (1.0)**: loss is 0. No penalty.
- **Good prediction (0.9)**: loss is small. Gentle nudge.
- **Bad prediction (0.01)**: loss is huge. Strong correction.
- **The curve is steep near 0**: being confidently wrong is punished much harder than being slightly unsure. This is what makes the network learn fast early on.

---

## Why Not Just Use (prediction - target)²?

Mean squared error (MSE) works but is worse for classification:

```
                    MSE              Cross-Entropy
prediction: 0.9    (0.9-1)² = 0.01   -log(0.9) = 0.105
prediction: 0.1    (0.1-1)² = 0.81   -log(0.1) = 2.303
prediction: 0.01   (0.01-1)²= 0.98   -log(0.01)= 4.605
```

MSE barely distinguishes between 0.1 and 0.01 (0.81 vs 0.98). Cross-entropy goes from 2.3 to 4.6 — it screams "you're getting worse!" much louder. This means stronger gradients when the network is very wrong, which means faster learning.

Also, the gradient of cross-entropy + softmax simplifies to `predictions - targets` — the cleanest possible gradient. MSE + softmax does not simplify this nicely, making backprop messier.

---

## The Gradient

This is why cross-entropy pairs perfectly with softmax. The combined gradient of softmax + cross-entropy with respect to the logits (the input to softmax) is:

```
dL/d(logits) = predictions - targets
```

Example:

```
predictions: [0.02, 0.01, 0.85, 0.03, 0.01, 0.02, 0.01, 0.03, 0.01, 0.01]
targets:     [0,    0,    1,    0,    0,    0,    0,    0,    0,    0   ]

gradient:    [0.02, 0.01, -0.15, 0.03, 0.01, 0.02, 0.01, 0.03, 0.01, 0.01]
                          ^^^^^^
                          negative = "push this logit up"
                    everything else positive = "push these logits down"
```

The gradient says: increase the score for the correct class, decrease everything else. The magnitude tells you by how much. If the prediction was already 0.99, the gradient would be tiny (-0.01) — almost no correction needed.

This is the starting gradient that flows backward through the entire network.

---

## Numerical Stability

Two things can go wrong:

### 1. prediction = 0 exactly → log(0) = -inf

If softmax outputs exactly 0.0 for the correct class, the loss explodes.

**Fix:** Clip predictions before taking the log:

```python
prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
loss = -np.sum(target * np.log(prediction))
```

### 2. Better: Use log-softmax directly

Instead of computing softmax then log separately, fuse them:

```
log_softmax(z_i) = z_i - max(z) - log(Σ exp(z_j - max(z)))
```

This avoids ever creating a near-zero probability. See `numerical-stability.md` for the full derivation.

For your from-scratch implementation, clipping is fine. Frameworks use log-softmax internally.

---

## Implementation

### cross_entropy_loss(predictions, targets)

```
Parameters:
    predictions — ndarray, shape (10,), output of softmax (probabilities)
    targets     — ndarray, shape (10,), one-hot encoded true label

Returns:
    loss — scalar float

Steps:
    1. Clip predictions to [1e-7, 1-1e-7] to avoid log(0)
    2. Compute: -sum(targets * log(predictions))
    3. Return scalar
```

### cross_entropy_gradient(predictions, targets)

```
Parameters:
    predictions — ndarray, shape (10,), output of softmax
    targets     — ndarray, shape (10,), one-hot encoded true label

Returns:
    gradient — ndarray, shape (10,), dL/d(logits)

Steps:
    1. Return: predictions - targets
    2. That's it. The softmax + cross-entropy gradient simplifies to this.
```

**Note:** This gradient is with respect to the **logits** (input to softmax), not the softmax output. The softmax derivative is already baked in. This is the gradient you pass backward into the dense layer.

---

## In the Training Loop

```
for each image:
    predictions = forward_pass(image)              → (10,) probabilities
    loss = cross_entropy_loss(predictions, target)  → scalar (for logging)
    gradient = cross_entropy_gradient(predictions, target)  → (10,) starting grad
    backward_pass(gradient)                         → update all weights
```

The loss value itself is only for monitoring — you print it to see if training is working. The gradient is what actually drives learning.

---

## Batch Loss

When training on a batch of images, average the loss:

```
batch_loss = (1/N) × Σ cross_entropy_loss(predictions_i, targets_i)
```

This means gradients are also averaged over the batch. Without averaging, bigger batches would produce bigger gradients, and you'd need to adjust the learning rate every time you change batch size.
