# Adam Optimizer

Adam (Adaptive Moment Estimation) is an optimizer that combines two ideas:

1. **Momentum** — keep a running average of past gradients (like a ball rolling downhill with inertia).
2. **RMSProp** — keep a running average of past squared gradients (to adapt the learning rate per-parameter).

## The Problem Adam Solves

With plain gradient descent (SGD), every parameter gets the same learning rate. This causes issues:

- Parameters with large gradients overshoot.
- Parameters with small gradients update too slowly.
- Noisy gradients (from mini-batches) cause erratic updates.

Adam fixes all three by tracking **how big** and **how variable** each parameter's gradients have been.

## How It Works

Adam maintains two values per parameter:

- **m** (first moment) — exponential moving average of the gradient. This is the **direction** signal (momentum).
- **v** (second moment) — exponential moving average of the squared gradient. This is the **scale** signal (per-parameter learning rate).

### Update Rules

At each step t:

```
g = gradient of loss w.r.t. parameter

m = β1 * m + (1 - β1) * g          # update momentum
v = β2 * v + (1 - β2) * g²         # update scale

m_hat = m / (1 - β1^t)             # bias correction
v_hat = v / (1 - β2^t)             # bias correction

parameter = parameter - lr * m_hat / (sqrt(v_hat) + ε)
```

### What Each Part Does

| Term | Role |
|------|------|
| **m** (momentum) | Smooths out noisy gradients. If gradients keep pointing the same direction, m grows and the update is larger. |
| **v** (scale) | Tracks how large gradients have been. Parameters with big gradients get a smaller effective learning rate. |
| **Bias correction** | m and v start at 0, so early values are too small. Dividing by (1 - β^t) corrects this. |
| **ε** (epsilon) | Tiny number (1e-7 or 1e-8) to prevent division by zero. |

## Default Hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| **lr** | 0.001 | Base learning rate |
| **β1** | 0.9 | Momentum decay rate (how much past gradients matter) |
| **β2** | 0.999 | Scale decay rate (how much past squared gradients matter) |
| **ε** | 1e-7 | Numerical stability constant |

These defaults work well for most tasks. You rarely need to tune β1, β2, or ε. The learning rate (lr) is the main one to adjust.

## Why Adam Is Popular

- **Works out of the box** — the defaults are good for most problems.
- **Fast convergence** — momentum helps it move quickly through flat regions.
- **Adaptive** — each parameter gets its own effective learning rate.
- **Handles sparse gradients** — useful for embeddings and NLP tasks.

## In Your Code

```python
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
```

This uses Adam with all default hyperparameters. Equivalent to:

```python
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
    metrics=["accuracy"],
)
```

## Adam vs Other Optimizers

| Optimizer | Pros | Cons |
|-----------|------|------|
| **SGD** | Simple, good generalization with tuning | Needs careful lr scheduling, slow |
| **SGD + Momentum** | Faster than plain SGD | Still needs lr tuning |
| **RMSProp** | Adaptive lr per parameter | No momentum |
| **Adam** | Adaptive + momentum, fast, good defaults | Can generalize slightly worse than tuned SGD on some tasks |

## Simple Intuition

Think of optimization as navigating a hilly landscape in fog:

- **SGD** — you take fixed-size steps downhill based on the slope right under your feet.
- **Momentum** — you're a heavy ball: you build up speed going downhill and don't stop immediately at every bump.
- **Adam** — you're a heavy ball with smart shoes: you build up speed (momentum) AND your shoes automatically take smaller steps on steep slopes and bigger steps on flat ground (adaptive learning rate).
