# Dropout: A Complete Breakdown

## What It Actually Does

During training, Dropout randomly sets a fraction of input values to zero on each forward pass. During inference, it does nothing. That's the entire mechanism. It's a regularization technique to prevent overfitting.

---

## The Operation

Given a vector and dropout rate of 0.5 (as in your model):

```
Training — each forward pass generates a different random mask:

Input:    [0.8,  1.2,  0.3,  0.5,  0.9,  1.1,  0.4,  0.7]
Mask:     [  1,    0,    1,    0,    1,    0,    1,    0 ]   (random)
Output:   [1.6,  0.0,  0.6,  0.0,  1.8,  0.0,  0.8,  0.0]

Next forward pass with same input:
Mask:     [  0,    1,    0,    1,    1,    1,    0,    0 ]   (different random)
Output:   [0.0,  2.4,  0.0,  1.0,  1.8,  2.2,  0.0,  0.0]
```

Notice the outputs are **scaled up**. The surviving values are multiplied by `1 / (1 - rate)`. At rate=0.5, survivors are multiplied by 2. This is called **inverted dropout**.

```
Inference (no dropout, no scaling):

Input:    [0.8,  1.2,  0.3,  0.5,  0.9,  1.1,  0.4,  0.7]
Output:   [0.8,  1.2,  0.3,  0.5,  0.9,  1.1,  0.4,  0.7]
```

---

## Why Scale the Surviving Values

Without scaling, the expected sum of the layer's output would differ between training and inference.

During training with rate=0.5, about half the values are zeroed. If you don't compensate, the expected output is half what it would be at inference when nothing is dropped.

**Without scaling (original dropout — Hinton 2012):**
- Training: values pass through unmodified, some are zeroed
- Inference: multiply all values by (1 - rate) to compensate

**With inverted scaling (standard today):**
- Training: surviving values are divided by (1 - rate), i.e. multiplied by 2 at rate=0.5
- Inference: values pass through unmodified

Inverted dropout is preferred because it makes inference simpler — no special handling needed at test time.

The math:

```
Training expected value per neuron:
E[output] = (1 - rate) × (value / (1 - rate)) + rate × 0 = value

Inference:
output = value

They match. ✓
```

---

## Parameters

Zero. Dropout has nothing to learn. It's a stochastic operation with one hyperparameter (the rate) set by you.

---

## Why It Works

### The Intuition

Imagine a team of 10 people where person A always handles the critical task because they're the best at it. The rest atrophy. If person A is absent one day, the team fails.

Dropout randomly removes team members during training. Everyone has to be competent. No single neuron can dominate or become a bottleneck. The network learns **redundant representations**.

### The Formal Explanations

**1. Prevents co-adaptation**

Without dropout, neurons develop complex co-dependencies. Neuron A fires only when neurons B and C fire together. This creates fragile, entangled representations. Dropout breaks these dependencies by making any neuron unreliable during training.

**2. Approximate ensemble**

Each training step uses a different random subset of neurons — effectively a different sub-network. A network with n neurons and dropout has 2^n possible sub-networks. The final model at inference approximates an **ensemble average** of all these sub-networks.

Ensembles are one of the most reliable ways to improve generalization. Dropout gives you an exponential ensemble for free (roughly).

**3. Noise injection as regularization**

Dropout adds multiplicative Bernoulli noise (each value is multiplied by either 0 or 1/(1-rate)). This is similar to other noise-based regularizers. The noise prevents the model from fitting to exact training values and forces it to learn robust features.

### The Original Paper

"Dropout: A Simple Way to Prevent Neural Networks from Overfitting" — Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov (2014). One of the most cited papers in deep learning.

---

## The Dropout Rate

The rate is the **fraction of values set to zero**, not the fraction kept.

```
Dropout(0.5):  50% zeroed, 50% kept — the most common choice
Dropout(0.2):  20% zeroed, 80% kept — lighter regularization
Dropout(0.8):  80% zeroed, 20% kept — aggressive regularization
Dropout(0.0):  no dropout — equivalent to removing the layer
Dropout(1.0):  everything zeroed — the network learns nothing
```

Common conventions:
- **0.5** for hidden dense layers (your model uses this)
- **0.2 - 0.3** after convolutional layers (if used at all)
- **0.1** in transformers (attention and feedforward sublayers)

Higher dropout = stronger regularization = more generalization but slower training and potentially underfitting.

---

## Training vs Inference Behavior

This is the most important practical detail. Dropout behaves differently in the two modes:

```python
# PyTorch
model.train()    # dropout is active
model.eval()     # dropout is disabled

# Keras
model.fit(...)           # dropout is active
model.predict(...)       # dropout is disabled
model.evaluate(...)      # dropout is disabled
```

If you forget to call `model.eval()` in PyTorch before inference, your predictions will be **noisy and non-deterministic**. This is one of the most common PyTorch bugs.

```python
# Wrong — predictions will vary each time
output = model(test_input)

# Correct
model.eval()
with torch.no_grad():
    output = model(test_input)
```

---

## Backpropagation

The gradient through dropout follows the same mask used in the forward pass:

```
Forward:
Input:    [0.8,  1.2,  0.3]
Mask:     [  1,    0,    1]
Output:   [1.6,  0.0,  0.6]     (scaled by 1/(1-0.5) = 2)

Backward (gradient from next layer = [0.5, 0.3, 0.1]):
Gradient: [1.0,  0.0,  0.2]     (same mask, same scaling)
```

Neurons that were dropped get **zero gradient** — they don't update this step. They'll get gradients on other steps when the random mask includes them. Over many steps, all neurons receive roughly equal total gradient.

---

## Where to Place Dropout

In your model:

```
Flatten       → (batch, 1600)
Dropout(0.5)  → (batch, 1600)    ← between flatten and dense
Dense(10)     → (batch, 10)
```

This is the most standard placement — between dense layers.

### Common placement patterns:

**After dense layers (standard):**
```
Dense → ReLU → Dropout → Dense → ReLU → Dropout → Dense
```

**After conv layers (less common, lower rate):**
```
Conv → ReLU → Pool → Dropout(0.25) → Conv → ReLU → Pool → Dropout(0.25)
```

**Inside transformers:**
```
Attention → Dropout(0.1) → Add&Norm → FFN → Dropout(0.1) → Add&Norm
```

### Where NOT to put dropout:

- **After the final output layer** — you'd be randomly zeroing your predictions
- **Before batch normalization** — the interaction between the two is problematic; BatchNorm's running statistics get distorted by dropout noise during training

---

## Spatial Dropout (Dropout2D)

Regular dropout zeros individual values. For convolutional feature maps, this is weak because adjacent values are highly correlated — a zeroed pixel can be reconstructed from its neighbors.

**Spatial dropout** zeros entire channels:

```
Regular dropout on a feature map:         Spatial dropout:
0.2  0.0  0.8  0.1                        0.0  0.0  0.0  0.0
0.0  0.3  0.0  0.5     vs                 0.0  0.0  0.0  0.0
0.1  0.4  0.0  0.3                        0.0  0.0  0.0  0.0
```

In PyTorch: `nn.Dropout2d(0.5)` — drops entire channels
In Keras: `layers.SpatialDropout2D(0.5)`

This is much more effective for conv layers because it forces the model not to rely on any single feature map.

---

## DropConnect

A variant where instead of zeroing activations, you zero random **weights**:

```
Dropout:     zero random outputs of the previous layer
DropConnect: zero random weights in the weight matrix
```

DropConnect is more fine-grained (each weight is independently dropped) but more expensive to compute. Rarely used in practice.

---

## Dropout and Batch Size

With small batch sizes and high dropout, training becomes very noisy — each sample in the batch sees a different mask, and with few samples the gradient estimate is unreliable.

With large batch sizes, the noise averages out more, and you may need to increase the dropout rate to get the same regularization effect.

---

## Alternatives to Dropout

**L2 Regularization (Weight Decay):**
Penalizes large weights directly. Often used together with dropout.

**Batch Normalization:**
Also acts as a regularizer through mini-batch noise. Some architectures use BatchNorm everywhere and skip dropout entirely.

**Data Augmentation:**
Adds noise at the input level (flips, rotations, crops). Addresses the same problem from a different angle.

**Early Stopping:**
Stop training when validation loss starts increasing. The simplest regularizer.

**DropBlock (Ghiasi et al., 2018):**
Drops contiguous rectangular regions of feature maps. More effective than regular or spatial dropout for conv layers because it forces the network to look at wider context.

---

## Tracing Through Your MNIST Model

```
Flatten:        (batch, 1600)

                Training:
                  1600 values → ~800 randomly zeroed, ~800 surviving × 2
                  Different mask every forward pass
                  Forces the Dense layer to not rely on any specific feature

Dropout(0.5):  (batch, 1600)    same shape, different values

                Inference:
                  1600 values → all pass through unchanged
                  Deterministic output

Dense(10):     (batch, 10)
```

Without this dropout, the model would have 1600 inputs feeding into 10 outputs with no regularization. On a dataset as simple as MNIST, the model could memorize training examples. Dropout forces it to generalize.

---

## Key Takeaways

1. Dropout randomly zeros values during training, does nothing during inference
2. Surviving values are scaled by 1/(1-rate) so expected values match between train and test
3. Zero parameters — it's a stochastic mask, not a learned operation
4. It works by preventing co-adaptation and approximating an ensemble of sub-networks
5. Forgetting `model.eval()` in PyTorch is a classic bug — predictions become noisy
6. 0.5 for dense layers, 0.1-0.3 for conv layers and transformers
7. Gradients flow only through non-dropped neurons, same mask as the forward pass
8. Spatial Dropout (drop entire channels) is more effective for convolutional layers
