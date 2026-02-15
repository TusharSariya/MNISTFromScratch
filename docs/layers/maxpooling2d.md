# MaxPooling2D: A Complete Breakdown

## What It Actually Does

MaxPooling2D slides a window across each feature map and takes the **maximum value** within that window. It has no learnable parameters. Its only job is to downsample the spatial dimensions while keeping the strongest activations.

---

## The Operation

Given a 4x4 feature map and pool_size=2, stride=2 (the defaults):

```
Input (4x4):                Output (2x2):

 1   3   2   1              6   8
 5   6   8   4      →
 2   4   7   3              4   7
 1   0   3   2
```

Step by step:

```
Top-left window:        Top-right window:
1  3                    2  1
5  6                    8  4
max = 6                 max = 8

Bottom-left window:     Bottom-right window:
2  4                    7  3
1  0                    3  2
max = 4                 max = 7
```

The windows don't overlap (stride = pool_size by default). Each 2x2 region collapses to its single largest value.

---

## Input and Output Shape

MaxPooling2D operates on each channel independently. It never mixes channels.

```
Input:  (batch, height, width, channels)
Output: (batch, height/pool, width/pool, channels)
```

In your MNIST model after the first Conv2D:

```
Input:  (batch, 26, 26, 32)
Pool:   2x2, stride 2
Output: (batch, 13, 13, 32)
```

The 32 channels remain 32 channels. Each one is pooled independently. The spatial dimensions halve.

---

## Output Size Calculation

Same formula as Conv2D:

```
output_size = floor((input_size - pool_size + 2 * padding) / stride) + 1
```

With pool_size=2, stride=2, padding=0:

```
output = floor((26 - 2 + 0) / 2) + 1 = 13
```

When the input dimension is odd (like 13):

```
output = floor((13 - 2 + 0) / 2) + 1 = 6
```

The last row/column is dropped. A 13x13 input becomes 6x6, not 7x7. The edge pixels are discarded.

---

## Parameters

Zero. None. MaxPooling has nothing to learn. It's a fixed, deterministic operation. This is one of its advantages — it adds no parameters to your model while reducing computation in subsequent layers by 4x (for 2x2 pooling).

---

## Why Take the Maximum

The maximum value in a region represents the **strongest activation** of the filter that produced it. If a Conv2D filter learned to detect vertical edges, the max in a 2x2 region tells you: "the strongest vertical edge response in this neighborhood."

This means:
- If the feature is present anywhere in the window, it survives
- If it's absent everywhere in the window, the output is low
- The exact position within the window is discarded

That last point is the key property: **translation invariance**.

---

## Translation Invariance

Consider a vertical edge that shifts 1 pixel to the right:

```
Before shift:           After shift:
0  0  8  0              0  0  0  8
0  0  7  0              0  0  0  7
0  0  9  0              0  0  0  9
0  0  6  0              0  0  0  6
```

After 2x2 max pooling:

```
Before:                 After:
  8   0                   8   8      ← different
  9   0                   9   6      ← different
```

The output changed. So max pooling doesn't give perfect translation invariance. But it provides **approximate** invariance — small shifts often produce the same or similar outputs, and deeper stacks of conv + pool layers compound this tolerance.

With a 1-pixel shift in a model with two pooling layers, the shift is now sub-pixel relative to the downsampled feature maps. The deeper you go, the less exact position matters.

---

## What Gets Lost

MaxPooling is a **lossy, irreversible** operation. From a 2x2 window producing the value 8, you cannot recover:
- What the other 3 values were
- Which position within the window held the maximum

This is by design. The information thrown away is the precise spatial location — which is exactly what you want when you care about "is there a 7 in this image" and not "is there a 7 at pixel (14, 12)."

For tasks where spatial precision matters (segmentation, object detection), this loss is a problem. Architectures like U-Net use **skip connections** to pipe the lost spatial information back in. Others avoid pooling entirely.

---

## Stride and Pool Size Don't Have to Match

The default is stride = pool_size, producing non-overlapping windows. But you can set them independently.

**Overlapping pooling** (pool_size=3, stride=2):

```
Input (5x5):

1  3  2  1  4
5  6  8  4  2
2  4  7  3  1
1  0  3  2  5
3  2  1  4  0

Window positions (top row):
[1  3  2]           stride 2 →    [2  1  4]
[5  6  8]                         [8  4  2]
[2  4  7]                         [7  3  1]

max = 8                           max = 8

Output: (2x2)
```

AlexNet (2012) used overlapping pooling and found it reduced overfitting slightly. It's rarely used today.

---

## Alternatives to MaxPooling

### Average Pooling

Takes the mean instead of the max:

```
1  3
5  6   →   max: 6,  average: 3.75
```

Less common in hidden layers (max pooling works better empirically). But **Global Average Pooling** — averaging each entire feature map down to a single number — is standard as the final layer before classification. It replaces the flatten + dense pattern.

### Strided Convolutions

Instead of Conv2D + MaxPool, use Conv2D with stride=2:

```
Traditional:    Conv2D(64, 3, stride=1) → MaxPool(2)    # 0 extra params for pooling
Modern:         Conv2D(64, 3, stride=2)                  # pooling built into conv
```

The strided convolution learns how to downsample rather than using a fixed max operation. The paper "Striving for Simplicity" (Springenberg et al., 2015) showed that all-convolutional networks (no pooling) can match or beat pooled architectures.

ResNet, most modern architectures, and nearly all GANs use strided convolutions instead of pooling.

### No Downsampling At All

Vision Transformers (ViT) split the image into fixed patches and process them as a sequence. There's no progressive spatial downsampling. The "pooling" equivalent is the attention mechanism aggregating information across patches.

---

## MaxPool During Backpropagation

Since only the max value passes through, the gradient flows **only to the position that had the maximum value**. All other positions get zero gradient.

```
Forward:                    Backward (gradient = 1.0 from above):
1  3                        0.0  0.0
5  6   → max = 6            0.0  1.0
```

This is called a **sub-gradient** approach. The max function isn't differentiable at the boundary where two values are equal, but in practice ties are rare and frameworks just pick one (typically the first occurrence).

This means during training:
- The neuron that "won" (had the max value) gets updated
- The neurons that "lost" get no gradient — they don't learn from this example at this location
- Different inputs will have different winners, so all neurons get gradients over the course of training

---

## Max Unpooling

Used in decoder architectures (like in segmentation). During max pooling, you record **which position** had the max (the "indices" or "switches"). During unpooling, you place the value back at the recorded position and fill the rest with zeros:

```
Pooling (record indices):        Unpooling:
1  3                             0  0
5  6   → 6  (index: row 1, col 1)   →   0  6
```

PyTorch exposes this via `nn.MaxPool2d(return_indices=True)` and `nn.MaxUnpool2d`. Keras doesn't have a built-in unpool layer.

---

## Tracing Through Your MNIST Model

```
Conv2D(32, 3):      (batch, 28, 28, 1)  → (batch, 26, 26, 32)
MaxPooling2D(2):    (batch, 26, 26, 32) → (batch, 13, 13, 32)   ← HERE
Conv2D(64, 3):      (batch, 13, 13, 32) → (batch, 11, 11, 64)
MaxPooling2D(2):    (batch, 11, 11, 64) → (batch, 5, 5, 64)     ← AND HERE
Flatten:            (batch, 5, 5, 64)   → (batch, 1600)
```

First pool: 26x26 → 13x13. Each of the 32 feature maps is halved independently.

Second pool: 11x11 → 5x5 (11 is odd, so floor((11-2)/2)+1 = 5, the last row/column is dropped). Each of the 64 feature maps is halved.

Without pooling, the flatten would produce 26 × 26 × 32 = 21,632 values after just the first conv. With both pool layers, it's 5 × 5 × 64 = 1,600. The dense layer after flatten has 1,600 × 10 = 16,000 weights instead of 216,320.

Pooling makes the model **cheaper** (fewer parameters in later layers), **faster** (smaller feature maps to convolve), and more **robust** (less sensitive to exact pixel positions).

---

## Key Takeaways

1. MaxPooling takes the maximum value in each window — no parameters, fixed operation
2. It operates on each channel independently, never mixing them
3. Default pool_size=2 with stride=2 halves the spatial dimensions and discards 75% of values
4. The purpose is downsampling + approximate translation invariance
5. Information is irreversibly lost — you can't reconstruct the input from the output
6. Gradients flow only to the max position; losing positions get zero gradient
7. Modern architectures increasingly replace it with strided convolutions
8. Despite being simple and "dumb," it works remarkably well for classification tasks
