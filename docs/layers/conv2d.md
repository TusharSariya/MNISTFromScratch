# Conv2D: A Complete Breakdown

## What It Actually Does

A Conv2D layer slides a small matrix (called a **kernel** or **filter**) across a 2D input (like an image), computing element-wise multiplications and summing the results at each position. The output is a new 2D grid called a **feature map** that represents where and how strongly a particular pattern was detected.

That's it. Everything else is details.

---

## The Input

A Conv2D layer expects a 4D tensor:

```
(batch_size, channels, height, width)      # PyTorch (NCHW)
(batch_size, height, width, channels)      # Keras/TensorFlow (NHWC)
```

For a batch of 32 grayscale 28x28 MNIST images:

```
PyTorch:    (32, 1, 28, 28)
Keras:      (32, 28, 28, 1)
```

For a batch of 32 RGB 224x224 images:

```
PyTorch:    (32, 3, 224, 224)
Keras:      (32, 224, 224, 3)
```

The **channels** dimension is critical. A grayscale image has 1 channel. An RGB image has 3. The output of a previous Conv2D layer has as many channels as it had filters.

---

## The Kernel (Filter)

A kernel is a small learnable weight matrix. In your MNIST model, `Conv2D(32, 3)` means 32 filters, each of size 3x3.

But a kernel is not just 2D. Its full shape is:

```
(kernel_height, kernel_width, in_channels)
```

For the first layer of your model (input has 1 channel, kernel size 3):

```
One kernel shape: (3, 3, 1)
```

For a layer receiving 32-channel input with 3x3 kernels:

```
One kernel shape: (3, 3, 32)
```

The kernel always extends through the **full depth** of the input. It's a 3D volume, not a 2D square. This is a common misconception.

Since you have multiple filters (say 32), the full weight tensor is:

```
(out_channels, in_channels, kernel_height, kernel_width)    # PyTorch
(kernel_height, kernel_width, in_channels, out_channels)    # Keras
```

For `Conv2D(32, 3)` on a 1-channel input:

```
Weight shape: (32, 1, 3, 3)  →  32 filters × 1 channel × 3 × 3 = 288 parameters
Bias shape:   (32,)           →  one bias per filter = 32 parameters
Total:        320 parameters
```

---

## The Convolution Operation

### Step by step for a single filter on a single-channel input

Given a 5x5 input and a 3x3 kernel:

```
Input:                  Kernel:
1  2  3  0  1           1  0  1
0  1  2  3  0           0  1  0
1  0  1  2  1           1  0  1
2  1  0  1  0
0  1  2  0  1
```

Position (0,0) — place the kernel over the top-left 3x3 region:

```
1  2  3       1  0  1
0  1  2   ×   0  1  0    (element-wise multiply, then sum)
1  0  1       1  0  1

= (1×1) + (2×0) + (3×1) + (0×0) + (1×1) + (2×0) + (1×1) + (0×0) + (1×1)
= 1 + 0 + 3 + 0 + 1 + 0 + 1 + 0 + 1
= 7
```

Then slide right by 1 (stride=1) and repeat. Then next row. The result:

```
Output (3x3):
7  ...  ...
...  ...  ...
...  ...  ...
```

Each position in the output is a **dot product** between the kernel and the local region of the input it overlaps.

### Multi-channel input

If the input has C channels (e.g., 3 for RGB), the kernel is (3, 3, C). You perform the convolution independently on each channel, then **sum across channels** to produce one number per spatial position.

```
output[i,j] = sum over c of (input[c] ⊛ kernel[c])[i,j] + bias
```

Where ⊛ is the 2D cross-correlation for one channel.

### Multiple filters

Each filter produces one feature map. If you have 32 filters, you get 32 feature maps stacked as 32 output channels.

```
Input:  (1, 28, 28)   →   32 filters of shape (1, 3, 3)   →   Output: (32, 26, 26)
```

---

## Output Size Calculation

```
output_size = floor((input_size - kernel_size + 2 * padding) / stride) + 1
```

For your model's first conv layer:

```
input:  28x28
kernel: 3x3
padding: 0
stride: 1

output = floor((28 - 3 + 0) / 1) + 1 = 26
```

So a 28x28 image becomes 26x26. You lose 1 pixel on each side.

### Examples with different parameters

| Input | Kernel | Padding | Stride | Output |
|-------|--------|---------|--------|--------|
| 28    | 3      | 0       | 1      | 26     |
| 28    | 3      | 1       | 1      | 28     |
| 28    | 5      | 0       | 1      | 24     |
| 28    | 3      | 0       | 2      | 13     |
| 224   | 7      | 3       | 2      | 112    |

---

## Padding

Padding adds zeros (or other values) around the border of the input before convolving.

```
Without padding (valid):     With padding=1 (same for 3x3):

  x x x x x                 0 0 0 0 0 0 0
  x x x x x                 0 x x x x x 0
  x x x x x                 0 x x x x x 0
  x x x x x                 0 x x x x x 0
  x x x x x                 0 x x x x x 0
                             0 x x x x x 0
                             0 0 0 0 0 0 0
```

**Why padding matters:**
- Without it, every layer shrinks the spatial dimensions. After several layers, your feature maps become tiny.
- Edge pixels participate in fewer convolutions than center pixels, so they contribute less. Padding fixes this asymmetry.
- `padding = (kernel_size - 1) / 2` with stride 1 preserves spatial dimensions ("same" padding).

---

## Stride

Stride controls how far the kernel moves between positions.

```
Stride 1:                    Stride 2:
[x x x] . .                 [x x x] . .
. [x x x] .                 . . [x x x]
. . [x x x]                 . . . . .
. . . . .                   [x x x] . .
. . . . .                   . . [x x x]
```

Stride 1 hits every position. Stride 2 skips every other position, halving the output size. This is an alternative to pooling for downsampling.

---

## What the Filters Learn

This is the key insight. Nobody hand-designs these filters. They are learned through backpropagation.

After training, filters in different layers tend to learn:

**Layer 1 (close to input):**
- Edge detectors (horizontal, vertical, diagonal)
- Gradient detectors
- Color blob detectors

These look like Gabor filters from classical computer vision. The network rediscovers them from data.

**Layer 2-3 (middle):**
- Corners, textures, small shapes
- Combinations of edges from layer 1

**Deeper layers:**
- Object parts (eyes, wheels, windows)
- Complex textures and patterns

This hierarchy emerges automatically. Each layer composes patterns from the previous layer's detections.

---

## Receptive Field

The **receptive field** is the region of the original input that influences a single output pixel.

For one 3x3 conv layer: receptive field = 3x3.

For two stacked 3x3 conv layers: receptive field = 5x5.

For three stacked 3x3 conv layers: receptive field = 7x7.

This is why VGGNet uses stacked 3x3 convolutions instead of larger filters:
- Two 3x3 layers have the same receptive field as one 5x5 layer
- But with fewer parameters: 2 × (3×3) = 18 vs 5×5 = 25
- And two non-linearities (ReLU) instead of one, making the function more expressive

With pooling or strided convolutions in between, the receptive field grows even faster because each pixel in the downsampled map covers a larger input region.

---

## Parameter Count vs Fully Connected

This is why convolutions exist. Consider a 224x224x3 input:

**Fully connected layer with 64 outputs:**
```
224 × 224 × 3 × 64 = 9,633,792 parameters
```

**Conv2D with 64 filters of 3x3:**
```
3 × 3 × 3 × 64 + 64 (bias) = 1,792 parameters
```

That's a ~5000x reduction. This comes from two properties:

1. **Parameter sharing** — the same 3x3 filter is applied at every spatial position. A vertical edge looks the same whether it's in the top-left or bottom-right.

2. **Local connectivity** — each output pixel depends only on a small local region, not the entire input. Pixels far apart rarely need to interact directly (at least in early layers).

---

## The Bias Term

Each filter has one scalar bias added to every position in its output feature map:

```
output[i,j] = (sum of element-wise products) + bias
```

In your model, `Conv2D(32, 3)` has 32 bias values, one per filter. The bias shifts the activation, affecting the threshold at which the filter "fires."

Some architectures omit the bias when followed by Batch Normalization, since BatchNorm has its own learnable shift parameter, making the conv bias redundant.

---

## Dilation

Dilated (atrous) convolutions insert gaps between kernel elements:

```
Dilation 1 (normal):     Dilation 2:
x x x                    x . x . x
x x x                    . . . . .
x x x                    x . x . x
                          . . . . .
                          x . x . x
```

A 3x3 kernel with dilation 2 has the same number of parameters as a normal 3x3 kernel, but covers a 5x5 receptive field. This is used in semantic segmentation (DeepLab) to capture large context without increasing computation.

---

## Groups

Grouped convolutions split the input channels into groups and convolve each group independently.

```
Normal conv:  all input channels → all output channels
Groups=2:     first half of input channels → first half of output channels
              second half of input channels → second half of output channels
```

**Depthwise convolution** is the extreme case where `groups = in_channels`. Each input channel gets its own separate filter. MobileNet uses this followed by a 1x1 "pointwise" convolution to mix channels, dramatically reducing computation.

Parameter comparison for 32 input channels, 64 output channels, 3x3 kernel:
```
Normal:              32 × 64 × 3 × 3 = 18,432
Depthwise + 1x1:     32 × 3 × 3 + 32 × 64 = 288 + 2,048 = 2,336
```

---

## What Convolution Really Is (Mathematically)

Technically, what deep learning calls "convolution" is actually **cross-correlation**. True convolution flips the kernel before sliding:

```
True convolution:       (f * g)(t) = ∫ f(τ) g(t - τ) dτ
Cross-correlation:      (f ⋆ g)(t) = ∫ f(τ) g(t + τ) dτ
```

The difference is that convolution flips the kernel (g(t - τ) vs g(t + τ)). Since the kernel weights are learned, it doesn't matter — the network will just learn the flipped version. Every deep learning framework implements cross-correlation and calls it convolution.

---

## Tracing Through Your MNIST Model

```python
layers.Input(shape=(28, 28, 1))           # (batch, 28, 28, 1)
layers.Conv2D(32, kernel_size=3, ...)     # (batch, 26, 26, 32)  — 320 params
layers.MaxPooling2D(pool_size=2)          # (batch, 13, 13, 32)  — 0 params
layers.Conv2D(64, kernel_size=3, ...)     # (batch, 11, 11, 64)  — 18,496 params
layers.MaxPooling2D(pool_size=2)          # (batch, 5, 5, 64)    — 0 params
layers.Flatten()                          # (batch, 1600)
layers.Dropout(0.5)                       # (batch, 1600)
layers.Dense(10, ...)                     # (batch, 10)          — 16,010 params
```

Breaking down the second Conv2D parameter count:
```
kernel:  3 × 3 × 32 channels × 64 filters = 18,432
bias:    64
total:   18,496
```

The 32 input channels come from the 32 filters of the first Conv2D. Each of the 64 new filters is a (3, 3, 32) volume that gets convolved across the (13, 13, 32) input to produce one (11, 11) feature map.

---

## Key Takeaways

1. A Conv2D filter is a 3D volume (height × width × in_channels), not a 2D square
2. Each filter produces one output channel via dot products at every spatial position
3. The weights are shared across all positions — same filter everywhere
4. Stacking small filters gives large receptive fields with fewer parameters
5. The hierarchy of learned features (edges → parts → objects) emerges from training
6. "Convolution" in deep learning is technically cross-correlation — it doesn't matter
7. The output size depends on kernel size, padding, stride, and dilation
