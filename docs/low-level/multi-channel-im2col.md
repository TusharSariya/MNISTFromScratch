# Multi-Channel im2col

## The Problem

Single-channel im2col (conv1) works on a 2D input — one grayscale image. But conv2's input is the output of conv1+relu+pool: a 3D tensor with multiple channels. Each filter in conv2 must look at ALL input channels at every spatial position.

---

## Single-Channel Recap

```
input:   (28, 28)       — 1 channel
kernel:  (1, 3, 3)      — 1 filter looking at 1 channel
patch:   3 × 3 = 9 values

im2col:
  patches: (676, 9)     — 26×26 positions, 9 values each
  kernel:  (9, 32)      — 32 filters, 9 weights each
  output:  (676, 32)    — one matmul does all filters at all positions
```

---

## Multi-Channel: What Changes

After conv1 → relu → pool, the input to conv2 is `(32, 13, 13)` — 32 feature maps, each 13×13.

Each of conv2's 64 filters needs to convolve a 3×3 window across ALL 32 channels at each position. So one filter's weights are `(32, 3, 3)` = 288 values, not 9.

```
input:   (32, 13, 13)   — 32 channels, 13×13 spatial
kernel:  (32, 3, 3)     — one filter sees all 32 channels
patch:   32 × 3 × 3 = 288 values
```

The patch at position (y, x) is: take the 3×3 window at (y, x) from EVERY channel and concatenate them into one flat vector.

---

## The Shapes

```
input:    (32, 13, 13)    — 32 channels
output:   (11, 11)        — 13 - 3 + 1 = 11 positions per axis

patches:  (121, 288)      — 11×11 = 121 positions, 32×3×3 = 288 values per patch
kernels:  (288, 64)       — 64 filters, each with 288 weights
output:   (121, 64)       — one matmul, done

reshape → (64, 11, 11)
```

The matmul structure is identical to single-channel. The only difference is the patch width: 288 instead of 9.

---

## Building the Patches Matrix

### Option 1: as_strided on the 3D array directly

The input is `(32, 13, 13)` with strides `(s0, s1, s2)`:

```
s0 = bytes to move one channel        (13 × 13 × 4 = 676 bytes)
s1 = bytes to move one row             (13 × 4 = 52 bytes)
s2 = bytes to move one column          (4 bytes)
```

You want a 5D view: `(out_h, out_w, channels, kernel_h, kernel_w)`

```
shape:   (11, 11, 32, 3, 3)
strides: (s1, s2, s0, s1, s2)
```

What each stride means:

```
dim 0 (slide window down):    s1    — same as moving one row in a channel
dim 1 (slide window right):   s2    — same as moving one column in a channel
dim 2 (next channel):         s0    — jump to the same (y,x) position in the next channel
dim 3 (down within patch):    s1    — one row within the 3×3 window
dim 4 (right within patch):   s2    — one column within the 3×3 window
```

Then reshape `(11, 11, 32, 3, 3)` → `(121, 288)`:

- The first two dims (11, 11) collapse to 121 rows (one per spatial position)
- The last three dims (32, 3, 3) collapse to 288 columns (one full multi-channel patch)

### Option 2: Per-channel as_strided + concatenate

Do single-channel im2col on each of the 32 channels, then stack horizontally:

```
for c in 0..31:
    patches_c = as_strided(input[c], (11, 11, 3, 3), (s1, s2, s1, s2))
    patches_c = patches_c.reshape(121, 9)

patches = np.concatenate([patches_0, patches_1, ..., patches_31], axis=1)
→ shape: (121, 288)
```

This is conceptually simpler but slower (32 separate as_strided calls + a concatenate with copying). Option 1 does it in one call.

---

## The Kernels

Conv2 has 64 filters, each with shape `(32, 3, 3)`:

```
kernels = np.random.randn(64, 32, 3, 3) * 0.1     — 64 filters, 32 channels, 3×3
kernels = kernels.reshape(64, 288).T                — (288, 64)
```

Each column of the reshaped kernel matrix is one filter's 288 weights, in the same order that the patches were flattened (channel 0's 3×3, then channel 1's 3×3, ..., channel 31's 3×3).

The ordering must match. reshape collapses from the right for both patches and kernels, so `(32, 3, 3)` → `288` puts the values in the same order in both matrices. This is why it works.

---

## The Matmul

```
output = (patches @ kernels).T.reshape(64, 11, 11)

(121, 288) @ (288, 64) → (121, 64)
.T → (64, 121)
.reshape → (64, 11, 11)
```

Identical to single-channel. The matmul doesn't care whether the 288 came from 1 channel of 17×17 patches or 32 channels of 3×3 patches. It just multiplies rows by columns.

---

## Why the Stride Order Is (s1, s2, s0, s1, s2)

This is the non-obvious part. The spatial dimensions (slide the window) use `s1` and `s2`, which are the row and column strides within a single channel. The channel dimension uses `s0`, which jumps to the next channel.

The key insight: you want `result[y][x][c]` to land on position (y, x) in channel c. That address is:

```
base + y * s1 + x * s2 + c * s0
```

Then the last two dimensions (patch height, patch width) walk the 3×3 window from that starting point using `s1` and `s2` again:

```
base + y * s1 + x * s2 + c * s0 + dy * s1 + dx * s2
```

Which is the address of pixel (y + dy, x + dx) in channel c. Exactly what you want.

---

## Generalizing conv2d_forward

The function should handle both cases:

```
Single-channel input (28, 28):
  as_strided shape:   (out_h, out_w, kernel_h, kernel_w)
  as_strided strides: (s0, s1, s0, s1)
  patch width:        kernel_h × kernel_w

Multi-channel input (C, H, W):
  as_strided shape:   (out_h, out_w, C, kernel_h, kernel_w)
  as_strided strides: (s1, s2, s0, s1, s2)
  patch width:        C × kernel_h × kernel_w
```

The difference: multi-channel adds a channel dimension in the middle of the shape, and the spatial strides shift from `(s[0], s[1])` to `(s[1], s[2])` because the first stride `s[0]` now belongs to the channel axis.

You can detect which case you're in by checking `len(input.shape)` — 2 means single-channel, 3 means multi-channel.

---

## Full Pipeline Shape Trace

```
input:          (28, 28)

conv1 im2col:   (676, 9) @ (9, 32) → (676, 32) → (32, 26, 26)
relu:           (32, 26, 26)
maxpool:        (32, 13, 13)

conv2 im2col:   (121, 288) @ (288, 64) → (121, 64) → (64, 11, 11)
relu:           (64, 11, 11)
maxpool:        (64, 5, 5)

flatten:        (1600,)
dense:          (1600,) @ (1600, 10) → (10,)
softmax:        (10,)                  — probabilities for digits 0-9
```
