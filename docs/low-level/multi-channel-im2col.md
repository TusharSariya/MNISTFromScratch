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

## Visual: What Is a Multi-Channel Patch?

Say we have 3 channels (not 32, for readability) and a 5×5 spatial size. The input is `(3, 5, 5)`:

```
Channel 0:          Channel 1:          Channel 2:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ a b c d e   │     │ A B C D E   │     │ α β γ δ ε   │
│ f g h i j   │     │ F G H I J   │     │ ζ η θ ι κ   │
│ k l m n o   │     │ K L M N O   │     │ λ μ ν ξ ο   │
│ p q r s t   │     │ P Q R S T   │     │ π ρ σ τ υ   │
│ u v w x y   │     │ U V W X Y   │     │ φ χ ψ ω ∅   │
└─────────────┘     └─────────────┘     └─────────────┘
```

The patch at position (0, 0) grabs the 3×3 window from ALL channels:

```
From ch0:  a b c    From ch1:  A B C    From ch2:  α β γ
           f g h               F G H               ζ η θ
           k l m               K L M               λ μ ν
```

Flattened into one row: `[a b c f g h k l m  A B C F G H K L M  α β γ ζ η θ λ μ ν]`
                         |---- ch0: 9 ----|  |---- ch1: 9 ----|  |---- ch2: 9 ----|
                         |---------------- 27 values (3 × 3 × 3) ----------------|

The patch at position (0, 1) slides one column right in ALL channels:

```
From ch0:  b c d    From ch1:  B C D    From ch2:  β γ δ
           g h i               G H I               η θ ι
           l m n               L M N               μ ν ξ
```

Flattened: `[b c d g h i l m n  B C D G H I L M N  β γ δ η θ ι μ ν ξ]`

---

## Visual: The Full im2col Matrix

For input `(3, 5, 5)` with 3×3 kernel, output is 3×3 spatial = 9 positions.
Each position produces a row of `3 channels × 9 = 27` values.

```
                    ├── ch0 ──┤├── ch1 ──┤├── ch2 ──┤

pos (0,0) →  [ a b c f g h k l m  A B C F G H K L M  α β γ ζ η θ λ μ ν ]
pos (0,1) →  [ b c d g h i l m n  B C D G H I L M N  β γ δ η θ ι μ ν ξ ]
pos (0,2) →  [ c d e h i j m n o  C D E H I J M N O  γ δ ε θ ι κ ν ξ ο ]
pos (1,0) →  [ f g h k l m p q r  F G H K L M P Q R  ζ η θ λ μ ν π ρ σ ]
pos (1,1) →  [ g h i l m n q r s  G H I L M N Q R S  η θ ι μ ν ξ ρ σ τ ]
pos (1,2) →  [ h i j m n o r s t  H I J M N O R S T  θ ι κ ν ξ ο σ τ υ ]
pos (2,0) →  [ k l m p q r u v w  K L M P Q R U V W  λ μ ν π ρ σ φ χ ψ ]
pos (2,1) →  [ l m n q r s v w x  L M N Q R S V W X  μ ν ξ ρ σ τ χ ψ ω ]
pos (2,2) →  [ m n o r s t w x y  M N O R S T W X Y  ν ξ ο σ τ υ ψ ω ∅ ]

              patches matrix: (9, 27)
```

Notice how values repeat across rows — position (0,0) and (0,1) share `b c g h l m` from channel 0 alone. This overlap is the whole reason im2col trades memory for speed.

---

## Visual: The Kernel Matrix

Say we have 2 filters (not 64), each looking at all 3 channels with a 3×3 kernel.
Kernel shape: `(2, 3, 3, 3)` — (C_out, C_in, K_h, K_w).

```
Filter 0 weights:                    Filter 1 weights:
  ch0: w00 w01 w02                     ch0: v00 v01 v02
       w03 w04 w05                          v03 v04 v05
       w06 w07 w08                          v06 v07 v08
  ch1: w09 w10 w11                     ch1: v09 v10 v11
       w12 w13 w14                          v12 v13 v14
       w15 w16 w17                          v15 v16 v17
  ch2: w18 w19 w20                     ch2: v18 v19 v20
       w21 w22 w23                          v21 v22 v23
       w24 w25 w26                          v24 v25 v26
```

Reshape `(2, 3, 3, 3)` → `(2, 27)` then `.T` → `(27, 2)`:

```
         filter0  filter1
         ┌──────┬──────┐
ch0 ──── │ w00  │ v00  │
         │ w01  │ v01  │
         │ ...  │ ...  │
         │ w08  │ v08  │
ch1 ──── │ w09  │ v09  │
         │ ...  │ ...  │
         │ w17  │ v17  │
ch2 ──── │ w18  │ v18  │
         │ ...  │ ...  │
         │ w26  │ v26  │
         └──────┴──────┘
         kernel matrix: (27, 2)
```

Each column is one filter's full weight vector across all channels + spatial positions.

---

## Visual: The Matmul

```
patches          @    kernels       =    output
(9, 27)               (27, 2)            (9, 2)

┌──────────────┐     ┌─────────┐       ┌─────────┐
│ patch (0,0)  │     │ f0 │ f1 │       │ o0 │ o1 │  ← output for position (0,0)
│ patch (0,1)  │  @  │    │    │   =   │ o0 │ o1 │  ← output for position (0,1)
│ patch (0,2)  │     │ 27 │ 27 │       │    │    │
│ ...          │     │rows│rows│       │ ...│... │
│ patch (2,2)  │     │    │    │       │ o0 │ o1 │  ← output for position (2,2)
└──────────────┘     └─────────┘       └─────────┘
  9 rows of 27                           9 rows of 2

.T → (2, 9)
.reshape → (2, 3, 3)    ← 2 output feature maps, each 3×3
```

Each element in the output is a dot product: one patch row (27 values) · one kernel column (27 weights). That dot product is exactly what the multi-channel convolution computes at that position for that filter.

---

## The Shapes (Real Numbers)

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

## Building the Patches Matrix with as_strided

### Memory layout of a (C, H, W) array

For input `(32, 13, 13)` stored in C-contiguous (row-major) memory:

```
memory: [ch0_row0_col0, ch0_row0_col1, ..., ch0_row12_col12, ch1_row0_col0, ...]
         |---------- channel 0: 169 floats ----------|  |--- channel 1 ---|
```

The strides (in elements, not bytes) are:
```
s0 = 13 × 13 = 169    — jump to same (row, col) in next channel
s1 = 13                — jump to same col in next row
s2 = 1                 — jump to next column
```

(numpy gives strides in bytes — multiply by element size. For float32: s0=676, s1=52, s2=4)

### The 5D view: (H_out, W_out, C_in, K_h, K_w)

We want `view[y][x][c][dy][dx]` to point at `input[c][y+dy][x+dx]`.

The address of that element in memory is:
```
base + c * s0 + (y + dy) * s1 + (x + dx) * s2
     = base + y * s1 + x * s2 + c * s0 + dy * s1 + dx * s2
```

Reading off the coefficient of each index gives the strides:
```
dim 0 — y  (slide window down):     s1    move one row within a channel
dim 1 — x  (slide window right):    s2    move one col within a channel
dim 2 — c  (next channel):          s0    jump to next channel
dim 3 — dy (down within 3×3):       s1    move one row (same as dim 0)
dim 4 — dx (right within 3×3):      s2    move one col (same as dim 1)
```

```python
s = input.strides   # (s0, s1, s2) in bytes
view = np.lib.stride_tricks.as_strided(
    input,
    shape   = (H_out, W_out, C_in, K, K),     # (11, 11, 32, 3, 3)
    strides = (s[1],  s[2],  s[0], s[1], s[2]) # spatial, channel, patch
)
```

Then reshape the 5D view to 2D:
```python
patches = view.reshape(H_out * W_out, C_in * K * K)   # (121, 288)
```

### Contrast with single-channel

```
2D input (H, W):  strides = (s0, s1)
  as_strided shape:   (H_out, W_out, K, K)          — 4D, no channel dim
  as_strided strides: (s[0],  s[1],  s[0], s[1])    — s[0] is row stride

3D input (C, H, W):  strides = (s0, s1, s2)
  as_strided shape:   (H_out, W_out, C, K, K)       — 5D, channel in the middle
  as_strided strides: (s[1],  s[2],  s[0], s[1], s[2])  — s[0] is now CHANNEL stride
```

The indices shift by 1 because a new dimension (channels) got prepended to the input.

### Alternative: Per-channel as_strided + concatenate

Do single-channel im2col on each of the 32 channels, then stack horizontally:

```
for c in 0..31:
    patches_c = as_strided(input[c], (11, 11, 3, 3), (s1, s2, s1, s2))
    patches_c = patches_c.reshape(121, 9)

patches = np.concatenate([patches_0, patches_1, ..., patches_31], axis=1)
→ shape: (121, 288)
```

This is conceptually simpler but slower (32 separate as_strided calls + a concatenate with copying). The 5D approach does it in one call with no copies.

---

## The Kernels

Conv2 has 64 filters, each with shape `(32, 3, 3)`:

```python
kernels = np.random.randn(64, 32, 3, 3) * 0.1     # (C_out, C_in, K, K)
kernels = kernels.reshape(64, 288).T                # (288, 64)
```

**WRONG** (what full.py currently does):
```python
kernels = kernels.reshape(64, 32, 9).T              # (9, 32, 64) — 3D, can't matmul
```

Each column of the reshaped kernel matrix is one filter's 288 weights, in the same order that the patches were flattened (channel 0's 3×3, then channel 1's 3×3, ..., channel 31's 3×3).

The ordering must match. reshape collapses from the right for both patches and kernels, so `(32, 3, 3)` → `288` puts the values in the same order in both matrices. This is why it works.

---

## Function Definitions

### im2col(input, kernel_size)

Converts a `(C, H, W)` input into a 2D patches matrix for matmul-based convolution.

```
Parameters:
    input       — ndarray, shape (C_in, H, W)
    kernel_size — int, spatial size of the kernel (e.g. 3)

Returns:
    patches     — ndarray, shape (H_out * W_out, C_in * K * K)

Where:
    H_out = H - kernel_size + 1
    W_out = W - kernel_size + 1
    K     = kernel_size

Steps:
    1. Read strides from input: s = input.strides  → (s0, s1, s2)
    2. Compute output dims: H_out = H - K + 1, W_out = W - K + 1
    3. Create 5D view with as_strided:
         shape   = (H_out, W_out, C_in, K, K)
         strides = (s[1], s[2], s[0], s[1], s[2])
    4. Reshape to 2D: (H_out * W_out, C_in * K * K)
    5. Return
```

### prepare_kernels(kernels)

Reshapes kernel tensor into a 2D matrix for matmul.

```
Parameters:
    kernels — ndarray, shape (C_out, C_in, K, K)

Returns:
    kernel_matrix — ndarray, shape (C_in * K * K, C_out)

Steps:
    1. Reshape: (C_out, C_in * K * K)     — flatten each filter into a row
    2. Transpose: (C_in * K * K, C_out)   — each column is now one filter
    3. Return
```

### conv2d_forward(input, kernel_matrix)

Performs one convolution using im2col + matmul. Works for any number of input channels.

```
Parameters:
    input         — ndarray, shape (C_in, H, W)
    kernel_matrix — ndarray, shape (C_in * K * K, C_out), from prepare_kernels

Returns:
    output — ndarray, shape (C_out, H_out, W_out)

Steps:
    1. K = infer kernel size from dimensions:
         K = sqrt(kernel_matrix.shape[0] / C_in)     (integer)
    2. patches = im2col(input, K)                      → (H_out * W_out, C_in * K * K)
    3. result = patches @ kernel_matrix                → (H_out * W_out, C_out)
    4. Transpose: (C_out, H_out * W_out)
    5. Reshape: (C_out, H_out, W_out)
    6. Return
```

**Note:** The first conv's input `(28, 28)` needs a channel dim added so it becomes `(1, 28, 28)` before calling this function. Then both convs go through the exact same code path.

---

## Full Pipeline Shape Trace

```
input:          (1, 28, 28)                        ← add channel dim to raw image

im2col:         (676, 1*9) = (676, 9)
conv1 matmul:   (676, 9) @ (9, 32) → (676, 32)
reshape:        (32, 26, 26)

relu:           (32, 26, 26)
maxpool:        (32, 13, 13)

im2col:         (121, 32*9) = (121, 288)
conv2 matmul:   (121, 288) @ (288, 64) → (121, 64)
reshape:        (64, 11, 11)

relu:           (64, 11, 11)
maxpool:        (64, 5, 5)

flatten:        (1600,)
dense:          (1600,) @ (1600, 10) → (10,)
softmax:        (10,)                  — probabilities for digits 0-9
```
