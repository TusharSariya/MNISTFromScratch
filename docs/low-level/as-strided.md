# numpy.lib.stride_tricks.as_strided

## What It Actually Does

Creates a new **view** of an existing array with a custom shape and custom strides. No data is copied. It just tells numpy "read the same bytes in memory, but navigate them differently."

That's it. It's pointer manipulation dressed up as a numpy function.

---

## What a Numpy Array Really Is

A numpy array is not the data — it's a **header** pointing at data:

```
Header:
  - data pointer:  0x7f3a...  (address of first byte)
  - shape:         (28, 28)
  - strides:       (112, 4)
  - dtype:         float32 (4 bytes per element)

Memory (the actual bytes):
  [0.0, 0.0, 0.0, ..., 0.5, 0.9, ...]
```

When you access `array[3][5]`, numpy computes:

```
address = data_pointer + 3 * strides[0] + 5 * strides[1]
        = base + 3 * 112 + 5 * 4
        = base + 336 + 20
        = base + 356
```

Then reads 4 bytes (one float32) from that address.

`as_strided` creates a new header with different shape and strides, pointing at the same memory. The data never moves.

---

## What Strides Are

Strides tell numpy how many **bytes** to jump to move one step along each dimension.

For a `(28, 28)` float32 array:

```
strides = (112, 4)

112 = 28 columns × 4 bytes per float
  → jump 112 bytes to move one row down

4 = 1 float × 4 bytes
  → jump 4 bytes to move one column right
```

For a `(32, 26, 26)` float32 array:

```
strides = (2704, 104, 4)

2704 = 26 × 26 × 4
  → jump 2704 bytes to move to the next channel

104 = 26 × 4
  → jump 104 bytes to move one row down within a channel

4 = 1 × 4
  → jump 4 bytes to move one column right
```

You never have to calculate these by hand. Every numpy array already knows its strides via `.strides`.

---

## The Function

```
numpy.lib.stride_tricks.as_strided(x, shape, strides)
```

- **x**: the source array (the memory you want to view differently)
- **shape**: the shape of the output view
- **strides**: how many bytes to jump along each dimension of the output

The number of dimensions in shape and strides must match. They don't need to match the original array's dimensions — that's the whole point.

---

## Use Case 1: im2col (Conv2D)

### The Problem

You have a `(28, 28)` image and a `(3, 3)` kernel. You need all 26×26 = 676 overlapping 3×3 patches.

### The Solution

```
image.shape = (28, 28)
image.strides = (112, 4)     # s[0]=112, s[1]=4

as_strided(image, shape=(26, 26, 3, 3), strides=(s[0], s[1], s[0], s[1]))
```

Output shape: `(26, 26, 3, 3)` — a 26×26 grid of 3×3 patches.

### Why Those Strides

The output has 4 dimensions. Each needs a stride:

```
dim 0: move to next patch row     → same as moving one row in the image     → s[0] = 112
dim 1: move to next patch column  → same as moving one column in the image  → s[1] = 4
dim 2: move down within a patch   → same as moving one row in the image     → s[0] = 112
dim 3: move right within a patch  → same as moving one column in the image  → s[1] = 4
```

The window slides with stride 1 (one pixel at a time), so the patch strides equal the image strides. If the window slid with stride 2, the first two strides would be doubled.

### Why It Works

`result[y][x]` gives you the 3×3 patch starting at position (y, x) in the original image. Numpy computes the memory address:

```
base + y * 112 + x * 4
```

That's exactly the top-left corner of the patch at (y, x). Then the inner two dimensions walk the 3×3 grid from that starting point using the same row/column strides.

No data is copied. All 676 patches are views into the same 28×28 block of memory. Overlapping patches share bytes.

---

## Use Case 2: MaxPooling2D

### The Problem

You have a `(26, 26)` feature map and want to do 2×2 max pooling with stride 2. Output is `(13, 13)`.

### The Solution

```
channel.shape = (26, 26)
channel.strides = (104, 4)     # s[0]=104, s[1]=4

as_strided(channel, shape=(13, 13, 2, 2), strides=(s[0]*2, s[1]*2, s[0], s[1]))
```

Output shape: `(13, 13, 2, 2)` — a 13×13 grid of 2×2 blocks.

Then `np.max(result, axis=(2, 3))` takes the max within each 2×2 block → `(13, 13)`.

### Why Those Strides

```
dim 0: move to next pool row      → skip 2 rows in the image     → s[0] * 2 = 208
dim 1: move to next pool column   → skip 2 columns in the image  → s[1] * 2 = 8
dim 2: move down within a block   → one row in the image          → s[0] = 104
dim 3: move right within a block  → one column in the image       → s[1] = 4
```

The first two strides are doubled because pool stride is 2 — each pool window starts 2 pixels apart, not 1 like in conv2d.

### Why No Overlap

Unlike conv2d (stride 1, patches overlap), maxpooling with stride 2 produces non-overlapping blocks. Each pixel belongs to exactly one 2×2 block. This means `reshape` after `as_strided` would NOT copy memory, because there's no duplication.

---

## Use Case 3: Transpose

`.T` on a numpy array is just swapping strides:

```
array.shape = (3, 4)
array.strides = (16, 4)

array.T.shape = (4, 3)
array.T.strides = (4, 16)
```

Same memory, same pointer. Just swap which stride goes with which dimension. Rows become columns and columns become rows because numpy now jumps 4 bytes to move "down" and 16 bytes to move "right" — the opposite of before.

---

## The General Pattern

Every use of `as_strided` follows the same logic:

1. Decide what shape you want: `(grid_rows, grid_cols, window_h, window_w)`
2. Figure out the byte jumps for each dimension:
   - Grid dimensions: how far apart are the windows? `image_stride * window_stride`
   - Window dimensions: how do you walk within a window? Same as the image strides.
3. Get the actual byte values from `x.strides` — never hardcode them.

```
s = x.strides
output_strides = (s[0] * stride_y, s[1] * stride_x, s[0], s[1])
```

For stride 1 (conv2d): `(s[0], s[1], s[0], s[1])` — grid strides = image strides
For stride 2 (maxpool): `(s[0]*2, s[1]*2, s[0], s[1])` — grid strides = 2× image strides

---

## Why It's Dangerous

`as_strided` does NO bounds checking. If your shape or strides are wrong, numpy will happily read memory beyond the array's allocation. You won't get an error — you'll get garbage values or a segfault.

```
# This is wrong but numpy won't stop you
as_strided(image, shape=(100, 100, 3, 3), strides=(112, 4, 112, 4))
# image is only 28x28 but you asked for 100x100 patches
# numpy reads past the end of the array into whatever is in memory next
```

Always verify your shape is valid:

```
For conv2d:  output_size = input_size - kernel_size + 1
For maxpool: output_size = input_size // pool_size
```

---

## Why Not Just Use Loops

Performance. `as_strided` is instant — it creates a view in O(1) time regardless of array size. Then a single `reshape` + `@` (matmul) or `np.max` does the actual work in optimized C.

Python loops over the same data would do millions of individual element accesses through the interpreter, each with type checking and object overhead.

The data is the same. The math is the same. The difference is whether you do it in C or in Python.
