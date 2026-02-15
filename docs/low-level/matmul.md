# Matrix Multiplication: A Complete Breakdown

## What It Actually Does

Matrix multiplication takes two 2D grids of numbers and produces a new 2D grid. Each element in the output is a **dot product** — multiply corresponding elements from a row of the left matrix and a column of the right matrix, then sum them all up.

That's it. Everything else is details.

---

## The Dot Product (The Core Operation)

Before matrices, understand the dot product of two vectors:

```
a = [1, 2, 3]
b = [4, 5, 6]

dot(a, b) = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
```

Multiply element-wise, then sum. Two vectors go in, one number comes out.

The dot product measures **how similar** two vectors are. If they point the same direction, the result is large and positive. If they're perpendicular, the result is zero. If they point opposite directions, the result is large and negative.

This is why it's useful in neural networks — the dot product of an input patch with a learned kernel produces a large value when the patch matches the pattern the kernel is looking for.

---

## The Shape Rule

```
(m, n) @ (n, p) → (m, p)
```

The inner dimensions must match. Left matrix has n columns, right matrix has n rows. The output has m rows and p columns.

If the inner dimensions don't match, it's an error — the dot products can't be computed because the vectors have different lengths.

Examples:

```
(3, 4) @ (4, 5) → (3, 5)     ✓  inner dimensions match (4)
(676, 9) @ (9, 32) → (676, 32) ✓  inner dimensions match (9)
(3, 4) @ (5, 6) → error        ✗  inner dimensions don't match (4 ≠ 5)
```

---

## Step by Step

Given:

```
A = [[1, 2, 3],       shape: (2, 3)
     [4, 5, 6]]

B = [[7,  8],          shape: (3, 2)
     [9,  10],
     [11, 12]]
```

Output shape: (2, 3) @ (3, 2) → (2, 2)

```
C[0][0] = dot(row 0 of A, col 0 of B)
        = 1*7  + 2*9  + 3*11
        = 7    + 18   + 33
        = 58

C[0][1] = dot(row 0 of A, col 1 of B)
        = 1*8  + 2*10 + 3*12
        = 8    + 20   + 36
        = 64

C[1][0] = dot(row 1 of A, col 0 of B)
        = 4*7  + 5*9  + 6*11
        = 28   + 45   + 66
        = 139

C[1][1] = dot(row 1 of A, col 1 of B)
        = 4*8  + 5*10 + 6*12
        = 32   + 50   + 72
        = 154
```

Result:

```
C = [[58,  64],
     [139, 154]]
```

Each output element = one dot product. Total dot products = m × p = 2 × 2 = 4.

---

## The Pattern

```
C[i][j] = sum(A[i][k] * B[k][j] for k in range(n))
```

Row i of A, dotted with column j of B. That's the entire algorithm. Everything else is implementation detail.

As nested loops in Python:

```
for i in range(m):
    for j in range(p):
        total = 0
        for k in range(n):
            total += A[i][k] * B[k][j]
        C[i][j] = total
```

Three nested loops. O(m * n * p) multiply-and-add operations.

---

## Why It Matters for Convolution (im2col)

In a naive conv2d, you have 6 nested loops:

```
for each image
  for each filter
    for each y position
      for each x position
        for each kernel row
          for each kernel col
            multiply and accumulate
```

im2col rearranges the data so that convolution becomes one matrix multiply:

```
patches: (676, 9)     — 676 positions, each a flattened 3x3 patch
kernels: (9, 32)      — 9 weights per filter, 32 filters

output = patches @ kernels → (676, 32)
```

Each row of the output is one position's response to all 32 filters. Each column is one filter's response at all 676 positions. The 6 nested loops collapse into 676 × 9 × 32 = ~194,000 multiply-adds done in one BLAS call.

---

## Why numpy's @ Is Fast

The naive 3-loop algorithm is O(m * n * p). For (676, 9) @ (9, 32) that's about 194,000 operations. Simple enough in Python — but Python's per-operation overhead makes it ~100-1000x slower than necessary.

numpy's `@` calls BLAS (Basic Linear Algebra Subprograms), typically OpenBLAS or Intel MKL. These libraries use:

### SIMD (Single Instruction, Multiple Data)

Modern CPUs have vector registers that process 4-8 floats simultaneously:

```
Without SIMD:  a[0]*b[0], then a[1]*b[1], then a[2]*b[2], then a[3]*b[3]  — 4 instructions
With SIMD:     [a[0], a[1], a[2], a[3]] * [b[0], b[1], b[2], b[3]]       — 1 instruction
```

For AVX2 (most modern x86 CPUs): 8 float32 operations per instruction. AVX-512: 16 at once.

### Cache Tiling

RAM access is ~100x slower than L1 cache access. BLAS breaks the matrices into small blocks (tiles) that fit in cache:

```
Without tiling:
  For each element in C, read a full row of A and full column of B from RAM.
  Most reads miss the cache because the matrices are too large.

With tiling:
  Load a small block of A and a small block of B into cache.
  Compute all the output elements that only need those blocks.
  Move to the next pair of blocks.
  Most reads hit the cache because the blocks are small.
```

Typical tile sizes: 32x32 to 256x256, tuned for the CPU's cache hierarchy.

### Loop Unrolling

Instead of:

```
for k in range(9):
    total += A[i][k] * B[k][j]
```

The compiler generates:

```
total += A[i][0] * B[0][j]
total += A[i][1] * B[1][j]
total += A[i][2] * B[2][j]
...
total += A[i][8] * B[8][j]
```

No loop counter increment, no branch prediction, no conditional jump. Just straight arithmetic. The compiler (or BLAS author) does this because the inner loop is the hottest code path.

### FMA (Fused Multiply-Add)

Modern CPUs have an instruction that does `a * b + c` in a single cycle instead of two separate operations (multiply then add). BLAS uses this everywhere since matmul is entirely multiply-and-add.

### Combined Effect

For a (676, 9) @ (9, 32) matmul:

```
Python loops:     ~194,000 Python-level operations, each with interpreter overhead
                  Estimated: ~50-200 ms

numpy @ (BLAS):   Same 194,000 arithmetic ops, but:
                  - 8 ops per SIMD instruction → ~24,000 instructions
                  - FMA halves that again → ~12,000 instructions
                  - All data fits in L1 cache (tiny matrices)
                  - No Python interpreter overhead
                  Estimated: <0.1 ms
```

Speedup: roughly 500-2000x for this size. The gap grows with larger matrices.

---

## Transpose

Transposing a matrix swaps rows and columns:

```
A = [[1, 2, 3],       A.T = [[1, 4],
     [4, 5, 6]]              [2, 5],
                              [3, 6]]

shape (2, 3)  →  shape (3, 2)
```

In numpy, `.T` doesn't copy data — it just swaps the strides in the header. Rows become columns and columns become rows by reading the same memory in a different order.

This is useful for matmul setup. If your kernels are stored as `(32, 9)` (32 filters, each with 9 weights) but you need `(9, 32)` for the matmul, `.T` gives you the right shape for free.

---

## In numpy

```
C = A @ B              # matrix multiply (Python 3.5+)
C = np.matmul(A, B)    # same thing, explicit function
C = np.dot(A, B)       # same for 2D arrays, different behavior for higher dims
C = A.dot(B)           # method form of np.dot
```

For 2D arrays these are all identical. `@` is the standard way.

---

## Common Gotchas

### Element-wise multiply is not matmul

```
A * B   — element-wise, shapes must match (or broadcast). Each element multiplied independently.
A @ B   — matrix multiply, inner dimensions must match. Dot products across rows and columns.
```

These are completely different operations. `*` is what you'd use for applying a mask or scaling. `@` is what you'd use for linear transformations, convolution via im2col, or dense layers.

### 1D vectors

For 1D arrays, `@` computes the dot product (a single number):

```
[1, 2, 3] @ [4, 5, 6] = 32
```

numpy treats the left as a row vector and the right as a column vector automatically.

### Batch dimensions

For 3D+ arrays, `@` does a matmul on the last two dimensions and broadcasts over the rest:

```
(batch, m, n) @ (batch, n, p) → (batch, m, p)
```

This is how you'd matmul an entire batch of images at once instead of looping over them.
