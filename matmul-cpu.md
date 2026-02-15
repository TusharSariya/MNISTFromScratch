# Implementing Matrix Multiplication on CPU

A practical guide to writing a matmul in C that gets within 2-5x of BLAS performance, in three stages.

---

## Stage 1: Naive Matmul in C

### What You Need

- A C compiler (gcc or clang)
- `<stdlib.h>` for malloc
- `<time.h>` or `clock_gettime` for benchmarking

### The Function Signature

```
void matmul_naive(float *A, float *B, float *C, int m, int n, int p)
```

A is (m, n), B is (n, p), C is (m, p). All stored as flat 1D arrays in row-major order.

### Row-Major Layout

A 2D matrix stored as a flat array. Element at row i, column j is at index `i * num_cols + j`.

```
Matrix:     [[1, 2, 3],
             [4, 5, 6]]

In memory:  [1, 2, 3, 4, 5, 6]

A[i][j] = A[i * n + j]
```

### The Algorithm

```
for i in 0..m:
    for j in 0..p:
        sum = 0
        for k in 0..n:
            sum += A[i * n + k] * B[k * p + j]
        C[i * p + j] = sum
```

Three nested loops. Straightforward. This is your baseline.

### Why It's Slow

The inner loop accesses `B[k * p + j]` — stepping through B by `p` elements each iteration. In row-major layout, consecutive elements of a **column** are `p` floats apart in memory. Each access likely misses the cache because you're jumping over entire rows.

```
Accessing row of A:     A[i*n+0], A[i*n+1], A[i*n+2], ...  → contiguous, cache-friendly
Accessing column of B:  B[0*p+j], B[1*p+j], B[2*p+j], ...  → strided, cache-hostile
```

### Expected Performance

For (676, 9) @ (9, 32): fast enough, matrices are tiny.
For (1024, 1024) @ (1024, 1024): roughly 10-50x slower than numpy.

### The Loop Order Trick

Before adding SIMD, try changing the loop order to `i, k, j` instead of `i, j, k`:

```
for i in 0..m:
    for k in 0..n:
        a_ik = A[i * n + k]
        for j in 0..p:
            C[i * p + j] += a_ik * B[k * p + j]
```

Now the inner loop walks through B **row-wise** (consecutive j values = consecutive memory addresses) and through C row-wise too. Both are contiguous. This simple reordering can give 2-5x speedup on large matrices because almost every memory access hits the cache.

Why: `B[k * p + j]` with j incrementing by 1 means consecutive addresses. The CPU prefetcher detects the sequential pattern and loads the next cache line before you need it.

---

## Stage 2: SIMD (Single Instruction, Multiple Data)

### What You Need

- `<immintrin.h>` — the header for Intel SIMD intrinsics
- Compile with `-mavx2 -mfma` flags

### What SIMD Does

Normal instruction: multiply two floats, get one result.
SIMD instruction: multiply two groups of 8 floats, get 8 results. Same number of clock cycles.

```
Scalar:   a0*b0 → c0                           (1 multiply per cycle)
AVX2:     [a0,a1,a2,a3,a4,a5,a6,a7]
        * [b0,b1,b2,b3,b4,b5,b6,b7]
        → [c0,c1,c2,c3,c4,c5,c6,c7]            (8 multiplies per cycle)
```

### The Key Data Type

`__m256` — a 256-bit register holding 8 float32 values. This is the thing you load, multiply, and store.

### The Key Intrinsics

You only need a handful:

```
__m256 _mm256_loadu_ps(float *addr)
```
Load 8 consecutive floats from memory into a 256-bit register.

```
__m256 _mm256_set1_ps(float a)
```
Fill all 8 slots of a register with the same value. Useful for broadcasting a single scalar.

```
__m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c)
```
Fused multiply-add: compute `a * b + c` element-wise. Returns 8 results. This is the workhorse — one instruction does 8 multiplies and 8 adds.

```
void _mm256_storeu_ps(float *addr, __m256 a)
```
Store 8 floats from a register back to memory.

### The Algorithm

Use the `i, k, j` loop order from Stage 1. The inner j loop processes 8 columns of B and C at a time:

```
for i in 0..m:
    for k in 0..n:
        a_ik = broadcast A[i * n + k] to all 8 lanes     // _mm256_set1_ps
        for j in 0..p step 8:
            b_vec = load 8 floats from B[k * p + j]       // _mm256_loadu_ps
            c_vec = load 8 floats from C[i * p + j]       // _mm256_loadu_ps
            c_vec = fma(a_ik, b_vec, c_vec)                // _mm256_fmadd_ps
            store c_vec to C[i * p + j]                    // _mm256_storeu_ps
```

Each iteration of the inner loop does 8 multiply-adds in one FMA instruction instead of 8 separate scalar operations.

### Handling Non-Multiples of 8

If p is not a multiple of 8, the last few columns don't fill a full register. Two options:

- Pad B and C to the next multiple of 8 with zeros. Simplest.
- Handle the remainder with a scalar loop after the SIMD loop.

### The Function Signature

```
void matmul_simd(float *A, float *B, float *C, int m, int n, int p)
```

Same interface as naive. The SIMD is an internal optimization.

### Expected Performance

Theoretical max: 8x over scalar (8 floats per instruction).
Realistic: 3-6x over the loop-reordered naive version. You lose some to memory bandwidth and instruction overhead.

---

## Stage 3: Cache Tiling

### The Problem

For large matrices, the data doesn't fit in L1 cache (typically 32-64 KB). Every time you read a row of B, it evicts data you'll need again later. You end up reading from L2 or L3 cache (or worse, main memory) repeatedly.

```
L1 cache:    ~32 KB,   ~1 ns access     (4 cycles)
L2 cache:    ~256 KB,  ~4 ns access     (12 cycles)
L3 cache:    ~8 MB,    ~12 ns access    (36 cycles)
Main memory: GB,       ~60 ns access    (200 cycles)
```

The difference between hitting L1 and going to main memory is ~50x. For matmul, which is memory-bound at large sizes, this dominates performance.

### The Idea

Instead of computing the entire output matrix in one sweep, break A, B, and C into small **tiles** (blocks) that fit in L1 cache. Compute all the partial results for one tile before moving to the next.

```
Without tiling:
  Each element of C reads a full row of A and full column of B.
  B gets evicted from cache and reloaded m times.

With tiling:
  Load a tile of A and a tile of B into cache.
  Compute the partial contributions to a tile of C.
  Everything stays in L1 for the duration of one tile computation.
```

### Choosing Tile Size

The constraint: tiles of A, B, and C must all fit in L1 simultaneously.

For L1 = 32 KB and float32 (4 bytes):

```
32 KB / 4 bytes = 8192 floats total budget

Three tiles: tile_A + tile_B + tile_C ≤ 8192 floats

If tiles are square (T × T):
  3 * T * T ≤ 8192
  T ≤ 52

Practical choice: T = 32 or T = 48
```

You don't need to be exact. Start with T = 32 and experiment.

### The Algorithm

```
for ii in 0..m step T:           // tile rows of A and C
    for jj in 0..p step T:       // tile columns of B and C
        for kk in 0..n step T:   // tile the shared dimension
            // multiply tile of A by tile of B, accumulate into tile of C
            for i in ii..min(ii+T, m):
                for k in kk..min(kk+T, n):
                    a_ik = A[i * n + k]
                    for j in jj..min(jj+T, p):
                        C[i * p + j] += a_ik * B[k * p + j]
```

The three outer loops step through tiles. The three inner loops are the same `i, k, j` matmul but confined to a small block that fits in cache.

### Combining with SIMD

Replace the innermost j loop with the SIMD version from Stage 2. Now you have cache-friendly data access AND 8-wide vector operations:

```
for ii in 0..m step T:
    for jj in 0..p step T:
        for kk in 0..n step T:
            for i in ii..min(ii+T, m):
                for k in kk..min(kk+T, n):
                    a_ik = broadcast A[i * n + k]              // _mm256_set1_ps
                    for j in jj..min(jj+T, p) step 8:
                        b_vec = load 8 from B[k * p + j]      // _mm256_loadu_ps
                        c_vec = load 8 from C[i * p + j]      // _mm256_loadu_ps
                        c_vec = fma(a_ik, b_vec, c_vec)        // _mm256_fmadd_ps
                        store c_vec to C[i * p + j]            // _mm256_storeu_ps
```

### The Function Signature

```
void matmul_tiled(float *A, float *B, float *C, int m, int n, int p, int tile_size)
```

Or hardcode the tile size if you've benchmarked and found the optimum.

### Expected Performance

Tiling + SIMD typically gets within 2-5x of BLAS for large matrices. The remaining gap is the micro-architectural stuff covered in matmul-cpu-fun-stuff.md.

For your MNIST matrices (676, 9) @ (9, 32), tiling won't help because everything already fits in L1. The benefit shows up at sizes like (1024, 1024) and above.

---

## Benchmarking

### How to Measure

Time many iterations and take the average. One matmul is too fast to time accurately:

```
start = clock()
for trial in 0..1000:
    matmul(A, B, C, m, n, p)
end = clock()
time_per_call = (end - start) / 1000
```

### How to Compare

The standard metric is GFLOPS (billion floating-point operations per second):

```
flops = 2 * m * n * p                    // each element: one multiply + one add
gflops = flops / (time_in_seconds * 1e9)
```

For reference, a modern CPU (single core) can do roughly:
- 50-100 GFLOPS with AVX2/FMA on float32
- Your naive C loop will probably hit 1-5 GFLOPS
- BLAS hits 30-80 GFLOPS (single threaded)

### What to Test

Use large square matrices for benchmarking (512x512, 1024x1024, 2048x2048). Small matrices like your MNIST sizes are too fast to show meaningful differences between implementations.

---

## Summary of Expected Speedups

```
Stage 0: Python loops           → baseline (painfully slow)
Stage 1: Naive C                → ~100-1000x over Python
Stage 1b: Loop reorder (i,k,j)  → ~2-5x over naive C
Stage 2: + SIMD (AVX2/FMA)      → ~3-6x over reordered C
Stage 3: + Cache tiling          → ~1.5-3x over SIMD alone (large matrices only)

Total: within 2-5x of BLAS
```
