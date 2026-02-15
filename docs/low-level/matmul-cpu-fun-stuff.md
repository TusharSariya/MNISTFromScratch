# Matmul on CPU: The Last 2-5x (Fun Stuff)

After naive C + SIMD + cache tiling, you're within 2-5x of BLAS. Closing that gap is a different kind of work — it's not about algorithms anymore, it's about understanding the CPU as a physical machine.

---

## Register Tiling (Micro-Kernels)

### The Problem

In Stage 3, the inner loop loads from C, does an FMA, and stores back to C every iteration. Those loads and stores are wasted — the data should just stay in registers.

### The Idea

A CPU has 16 YMM registers (AVX2), each holding 8 floats. That's 128 floats you can keep in-flight without touching memory. A **micro-kernel** computes a small tile of C entirely in registers.

### How It Works

Pick a micro-tile size that fits in registers. A common choice: 6 rows × 16 columns of C.

```
6 rows × 16 cols = 96 floats
16 cols = 2 YMM registers per row
6 rows × 2 registers = 12 registers for C tile
```

That leaves 4 registers for loading A and B values. Total: 16 registers, all used.

The micro-kernel pseudocode:

```
// c00..c05 and c10..c15 are YMM registers holding the C tile
// 12 registers for C, 1 for a broadcast of A, 1 for loading B

for k in 0..K:
    b0 = load 8 floats from B[k, j]           // columns j..j+7
    b1 = load 8 floats from B[k, j+8]         // columns j+8..j+15

    a0 = broadcast A[i+0, k]
    c00 = fma(a0, b0, c00)                     // C[i+0, j..j+7]  += A[i+0,k] * B[k, j..j+7]
    c01 = fma(a0, b1, c01)                     // C[i+0, j+8..j+15]

    a0 = broadcast A[i+1, k]
    c10 = fma(a0, b0, c10)                     // C[i+1, j..j+7]
    c11 = fma(a0, b1, c11)

    a0 = broadcast A[i+2, k]
    c20 = fma(a0, b0, c20)
    c21 = fma(a0, b1, c21)

    ... (repeat for rows 3, 4, 5)
```

Each iteration of k: 2 loads from B, 6 broadcasts from A, 12 FMA instructions. No loads or stores to C until the entire k loop finishes. C lives in registers the whole time.

### Why This Matters

Each FMA does 8 multiplies + 8 adds = 16 FLOP. 12 FMAs per k iteration = 192 FLOP. If FMA throughput is 2 per cycle (typical on modern Intel), that's 6 cycles per k iteration for 192 FLOP = 32 FLOP/cycle. The theoretical peak for a single core with AVX2 is 32 FLOP/cycle (2 FMA units × 8 floats × 2 ops). So this micro-kernel is **at theoretical peak** if the loads don't stall it.

The loads don't stall it because there are enough FMA instructions between consecutive loads that the load latency is hidden. This is called **instruction-level parallelism** — the CPU works on multiple instructions simultaneously, overlapping loads with computation.

### Choosing Micro-Tile Size

The choice of 6×16 isn't arbitrary. It's determined by:

```
Available registers: 16 (AVX2)
Registers for C tile: rows × (cols / 8)
Registers for A and B loads: at least 2-3

Maximize rows × cols such that:
  rows × (cols / 8) + 3 ≤ 16
```

Different shapes trade off:
- **6×16**: 12 registers for C, good balance of row and column reuse
- **4×24**: 12 registers for C, more column reuse, fewer rows
- **8×8**: 8 registers for C, more register headroom, but less work per k step

BLAS libraries benchmark all reasonable shapes on each CPU and pick the best one.

---

## Packing (Data Layout Transformation)

### The Problem

Even with tiling, accessing B in the inner loop may still cause TLB (Translation Lookaside Buffer) misses. The TLB maps virtual addresses to physical addresses, and it's small (~64 entries for L1 TLB). If your tile of B spans many memory pages (4 KB each), the TLB can't hold all the mappings.

### The Idea

Before the matmul, **copy** (pack) each tile of A and B into a contiguous buffer with a layout optimized for the micro-kernel's access pattern.

```
Original B: row-major, tiles are scattered across memory
Packed B:   each tile stored contiguously, columns interleaved for SIMD access
```

For B, packing means rearranging a tile so that the 8 consecutive floats the micro-kernel loads with `_mm256_loadu_ps` are actually contiguous in memory, even if they weren't in the original matrix.

For A, packing means storing the column values that get broadcast in the order the micro-kernel reads them.

### The Cost

Packing copies data — O(m*n + n*p) extra work. But the matmul itself is O(m*n*p), which is much larger. The packing cost is amortized, especially for large matrices.

For small matrices, packing overhead can actually hurt. BLAS libraries have heuristics for when to pack and when not to.

### When It Matters

- Matrix dimensions > ~256: packing helps
- Matrix dimensions < ~64: packing hurts (overhead > benefit)
- Non-contiguous or transposed inputs: packing is essential

---

## Prefetching

### The Problem

Even with tiling and packing, there's latency when the CPU first reads a cache line from L2 or L3. The data isn't in L1 yet, so the load stalls for a few cycles.

### The Idea

Tell the CPU to start loading data you'll need in the future **before you actually need it**. By the time your code gets to that data, it's already in L1.

```
__builtin_prefetch(addr, 0, 3)
```

- `addr`: the memory address you'll need soon
- `0`: read access (vs 1 for write)
- `3`: high temporal locality (keep it in all cache levels)

### Where to Use It

In the micro-kernel, prefetch the next tile of B while computing the current one:

```
for k in 0..K:
    prefetch B tile for k+4              // start loading future data
    ... do FMA work for current k ...    // by the time we reach k+4, it's in cache
```

The distance (4 in this example) depends on the load latency and how much work happens per k step. Too close and the data isn't ready yet. Too far and it gets evicted before use.

### Realistic Impact

Prefetching gives 5-15% improvement on top of tiling + SIMD. It's a fine-tuning knob, not a game changer. But when you're already at 80% of peak, every percent matters.

---

## Multi-Level Tiling

### The Problem

Stage 3 used one level of tiling (for L1). But the CPU has three cache levels, and large matrices spill out of L2 and L3 too.

### The Idea

BLAS uses a **three-level tiling strategy**:

```
Level 1 (L3 cache): Break the problem into panels that fit in L3.
  Level 2 (L2 cache): Break each panel into blocks that fit in L2.
    Level 3 (L1 cache): Break each block into micro-tiles that fit in registers.
      Micro-kernel: the 6×16 register-tiled inner loop.
```

The specific decomposition (which matrix gets tiled at which level) is called the **loop ordering** and it's different for different matrix shapes.

### The BLIS Framework

BLIS (BLAS-Like Library Instantiation Software) formalizes this as a 5-loop structure around the micro-kernel:

```
Loop 5 (outermost): tiles along n (columns of C), sized for L3
  Loop 4: tiles along k (shared dimension), sized for L2
    Loop 3: tiles along m (rows of C), sized for L1
      Pack A panel into contiguous buffer
      Loop 2: micro-tile columns of C
        Loop 1: micro-tile rows of C
          Micro-kernel: 6×16 register-tiled FMA loop
```

Each loop level is designed so that the data it works on fits in the corresponding cache level. The pack operations happen at the boundaries where data transitions between cache levels.

### The GOTO Algorithm

This tiling strategy was formalized by Kazushige Goto (pronounced "go-to") in his 2008 paper "Anatomy of High-Performance Matrix Multiplication." It's the basis of GotoBLAS, which became OpenBLAS. The key insight: **the optimal loop ordering and tile sizes can be derived from the cache sizes alone**, independent of the specific matrix dimensions.

---

## Branch-Free and Alignment

### Aligned Loads

```
_mm256_load_ps    — requires 32-byte aligned address, slightly faster
_mm256_loadu_ps   — no alignment requirement, ~0-5% slower
```

BLAS ensures packed buffers are 32-byte (or 64-byte for AVX-512) aligned using `posix_memalign` or `_mm_malloc`. This guarantees every SIMD load is aligned.

### Avoiding Branches

The micro-kernel should have zero conditional branches in the hot loop. Branches cause pipeline stalls if the CPU mispredicts which way they go.

Edge cases (like when the matrix dimensions aren't multiples of the tile size) are handled **outside** the micro-kernel — either by padding or by having separate cleanup code that runs after the main loop.

---

## CPU-Specific Tuning

### Why BLAS Has Multiple Code Paths

Different CPUs have:
- Different numbers of SIMD units (1 vs 2 FMA units)
- Different register counts (16 YMM for AVX2, 32 ZMM for AVX-512)
- Different cache sizes (L1 = 32 KB vs 48 KB, L2 = 256 KB vs 1 MB)
- Different load/store port counts
- Different pipeline depths

OpenBLAS detects the CPU at runtime and selects a kernel written specifically for that microarchitecture. There are separate hand-tuned micro-kernels for Haswell, Skylake, Zen 2, Zen 3, etc.

### AVX-512 (If Your CPU Has It)

```
__m512 — 512 bits = 16 floats per register
32 ZMM registers instead of 16 YMM registers
```

Double the floats per instruction and double the registers. The micro-kernel can compute a larger tile:

```
AVX2:   6×16  tile, 12 registers for C, ~32 FLOP/cycle peak
AVX-512: 14×32 tile, 28 registers for C, ~64 FLOP/cycle peak
```

But AVX-512 has caveats: many CPUs downclock when using it, so the higher FLOP/cycle doesn't always translate to higher FLOP/second.

---

## Amortized Costs and When None of This Matters

### Small Matrix Reality

For your MNIST matmul — (676, 9) @ (9, 32):

```
Total FLOP: 2 × 676 × 9 × 32 = 389,000
```

At 50 GFLOPS (single-core BLAS), that's 0.008 ms. The overhead of function calls, packing, and tiling decisions is comparable to the actual computation.

For small matrices, BLAS skips most of this machinery and falls back to a simpler kernel. The optimizations in this document matter for matrices larger than ~256×256.

### The Pareto Curve

```
Effort to implement         Performance vs BLAS
─────────────────────       ───────────────────
Naive C:           1 hour   10-50x slower
Loop reorder:      10 min   3-10x slower
SIMD:              1 day    2-5x slower
Tiling:            1 day    2-3x slower
Register tiling:   1 week   1.3-2x slower
Packing:           3 days   1.1-1.5x slower
Prefetch + tuning: 1 week   1.05-1.2x slower
Per-CPU assembly:  months   1.0x (you are BLAS now)
```

The returns diminish rapidly after register tiling. The last 20% of performance takes 80% of the effort. That's why BLAS libraries are maintained by teams over decades.

---

## Further Reading

- **"Anatomy of High-Performance Matrix Multiplication"** — Kazushige Goto, Robert van de Geijn (2008). The foundational paper.
- **BLIS project** — https://github.com/flame/blis — clean, well-documented BLAS implementation. Best source for understanding the 5-loop structure.
- **"How to Optimize a GEMM"** — step-by-step tutorial from the FLAME project. Walks through each optimization level with actual code.
- **Agner Fog's optimization manuals** — https://agner.org/optimize/ — definitive reference for x86 microarchitecture, instruction latencies, and SIMD programming. Free PDFs.
- **Intel Intrinsics Guide** — https://www.intel.com/content/www/us/en/docs/intrinsics-guide/ — searchable reference for every SIMD intrinsic.
