# Implementing MNIST CNN in Raw CUDA and C

## Scale of the Task

```
Keras:    ~40 lines of Python
PyTorch:  ~75 lines of Python
Raw C:    ~1,500-2,500 lines
CUDA+C:   ~3,000-5,000 lines (with reasonably efficient kernels)
```

---

## What You'd Have to Write Yourself

### The Easy Parts (tedious but straightforward)

- **Matrix multiplication in C** — nested loops
- **ReLU, softmax, flatten** — simple element-wise operations
- **Loading MNIST data** — parse the IDX binary file format
- **Memory allocation and layout** — manual malloc/free

### The Hard Parts

| Component | Why it's hard |
|-----------|---------------|
| Conv2D forward | Naive nested loops are simple but ~100x slower than cuDNN. Efficient GPU conv requires im2col transform or Winograd algorithm |
| Conv2D backward | You need to derive and implement the gradient w.r.t. weights, bias, AND input — three separate kernel implementations |
| CUDA kernels | Thread blocks, shared memory tiling, memory coalescing, bank conflicts — getting correctness is easy, getting performance is the real work |
| Backprop chain | You're implementing autograd by hand. Every layer needs a forward AND backward function, and you must get the chain rule right through the entire graph |
| Adam optimizer | Momentum, second moment, bias correction, per-parameter state — not hard conceptually but lots of bookkeeping |
| Numerical stability | Softmax overflows without the log-sum-exp trick. Gradient scaling issues. Float32 precision edge cases |
| Memory management | Manual GPU malloc/free, host-device transfers, no garbage collection, memory leaks are silent |

---

## What You'd Learn

- How GPUs actually execute parallel work
- Why memory layout (NCHW vs NHWC) matters for performance
- What autograd frameworks actually do under the hood
- Why cuDNN exists (their conv implementations took years to optimize)

---

## A Practical Path

1. **Dense-only network in C** — no conv, just MNIST pixels → dense → softmax. ~500 lines. Gets you comfortable with forward/backward passes and weight updates by hand.
2. **Add CUDA kernels** for the matrix multiplications. Learn thread blocks, grid dimensions, host-device memory transfers.
3. **Add Conv2D** — this is where the real learning happens. Implement im2col to turn convolution into matrix multiplication, then reuse your existing matmul kernel.
4. **Benchmark against PyTorch** and see the 10-100x gap, then try to close it with shared memory tiling and memory coalescing.

---

## Reference Projects

- **llm.c** (Andrej Karpathy) — GPT-2 training in raw C/CUDA. Same spirit, transformers instead of CNNs. https://github.com/karpathy/llm.c
- **cuda-neural-network** (community projects on GitHub) — various minimal CNN implementations in CUDA
- **NVIDIA cuDNN documentation** — to understand what the optimized implementations actually do and why they're fast
