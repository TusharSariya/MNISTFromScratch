# MNIST CNN From Scratch

Implementing a convolutional neural network for handwritten digit recognition as close to the metal as possible. Starting from high-level frameworks (Keras, PyTorch) and working down to raw numpy, hand-written convolutions, and understanding the CPU-level optimizations that make it all fast.

AI-assisted for documentation and reference material, but the implementation logic is human-written.

## Model Architecture

- Conv2D(32, 3x3) → MaxPool(2x2)
- Conv2D(64, 3x3) → MaxPool(2x2)
- Flatten → Dropout(0.5) → Dense(10, softmax)

Trained with Adam optimizer, cross-entropy loss, batch size 128, 5 epochs on MNIST (60k train / 10k test). Expects ~99% accuracy.

## Repo Structure

```
implementations/
├── keras_mnist.py        # High-level Keras implementation (~48 lines)
├── pytorch_mnist.py      # Explicit PyTorch implementation (~75 lines)
├── conv2d.py             # Naive conv2d forward pass with nested loops
├── conv2dim2col.py       # Conv2d using im2col + matmul
├── stoplight.py          # Toy 2-layer network in numpy (streetlight walk/stop)
├── stoplight.c           # Same network in C with manual matmul
└── stoplight.cu          # CUDA boilerplate (multiply-by-two kernel)

docs/
├── layers/               # What each layer does (math, shapes, forward pass)
│   ├── conv2d.md
│   ├── maxpooling2d.md
│   ├── dense.md
│   ├── flatten.md
│   └── dropout.md
├── training/             # How the network learns
│   ├── adam-optimizer.md
│   ├── backprop-chain.md
│   └── numerical-stability.md
├── low-level/            # What happens under the hood
│   ├── conv2d-backward.md
│   ├── raw-cuda-c.md
│   ├── matmul.md
│   ├── matmul-cpu.md
│   └── matmul-cpu-fun-stuff.md
└── tooling/
    └── profiling.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install keras torch torchvision numpy pyinstrument
```

## Usage

```bash
python implementations/keras_mnist.py      # high-level baseline
python implementations/pytorch_mnist.py    # explicit baseline
python implementations/conv2dim2col.py     # numpy im2col implementation
```

### Stoplight (toy network)

```bash
# Python
python implementations/stoplight.py

# C
gcc -O2 -o stoplight implementations/stoplight.c -lm
./stoplight

# CUDA (requires nvcc)
nvcc -o stoplight_cuda implementations/stoplight.cu
./stoplight_cuda
```

Both `stoplight.py` and `stoplight.c` print per-iteration timing percentiles (p50/p90/p99) after training. See `docs/benchmarking_tips.md` for how to get stable results and `docs/stoplight-c.md` for build/debug details.

### Profiling CUDA (Nsight)

You can profile the CUDA binary with **Nsight Systems** (timeline) and **Nsight Compute** (kernel-level metrics). Both require an NVIDIA GPU and the Nsight tools (from the CUDA toolkit or standalone install).

| Tool | Use for |
|------|--------|
| **Nsight Systems** | Where time is spent (CPU/GPU timeline), kernel launches, memory transfers, API/sync behavior |
| **Nsight Compute** | Why a kernel is slow: occupancy, memory throughput, warp utilization, etc. |

```bash
# Timeline profile (creates stoplight_report.nsys-rep)
nsys profile -o stoplight_report ./stoplight_cuda

# Kernel-level profile (creates stoplight_ncu.ncu-rep)
ncu -o stoplight_ncu ./stoplight_cuda

# Quick kernel summary without GUI
ncu --print-summary per-kernel ./stoplight_cuda
```

Open `.nsys-rep` in the Nsight Systems GUI and `.ncu-rep` in the Nsight Compute GUI. The `learn` kernel uses a single block of 16 threads, so occupancy is low and launch overhead dominates; the tools still work for learning and for larger kernels later.
