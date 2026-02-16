# CLAUDE.md

## What This Is

A learning project: implementing a CNN for MNIST handwritten digit recognition from scratch. The approach is top-down — starting from high-level frameworks (Keras, PyTorch) and reimplementing everything in raw numpy to understand what happens under the hood.

AI-assisted for docs/reference only. Implementation logic is human-written.

## Model Architecture

```
Input (1, 28, 28)
  → Conv2D(32, 3×3) + ReLU  → (32, 26, 26)
  → MaxPool(2×2)             → (32, 13, 13)
  → Conv2D(64, 3×3) + ReLU  → (64, 11, 11)
  → MaxPool(2×2)             → (64, 5, 5)
  → Flatten                  → (1600,)
  → Dropout(0.5)             → (1600,)
  → Dense(10) + Softmax      → (10,)
```

34,826 total learnable params. Trained with Adam, cross-entropy loss, batch size 128, 5 epochs. Target: ~99% accuracy.

## Repo Layout

```
implementations/
  keras_mnist.py      # High-level Keras baseline (reference)
  pytorch_mnist.py    # Explicit PyTorch baseline (reference)
  conv2d.py           # Naive conv2d — 6 nested loops, extremely slow
  conv2dim2col.py     # Single-channel conv2d via im2col + as_strided + matmul
  full.py             # WIP: the from-scratch numpy implementation (main working file)

docs/                 # Reference material (AI-generated)
  layers/             # What each layer does (conv2d, maxpooling, relu, dense, flatten, dropout)
  training/           # Adam optimizer, backprop chain rule, numerical stability, cross-entropy loss
  low-level/          # conv2d backward, im2col, matmul, SIMD/tiling, CUDA/C
  tooling/            # Profiling (pyinstrument, flamegraphs)

architecture.md       # Full implementation blueprint — all phases, function signatures, build order
```

## Current Progress

The project follows the phased plan in `architecture.md`. Status:

- **Done**: Full forward pass end-to-end — data loading, one-hot encoding, generic multi-channel conv2d (im2col), relu, maxpooling, flatten, dropout, dense, softmax, cross-entropy loss
- **Not started**: Backward passes, optimizer (Adam), training loop, evaluation

`full.py` is the active working file. The complete forward pipeline runs:
conv2d → relu → maxpool → conv2d → relu → maxpool → flatten → dropout → dense → softmax → cross-entropy loss

## Key Technical Details

- **im2col trick**: Reorganizes conv input into a matrix so convolution becomes matmul. Uses `np.lib.stride_tricks.as_strided` for zero-copy patch extraction, then reshape + `@` for BLAS-optimized matmul.
- **Single-channel kernel shape**: `(32, 1, 3, 3)` reshaped to `(9, 32)` for matmul.
- **Multi-channel kernel shape**: `(64, 32, 3, 3)` reshaped to `(288, 64)` for matmul. Uses 5D as_strided with strides `(s[1], s[2], s[0], s[1], s[2])` to extract `(H_out, W_out, C_in, 3, 3)` patches.
- **MaxPool**: Uses `as_strided` to create `(C, H/2, W/2, 2, 2)` view, then `np.max` over last two axes.
- Data loaded via `keras.datasets.mnist.load_data()` (returns numpy arrays directly, faster than torchvision).

## Running

```bash
source venv/bin/activate
python implementations/keras_mnist.py       # ~99% acc baseline
python implementations/pytorch_mnist.py     # ~99% acc baseline
python implementations/full.py              # WIP forward pass
```

Dependencies: `keras torch torchvision numpy pyinstrument`

## Conventions

- Comments are informal/personal notes, not docstrings
- Profiling artifacts (`.prof`, `.svg`, `pyinstrument.txt`) are gitignored
- `data/` and `venv/` are gitignored
