# Conv2D Backward Pass

The backward pass of Conv2D computes **three gradients**:

1. **dL/dW** — gradient w.r.t. the kernel weights (to update them)
2. **dL/db** — gradient w.r.t. the bias (to update it)
3. **dL/dX** — gradient w.r.t. the input (to pass back to the previous layer)

This is the hardest part of implementing a CNN from scratch. The forward pass is intuitive; the backward pass requires thinking carefully about the chain rule applied to sliding windows.

---

## Setup / Notation

```
X       = input          (C_in, H_in, W_in)
W       = kernel weights (C_out, C_in, K, K)
b       = bias           (C_out,)
Y       = output         (C_out, H_out, W_out)
dL/dY   = upstream gradient (same shape as Y)
```

The forward pass was:

```
Y[f, i, j] = b[f] + Σ_c Σ_p Σ_q  W[f, c, p, q] * X[c, i+p, j+q]
```

Where f = filter index, c = input channel, (p, q) = kernel position, (i, j) = output position. Stride=1, no padding for simplicity.

---

## 1. Gradient w.r.t. Bias (dL/db)

This is the easiest one. Since `Y[f, i, j] = ... + b[f]`, the bias contributes equally to every spatial position in the output for filter f.

```
dL/db[f] = Σ_i Σ_j  dL/dY[f, i, j]
```

In words: **sum the upstream gradient over all spatial positions** for each filter.

```python
# NumPy
dL_db = dL_dY.sum(axis=(1, 2))    # shape: (C_out,)
```

This is why bias has one value per filter — all positions share it, so the gradient is the sum over all positions.

---

## 2. Gradient w.r.t. Weights (dL/dW)

From the forward pass: `Y[f, i, j] = ... + W[f, c, p, q] * X[c, i+p, j+q] + ...`

Taking the derivative of Y w.r.t. W[f, c, p, q]:

```
∂Y[f, i, j] / ∂W[f, c, p, q] = X[c, i+p, j+q]
```

Applying the chain rule and summing over all positions where this weight was used:

```
dL/dW[f, c, p, q] = Σ_i Σ_j  dL/dY[f, i, j] * X[c, i+p, j+q]
```

In words: **correlate the upstream gradient with the input**. For each filter f and channel c, slide dL/dY over X and compute dot products — this is itself a convolution!

```python
# Pseudocode
for f in range(C_out):
    for c in range(C_in):
        for p in range(K):
            for q in range(K):
                dL_dW[f, c, p, q] = sum over (i,j) of dL_dY[f, i, j] * X[c, i+p, j+q]
```

The key insight: **computing dL/dW is a convolution of X with dL/dY as the kernel.**

---

## 3. Gradient w.r.t. Input (dL/dX)

This is the tricky one, and you need it to propagate gradients to earlier layers.

From the forward pass, each input pixel X[c, m, n] contributes to multiple output positions. Specifically, X[c, m, n] appears in Y[f, i, j] whenever `i+p = m` and `j+q = n`, i.e., `i = m-p` and `j = n-q`.

```
dL/dX[c, m, n] = Σ_f Σ_p Σ_q  dL/dY[f, m-p, n-q] * W[f, c, p, q]
```

(where the sum only includes valid indices)

This is a **full convolution** (with padding) of dL/dY with the **flipped** (rotated 180°) kernel.

```python
# Pseudocode
W_rot = W[:, :, ::-1, ::-1]       # rotate kernel 180°
dL_dX = full_convolution(dL_dY, W_rot)
```

### Why flip the kernel?

In the forward pass, W[f, c, p, q] multiplied X[c, i+p, j+q]. Going backward, we need to figure out "how much did X[c, m, n] affect the loss?" Since i = m-p, the kernel index p maps to position (m-p) in the output — this reversal is exactly what flipping the kernel achieves.

### Why full convolution?

A "full" convolution pads dL/dY so the kernel can slide beyond the edges. This is necessary because border input pixels participated in fewer output positions than center pixels, and the full convolution correctly accounts for this.

```
Forward:   valid conv of X with W       → shrinks spatial dims
Backward:  full conv of dL/dY with W_rot → grows spatial dims back to input size
```

---

## All Three Together (Pseudocode)

```python
def conv2d_backward(dL_dY, X, W, b):
    C_out, C_in, K, _ = W.shape
    _, H_out, W_out = dL_dY.shape

    # 1. Bias gradient
    dL_db = np.zeros_like(b)
    for f in range(C_out):
        dL_db[f] = np.sum(dL_dY[f])

    # 2. Weight gradient
    dL_dW = np.zeros_like(W)
    for f in range(C_out):
        for c in range(C_in):
            for p in range(K):
                for q in range(K):
                    # patch of X that was multiplied by W[f,c,p,q] during forward
                    patch = X[c, p:p+H_out, q:q+W_out]
                    dL_dW[f, c, p, q] = np.sum(dL_dY[f] * patch)

    # 3. Input gradient
    dL_dX = np.zeros_like(X)
    # Pad dL_dY with (K-1) zeros on each side
    padded = np.pad(dL_dY, ((0,0), (K-1, K-1), (K-1, K-1)))
    W_rot = W[:, :, ::-1, ::-1]   # flip kernels 180°
    for c in range(C_in):
        for i in range(X.shape[1]):
            for j in range(X.shape[2]):
                for f in range(C_out):
                    dL_dX[c, i, j] += np.sum(
                        padded[f, i:i+K, j:j+K] * W_rot[f, c]
                    )

    return dL_dW, dL_db, dL_dX
```

This is O(C_out × C_in × K² × H_out × W_out) — exactly as expensive as the forward pass.

---

## The im2col Trick

Nobody implements the nested loops above in practice. Instead, the **im2col** (image to column) trick converts convolution into matrix multiplication:

**Forward:**
1. Extract every K×K patch from the input and lay them out as rows of a matrix (im2col).
2. Reshape the kernel weights into a matrix.
3. Matrix multiply → output.

**Backward:**
1. dL/dW becomes a matrix multiply between im2col(X)ᵀ and dL/dY.
2. dL/dX becomes a matrix multiply between dL/dY and Wᵀ, then a col2im operation (scatter the columns back into the image layout).

```
Forward:    Y = W_matrix @ im2col(X)
dL/dW:      dL_dW_matrix = dL_dY_matrix @ im2col(X)ᵀ
dL/dX:      dL_dX = col2im(W_matrixᵀ @ dL_dY_matrix)
```

This is how cuDNN, PyTorch, and every practical implementation does it. Matrix multiplication is the most optimized operation on GPUs, so converting everything to matmul is a huge win.

---

## Tracing Through Your MNIST Model's Backward Pass

During one training step on your Keras model, the backward pass flows in reverse:

```
Loss (categorical cross-entropy)
  ↓ dL/dY
Dense(10, softmax)         → compute dL/dW_dense, dL/db_dense, dL/dX_dense
  ↓
Dropout(0.5)               → zero out same neurons as forward, scale the rest
  ↓
Flatten                    → reshape gradient back to (5, 5, 64)
  ↓
MaxPooling2D               → gradient goes only to the max positions (rest = 0)
  ↓
Conv2D(64, 3)              → compute dL/dW, dL/db, dL/dX (the hard part)
  ↓
MaxPooling2D               → gradient goes only to the max positions
  ↓
Conv2D(32, 3)              → compute dL/dW, dL/db, dL/dX
  ↓
(input — no further backprop needed)
```

Each arrow is one layer's backward function receiving the upstream gradient and passing its own downstream.

---

## Key Takeaways

1. Conv2D backward computes **three separate gradients**: weights, bias, and input.
2. **Bias gradient** is the simplest — just sum the upstream gradient spatially.
3. **Weight gradient** is a convolution of the input with the upstream gradient.
4. **Input gradient** is a full convolution of the upstream gradient with the flipped kernel.
5. In practice, all of these are converted to **matrix multiplications** via im2col/col2im.
6. The backward pass has the **same computational cost** as the forward pass.
7. The flipped kernel in the input gradient is why this operation is called "convolution" — true mathematical convolution involves a kernel flip, and the backward pass is where it actually shows up.
