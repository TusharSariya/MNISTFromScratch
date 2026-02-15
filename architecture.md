# MNIST CNN Architecture — Implementation Blueprint

No code. Only what each function does, what it takes, what it returns, and what order to build it in.

---

## The Network

```
Input (1, 28, 28)
  → Conv2D(32 filters, 3×3) + ReLU    → (32, 26, 26)
  → MaxPool(2×2)                       → (32, 13, 13)
  → Conv2D(64 filters, 3×3) + ReLU    → (64, 11, 11)
  → MaxPool(2×2)                       → (64, 5, 5)
  → Flatten                            → (1600,)
  → Dropout(0.5)                       → (1600,)
  → Dense(10) + Softmax                → (10,)
  → Cross-Entropy Loss                 → scalar
```

---

## Phase 1: Data Loading

### load_mnist()

- **What:** Read MNIST binary files (IDX format) from disk.
- **Input:** File paths for images and labels.
- **Output:** Images as float array (N, 1, 28, 28) normalized to [0, 1]. Labels as int array (N,).
- **Notes:** Raw files are unsigned 8-bit ints (0–255). Divide by 255.0. Add channel dimension.

### to_one_hot(labels, num_classes)

- **What:** Convert integer labels to one-hot vectors.
- **Input:** Integer array (N,), number of classes (10).
- **Output:** Float array (N, 10). Row i has a 1.0 at index labels[i], 0.0 elsewhere.

### batch_iterator(images, labels, batch_size, shuffle)

- **What:** Yield (image_batch, label_batch) pairs.
- **Input:** Full image and label arrays, batch size, whether to shuffle.
- **Output:** Iterator of (batch_images, batch_labels) tuples.
- **Notes:** If shuffle, randomize order each epoch. Last batch may be smaller.

---

## Phase 2: Layer Forward Functions

Implement each forward function independently. Each takes an input and returns an output. Test each one in isolation before combining.

### conv2d_forward(input, weights, bias)

- **What:** Slide each filter across the input, compute dot products, add bias.
- **Input:** input (C_in, H, W), weights (C_out, C_in, K, K), bias (C_out,).
- **Output:** output (C_out, H_out, W_out).
- **H_out** = H - K + 1, **W_out** = W - K + 1 (no padding, stride 1).
- **Must save:** input (needed for backward pass).

### relu_forward(input)

- **What:** Element-wise max(0, x).
- **Input:** Array of any shape.
- **Output:** Same shape, all negatives replaced with 0.
- **Must save:** Boolean mask of where input > 0 (needed for backward).

### maxpool2d_forward(input, pool_size)

- **What:** Divide each channel into non-overlapping pool_size × pool_size windows. Keep the max of each window.
- **Input:** input (C, H, W), pool_size (int, e.g. 2).
- **Output:** output (C, H/pool_size, W/pool_size).
- **Must save:** Indices of where each max came from (needed for backward).

### flatten_forward(input)

- **What:** Reshape from (C, H, W) to (C × H × W,).
- **Input:** Any multi-dimensional array.
- **Output:** 1D vector.
- **Must save:** Original shape (needed for backward).

### dropout_forward(input, rate, training)

- **What:** During training, randomly zero out elements with probability `rate` and scale survivors by 1/(1-rate). During inference, do nothing.
- **Input:** Array of any shape, drop rate (0.5), training flag.
- **Output:** Same shape, with some elements zeroed (training) or unchanged (inference).
- **Must save:** The random mask (needed for backward).

### dense_forward(input, weights, bias)

- **What:** Matrix multiply input by weights, add bias. (Fully connected layer.)
- **Input:** input (D_in,), weights (D_out, D_in), bias (D_out,).
- **Output:** output (D_out,).
- **Must save:** input (needed for backward).

### softmax(input)

- **What:** Convert raw scores (logits) to probabilities that sum to 1.
- **Input:** Array (num_classes,).
- **Output:** Same shape, all values in [0, 1], sums to 1.
- **Stability:** Subtract max(input) before exp to prevent overflow.

---

## Phase 3: Loss Function

### cross_entropy_loss(predictions, targets)

- **What:** Measure how far the predicted probabilities are from the true labels.
- **Input:** predictions (10,) from softmax, targets (10,) one-hot.
- **Output:** Scalar loss value.
- **Formula:** -Σ targets[i] × log(predictions[i]).
- **Stability:** Clip predictions to [1e-7, 1-1e-7] before log, or use log-softmax.

### cross_entropy_gradient(predictions, targets)

- **What:** Gradient of the loss w.r.t. the softmax input (logits).
- **Input:** predictions (10,), targets (10,).
- **Output:** Gradient (10,).
- **Formula:** predictions - targets. (This is the combined softmax + cross-entropy gradient.)

---

## Phase 4: Layer Backward Functions

Each backward function receives the upstream gradient (dL/doutput) and returns the downstream gradient (dL/dinput). Layers with learnable parameters also compute dL/dW and dL/db.

Implement in any order, but test each against numerical gradients.

### conv2d_backward(upstream_grad, saved_input, weights, bias)

- **What:** Compute three gradients: dL/dweights, dL/dbias, dL/dinput.
- **Input:** upstream_grad (C_out, H_out, W_out), saved input, weights, bias.
- **Output:** dL/dweights (same shape as weights), dL/dbias (C_out,), dL/dinput (same shape as input).
- **dL/dbias:** Sum upstream_grad over spatial dimensions per filter.
- **dL/dweights:** Correlate input with upstream_grad.
- **dL/dinput:** Full convolution of upstream_grad with flipped weights.
- **This is the hardest function to implement.**

### relu_backward(upstream_grad, saved_mask)

- **What:** Pass gradient through where input was positive, block where it was negative.
- **Input:** upstream_grad (any shape), saved boolean mask.
- **Output:** downstream_grad = upstream_grad × mask.

### maxpool2d_backward(upstream_grad, saved_max_indices, input_shape, pool_size)

- **What:** Route each gradient value to the position that held the max during forward.
- **Input:** upstream_grad (C, H_out, W_out), indices of maxes, original input shape.
- **Output:** downstream_grad (C, H, W) — zeros everywhere except at max positions.

### flatten_backward(upstream_grad, original_shape)

- **What:** Reshape 1D gradient back to the original shape.
- **Input:** upstream_grad (D,), original shape (C, H, W).
- **Output:** downstream_grad reshaped to (C, H, W).

### dropout_backward(upstream_grad, saved_mask, rate)

- **What:** Zero out same positions as forward, scale the rest.
- **Input:** upstream_grad, saved mask, rate.
- **Output:** downstream_grad = upstream_grad × mask / (1 - rate).

### dense_backward(upstream_grad, saved_input, weights, bias)

- **What:** Compute dL/dweights, dL/dbias, dL/dinput.
- **Input:** upstream_grad (D_out,), saved input (D_in,), weights (D_out, D_in).
- **Output:** dL/dweights (D_out, D_in), dL/dbias (D_out,), dL/dinput (D_in,).
- **dL/dbias:** upstream_grad (directly).
- **dL/dweights:** outer product of upstream_grad and saved_input.
- **dL/dinput:** Wᵀ @ upstream_grad.

---

## Phase 5: Optimizer

### adam_init(parameters)

- **What:** Create optimizer state: m (first moment) and v (second moment) for every parameter, both initialized to zero. Set step counter t = 0.
- **Input:** List of all learnable parameter arrays.
- **Output:** Optimizer state dict.

### adam_step(parameters, gradients, state, lr, beta1, beta2, epsilon)

- **What:** Update every parameter using its gradient and the Adam update rule.
- **Input:** List of parameter arrays, matching list of gradient arrays, optimizer state, hyperparameters.
- **Output:** Updated parameters (in-place), updated state.
- **Steps per parameter:**
  1. t += 1
  2. m = beta1 × m + (1 - beta1) × grad
  3. v = beta2 × v + (1 - beta2) × grad²
  4. m_hat = m / (1 - beta1^t)
  5. v_hat = v / (1 - beta2^t)
  6. param -= lr × m_hat / (sqrt(v_hat) + epsilon)

---

## Phase 6: Weight Initialization

### init_weights(shape)

- **What:** Create a weight array with values drawn from a sensible distribution.
- **Input:** Shape tuple.
- **Output:** Float array of that shape.
- **Use Glorot/Xavier:** Uniform in [-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out))].
- **Or He/Kaiming:** Normal with std = √(2/fan_in). Better for ReLU.

### init_bias(size)

- **What:** Create a bias array initialized to zeros.
- **Input:** Number of output units.
- **Output:** Float array of zeros (size,).

---

## Phase 7: Full Forward Pass

Chain all forward functions together in order:

```
full_forward(image, params, training):

    1.  h = conv2d_forward(image, params.conv1_w, params.conv1_b)
    2.  h = relu_forward(h)
    3.  h = maxpool2d_forward(h, pool_size=2)
    4.  h = conv2d_forward(h, params.conv2_w, params.conv2_b)
    5.  h = relu_forward(h)
    6.  h = maxpool2d_forward(h, pool_size=2)
    7.  h = flatten_forward(h)
    8.  h = dropout_forward(h, rate=0.5, training=training)
    9.  h = dense_forward(h, params.dense_w, params.dense_b)
    10. h = softmax(h)

    return h, all_saved_values
```

Each step saves what the backward pass needs (input, mask, indices, etc.).

---

## Phase 8: Full Backward Pass

Walk backwards through the saved values:

```
full_backward(predictions, targets, saved_values, params):

    1.  grad = cross_entropy_gradient(predictions, targets)
    2.  grad, dW_dense, db_dense = dense_backward(grad, ...)
    3.  grad = dropout_backward(grad, ...)
    4.  grad = flatten_backward(grad, ...)
    5.  grad = maxpool2d_backward(grad, ...)
    6.  grad = relu_backward(grad, ...)
    7.  grad, dW_conv2, db_conv2 = conv2d_backward(grad, ...)
    8.  grad = maxpool2d_backward(grad, ...)
    9.  grad = relu_backward(grad, ...)
    10. _, dW_conv1, db_conv1 = conv2d_backward(grad, ...)

    return all_gradients
```

---

## Phase 9: Training Loop

```
train(images, labels, params, optimizer_state, epochs, batch_size, lr):

    for each epoch:
        for each (batch_images, batch_labels) from batch_iterator:
            for each (image, label) in batch:
                predictions, saved = full_forward(image, params, training=True)
                loss = cross_entropy_loss(predictions, label)
                gradients = full_backward(predictions, label, saved, params)
                accumulate gradients across batch
            average gradients over batch
            adam_step(params, avg_gradients, optimizer_state, lr)
        print epoch loss and accuracy
```

---

## Phase 10: Evaluation

```
evaluate(images, labels, params):

    total_loss = 0
    correct = 0
    for each (image, label):
        predictions, _ = full_forward(image, params, training=false)
        total_loss += cross_entropy_loss(predictions, label)
        if argmax(predictions) == argmax(label):
            correct += 1
    print average loss, accuracy
```

---

## Implementation Order

Build and test each piece before moving on. Earlier phases don't depend on later ones.

| Order | What | Why this order |
|-------|------|----------------|
| 1 | load_mnist, to_one_hot, batch_iterator | Need data to test anything |
| 2 | init_weights, init_bias | Need parameters before forward pass |
| 3 | dense_forward, relu_forward, softmax | Simplest forward functions |
| 4 | dense_backward, relu_backward | Simplest backward functions — test with numerical gradients |
| 5 | cross_entropy_loss, cross_entropy_gradient | Can now train a dense-only network |
| 6 | adam_init, adam_step | Can now train and see if loss goes down |
| 7 | conv2d_forward | The core operation |
| 8 | maxpool2d_forward, flatten_forward, dropout_forward | Remaining forward functions |
| 9 | conv2d_backward | The hardest function |
| 10 | maxpool2d_backward, flatten_backward, dropout_backward | Remaining backward functions |
| 11 | full_forward, full_backward | Wire everything together |
| 12 | train, evaluate | The training loop |

---

## All Learnable Parameters

| Layer | Weights shape | Bias shape | Total params |
|-------|--------------|------------|-------------|
| Conv2D #1 | (32, 1, 3, 3) | (32,) | 320 |
| Conv2D #2 | (64, 32, 3, 3) | (64,) | 18,496 |
| Dense | (10, 1600) | (10,) | 16,010 |
| **Total** | | | **34,826** |

Adam needs to store m and v for each of these 34,826 values.

---

## Testing Strategy

After implementing each function, verify it:

1. **Forward functions:** Compare output shape and values against PyTorch equivalent.
2. **Backward functions:** Use **numerical gradient checking** — perturb each weight by ε, compute (loss+ - loss-) / 2ε, compare to your analytical gradient. They should match to ~1e-5.
3. **Full network:** Train on a tiny subset (e.g. 100 images). Loss should decrease. If it doesn't, a backward function is wrong.
4. **Final test:** Train on full MNIST. Should reach ~99% test accuracy after 5 epochs, matching the Keras/PyTorch versions.
