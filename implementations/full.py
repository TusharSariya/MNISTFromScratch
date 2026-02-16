import numpy as np
import keras
import math
import time

# ============ Phase 1: Data Loading ============

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#add a chanel dimention, 1 chanel
x_train = np.expand_dims(x_train, 1)  # (60000, 1, 28, 28)
x_test = np.expand_dims(x_test, 1)    # (10000, 1, 28, 28)

def to_one_hot_batch(labels, num_classes=10):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot

y_train_one_hot = to_one_hot_batch(y_train)
y_test_one_hot = to_one_hot_batch(y_test)

# ============ Phase 6: Weight Initialization ============

# He init for conv layers (followed by ReLU)
# fan_in = C_in * K * K
conv1_w = np.random.randn(32, 1, 3, 3) * np.sqrt(2.0 / 9)
conv1_w = conv1_w.reshape(32, 9).T      # (9, 32)

conv2_w = np.random.randn(64, 32, 3, 3) * np.sqrt(2.0 / 288)
conv2_w = conv2_w.reshape(64, 288).T    # (288, 64)

# Xavier init for dense layer (followed by softmax, not ReLU)
dense_w = np.random.randn(1600, 10) * np.sqrt(2.0 / (1600 + 10))
dense_b = np.zeros(10)

# ============ Phase 2: Forward Functions ============

def im2col(input, K=3):
    # input: (C_in, H, W) -> patches: (H_out * W_out, C_in * K * K)
    C_in, H, W = input.shape
    H_out = H - K + 1
    W_out = W - K + 1
    s = input.strides
    view = np.lib.stride_tricks.as_strided(
        input,
        (H_out, W_out, C_in, K, K),
        (s[1], s[2], s[0], s[1], s[2])
    )
    return view.reshape(H_out * W_out, C_in * K * K)

def conv2d_forward(input, kernel):
    # input: (C_in, H, W), kernel: (C_in*K*K, C_out)
    # output: (C_out, H_out, W_out)
    C_in, H, W = input.shape
    K = int(math.sqrt(kernel.shape[0] / C_in))
    H_out = H - K + 1
    W_out = W - K + 1
    patches = im2col(input, K)
    # save patches (copy! as_strided shares memory) and input shape for backward
    output = (patches @ kernel).T.reshape(kernel.shape[1], H_out, W_out)
    return output, patches.copy(), input.shape

def relu_forward(input):
    mask = input > 0
    return input * mask, mask

def maxpool_forward(input, pool_size=2):
    # input: (C, H, W) -> output: (C, H//pool, W//pool)
    C, H, W = input.shape
    H_out = H // pool_size
    W_out = W // pool_size
    s = input.strides
    view = np.lib.stride_tricks.as_strided(input,
        (C, H_out, W_out, pool_size, pool_size),
        (s[0], s[1]*pool_size, s[2]*pool_size, s[1], s[2]))
    output = np.max(view, axis=(3, 4))
    # save which element was max in each window (for backward)
    max_mask = (view == output[:, :, :, None, None])
    return output, max_mask, (C, H, W)

def flatten_forward(input):
    return input.reshape(-1), input.shape

def dropout_forward(input, rate=0.5, training=True):
    if not training:
        return input, None
    mask = np.random.rand(*input.shape) > rate
    return input * mask / (1 - rate), mask

def dense_forward(input, weights, bias):
    # (1600,) @ (1600, 10) + (10,) -> (10,)
    return input @ weights + bias, input.copy()

def softmax(input):
    # subtract max for numerical stability
    e = np.exp(input - np.max(input))
    return e / e.sum()

# ============ Phase 3: Loss ============

def cross_entropy_loss(prediction, target):
    prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
    return -np.sum(target * np.log(prediction))

def cross_entropy_gradient(prediction, target):
    # softmax + cross-entropy combined: dL/d(logits) = predictions - targets
    return prediction - target

# ============ Phase 4: Backward Functions ============

def col2im(cols, input_shape, K=3):
    # reverse of im2col: scatter and accumulate overlapping patches back
    # cols: (H_out * W_out, C_in * K * K) -> output: (C_in, H, W)
    C_in, H, W = input_shape
    H_out = H - K + 1
    W_out = W - K + 1
    cols_reshaped = cols.reshape(H_out, W_out, C_in, K, K)
    dinput = np.zeros((C_in, H, W))
    for y in range(H_out):
        for x in range(W_out):
            dinput[:, y:y+K, x:x+K] += cols_reshaped[y, x]
    return dinput

def conv2d_backward(upstream_grad, saved_patches, input_shape, kernel):
    # upstream_grad: (C_out, H_out, W_out)
    # returns: dinput, dkernel
    C_out, H_out, W_out = upstream_grad.shape
    grad_2d = upstream_grad.reshape(C_out, H_out * W_out).T  # (H_out*W_out, C_out)

    # dkernel: how much each weight contributed to the loss
    # patches.T @ grad_2d = (C_in*K*K, H_out*W_out) @ (H_out*W_out, C_out) = (C_in*K*K, C_out)
    dkernel = saved_patches.T @ grad_2d

    # dpatches: gradient flowing back to the im2col patches
    # grad_2d @ kernel.T = (H_out*W_out, C_out) @ (C_out, C_in*K*K) = (H_out*W_out, C_in*K*K)
    dpatches = grad_2d @ kernel.T

    # col2im: scatter patch gradients back to the original input positions
    # overlapping patches accumulate (this is why col2im sums)
    C_in = input_shape[0]
    K = int(math.sqrt(kernel.shape[0] / C_in))
    dinput = col2im(dpatches, input_shape, K)
    return dinput, dkernel

def relu_backward(upstream_grad, mask):
    # gradient flows through where input was positive, blocked where negative
    return upstream_grad * mask

def maxpool_backward(upstream_grad, max_mask, input_shape, pool_size=2):
    # route gradient to where the max was in each pool window
    C, H, W = input_shape
    H_out = H // pool_size
    W_out = W // pool_size
    # broadcast gradient to max positions, zero everywhere else
    grad_expanded = upstream_grad[:, :, :, None, None] * max_mask
    # reshape back: (C, H_out, pool, W_out, pool) -> (C, H_out*pool, W_out*pool)
    dinput_small = grad_expanded.transpose(0, 1, 3, 2, 4).reshape(C, H_out * pool_size, W_out * pool_size)
    # if H or W was odd, the last row/col was never pooled — gets zero gradient
    dinput = np.zeros((C, H, W))
    dinput[:, :H_out*pool_size, :W_out*pool_size] = dinput_small
    return dinput

def flatten_backward(upstream_grad, original_shape):
    return upstream_grad.reshape(original_shape)

def dropout_backward(upstream_grad, mask, rate=0.5):
    if mask is None:
        return upstream_grad
    # same mask, same scaling as forward
    return upstream_grad * mask / (1 - rate)

def dense_backward(upstream_grad, saved_input, weights):
    # upstream_grad: (10,), saved_input: (1600,), weights: (1600, 10)
    # dweights: outer product of input and gradient
    dweights = np.outer(saved_input, upstream_grad)  # (1600, 10)
    dbias = upstream_grad.copy()                      # (10,)
    # dinput: pass gradient back through the weights
    dinput = weights @ upstream_grad                   # (1600, 10) @ (10,) = (1600,)
    return dinput, dweights, dbias

# ============ Phase 5: Adam Optimizer ============

def adam_init(params):
    return {
        't': 0,
        'm': [np.zeros_like(p) for p in params],
        'v': [np.zeros_like(p) for p in params],
    }

def adam_step(params, grads, state, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
    state['t'] += 1
    t = state['t']
    for i in range(len(params)):
        state['m'][i] = beta1 * state['m'][i] + (1 - beta1) * grads[i]
        state['v'][i] = beta2 * state['v'][i] + (1 - beta2) * grads[i]**2
        m_hat = state['m'][i] / (1 - beta1**t)
        v_hat = state['v'][i] / (1 - beta2**t)
        params[i] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

# ============ Phase 7: Full Forward Pass ============

def full_forward(image, conv1_w, conv2_w, dense_w, dense_b, training=True):
    saved = {}

    h, saved['patches1'], saved['shape1'] = conv2d_forward(image, conv1_w)
    h, saved['relu_mask1'] = relu_forward(h)
    h, saved['pool_mask1'], saved['pool_shape1'] = maxpool_forward(h)

    h, saved['patches2'], saved['shape2'] = conv2d_forward(h, conv2_w)
    h, saved['relu_mask2'] = relu_forward(h)
    h, saved['pool_mask2'], saved['pool_shape2'] = maxpool_forward(h)

    h, saved['flatten_shape'] = flatten_forward(h)
    h, saved['dropout_mask'] = dropout_forward(h, training=training)
    h, saved['dense_input'] = dense_forward(h, dense_w, dense_b)
    h = softmax(h)

    return h, saved

# ============ Phase 8: Full Backward Pass ============

def full_backward(predictions, target, saved, conv1_w, conv2_w, dense_w):
    # start: dL/d(logits) = predictions - targets
    grad = cross_entropy_gradient(predictions, target)

    # walk backwards through every layer
    grad, dW_dense, db_dense = dense_backward(grad, saved['dense_input'], dense_w)
    grad = dropout_backward(grad, saved['dropout_mask'])
    grad = flatten_backward(grad, saved['flatten_shape'])

    grad = maxpool_backward(grad, saved['pool_mask2'], saved['pool_shape2'])
    grad = relu_backward(grad, saved['relu_mask2'])
    grad, dW_conv2 = conv2d_backward(grad, saved['patches2'], saved['shape2'], conv2_w)

    grad = maxpool_backward(grad, saved['pool_mask1'], saved['pool_shape1'])
    grad = relu_backward(grad, saved['relu_mask1'])
    _, dW_conv1 = conv2d_backward(grad, saved['patches1'], saved['shape1'], conv1_w)

    return dW_conv1, dW_conv2, dW_dense, db_dense

# ============ Phase 9: Training Loop ============

def train(x_train, y_train_oh, conv1_w, conv2_w, dense_w, dense_b,
          epochs=5, batch_size=128, lr=0.001):
    params = [conv1_w, conv2_w, dense_w, dense_b]
    state = adam_init(params)
    n = len(x_train)

    for epoch in range(1, epochs + 1):
        # shuffle
        indices = np.random.permutation(n)
        total_loss = 0
        correct = 0

        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_idx = indices[batch_start:batch_end]
            bs = len(batch_idx)

            # accumulate gradients over batch
            dW1_acc = np.zeros_like(conv1_w)
            dW2_acc = np.zeros_like(conv2_w)
            dWd_acc = np.zeros_like(dense_w)
            dbd_acc = np.zeros_like(dense_b)

            for idx in batch_idx:
                pred, saved = full_forward(x_train[idx], conv1_w, conv2_w, dense_w, dense_b, training=True)

                total_loss += cross_entropy_loss(pred, y_train_oh[idx])
                if np.argmax(pred) == np.argmax(y_train_oh[idx]):
                    correct += 1

                dW1, dW2, dWd, dbd = full_backward(pred, y_train_oh[idx], saved, conv1_w, conv2_w, dense_w)
                dW1_acc += dW1
                dW2_acc += dW2
                dWd_acc += dWd
                dbd_acc += dbd

            # average gradients over batch, then update
            grads = [dW1_acc / bs, dW2_acc / bs, dWd_acc / bs, dbd_acc / bs]
            adam_step(params, grads, state, lr=lr)

            # progress
            done = min(batch_end, n)
            print(f"\rEpoch {epoch}/{epochs} - {done}/{n}", end="", flush=True)

        avg_loss = total_loss / n
        accuracy = correct / n
        print(f"\rEpoch {epoch}/{epochs} - loss: {avg_loss:.4f} - accuracy: {accuracy:.4f}")

# ============ Phase 10: Evaluation ============

def evaluate(x_test, y_test_oh, conv1_w, conv2_w, dense_w, dense_b):
    total_loss = 0
    correct = 0
    n = len(x_test)
    for i in range(n):
        pred, _ = full_forward(x_test[i], conv1_w, conv2_w, dense_w, dense_b, training=False)
        total_loss += cross_entropy_loss(pred, y_test_oh[i])
        if np.argmax(pred) == np.argmax(y_test_oh[i]):
            correct += 1
        if (i + 1) % 1000 == 0:
            print(f"\rEvaluating: {i+1}/{n}", end="", flush=True)
    print(f"\rTest loss: {total_loss/n:.4f} - Test accuracy: {correct/n:.4f}")

# ============ Run ============

print("Training on full MNIST (this will be slow — ~minutes per epoch)")
print("Tip: reduce x_train[:1000] to test quickly\n")

t0 = time.time()
train(x_train, y_train_one_hot, conv1_w, conv2_w, dense_w, dense_b,
      epochs=5, batch_size=128, lr=0.001)
t1 = time.time()
print(f"\nTraining time: {t1 - t0:.1f}s\n")

evaluate(x_test, y_test_one_hot, conv1_w, conv2_w, dense_w, dense_b)
