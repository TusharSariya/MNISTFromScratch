from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
from pyinstrument import Profiler
import keras
import math
import time

np.set_printoptions(threshold=np.inf)     

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#add a chanel dimention, 1 chanel
x_train = np.expand_dims(x_train, 1)  # (60000, 1, 28, 28)
x_test = np.expand_dims(x_test, 1)

# 32X1X3X3
kernels_32 = np.random.randn(32, 1, 3, 3) * 0.1 
# 9X32
kernels_32 = kernels_32.reshape(32,9).T

#wtf 32 chanels?
kernels_64 = np.random.randn(64, 32, 3, 3) * 0.1 
# 288X64
kernels_64 = kernels_64.reshape(64,288).T

# val = W*x+B
#1600X10 values magic for weight
dense_w = np.random.randn(1600, 10) * np.sqrt(2.0 / 1600)                        
dense_b = np.zeros(10)

#kernel
# first -> 9X32
#second -> 288X64
#x_test
# first -> 28X28
#second -> 32X13X13


def conv2d_forward(y_test,x_test,kernel):
    x_flattened = []
    #in the first iteration for Mat Mul the inner dimentions are 9 (3X3) because we do one filter
    #in the second iteration we have 32 chanels on input so it is 288 (32X3X3)
    print(x_test[0].shape)
    s = x_test[0].strides
    print(s)
    for i in range(len(x_test)):
        s = x_test[i].strides

        #to make convolutions efficient MatMuls we want to do a magic (memeory) trick with pointers
        # first -> 1X28X28 -> 26X26X1X3X3 stride 1 so there is overlap
        #second -> 32X13X13 -> 11X11X32X3X3
        #had to use AI to help with strides magic
        flattened1 = np.lib.stride_tricks.as_strided(                                   
            x_test[i],
            (x_test[i].shape[1]-2, x_test[i].shape[2]-2, x_test[i].shape[0], 3, 3),
            (s[1], s[2], s[0], s[1], s[2])
        )


        #we want to flatten it completely for MatMul
        # first -> 26X26X3X3 -> 676X9
        #second -> 11X11X32X3X3 -> 121X288
        #had to use AI to help with strides magic
        x_flattened.append(flattened1.reshape(
            (x_test[i].shape[1]-2) * (x_test[i].shape[2]-2),
            x_test[i].shape[0] * 9)
        )


    #kernels
    # first -> 9X32
    #second -> 288X64
    
    conv2d32_out = []
    output_dim = int(math.sqrt(len(x_flattened[1])))
    for i in range(len(x_flattened)):
        label = y_test[i]
        image = x_flattened[i]
        new = (image @ kernel).T.reshape(kernel.shape[1],output_dim,output_dim)
        conv2d32_out.append(new)

    return conv2d32_out

def relu(input):
    return [np.maximum(0,img) for img in input]

def maxpooling2d(input):
    output = []
    for label, image in input:
        this_image = []
        for kernel in image:
            this_kernel = []
            for j in range(0,len(kernel),2):
                rows = []
                for i in range(0,len(kernel[0]),2):
                    maximum = max(kernel[j][i],kernel[j+1][i],kernel[j][i+1],kernel[j+1][i+1])
                    rows.append(maximum)
                this_kernel.append(rows)
            this_image.append(this_kernel)
        output.append((label,this_image))
    return output


# list of tuples of label and kernels
# kernels 32X26X26 32 kernels 26X26 dim
def maxpooling2dbutfast(input):
      output = []
      for img in input:
          s = img.strides
          view = np.lib.stride_tricks.as_strided(img,
              (img.shape[0], img.shape[1]//2, img.shape[2]//2, 2, 2),
              (s[0], s[1]*2, s[2]*2, s[1], s[2]))
          pooled = np.max(view, axis=(3, 4))
          output.append(pooled)
      return output

def flatten(input):
    #takes a multiudimentional array and returns a flattened version of it
    return input.reshape(-1)

# dropout
def dropout(input, rate=0.5, training=True):                                     
    if not training:                                                                                  
        return input   
    #creates random numbers between zero and one for the size of input, filter by rate
    mask = np.random.rand(*input.shape) > rate
    return input * mask / (1 - rate)

def dense(input, weights, bias):
    # (1600,) @ (1600, 10) + (10,) -> (10,)
    return input @ weights + bias

def softmax(input):
    # subtract max for numerical stability (prevents exp overflow)
    #substract max val
    #then e^x for every value in the array, e^x because its derrevative and integral are the same
    #then normalize again
    e = np.exp(input - np.max(input))
    return e / e.sum()



t0 = time.time()
conv2d32_out = conv2d_forward(y_test,x_test,kernels_32)
t1 = time.time()
print(f"conv2d_forward: {t1 - t0:.3f}s")

normalized = relu(conv2d32_out)
t2 = time.time()
print(f"relu:           {t2 - t1:.3f}s")

pooled = maxpooling2dbutfast(normalized)
t3 = time.time()
print(f"maxpooling2d:   {t3 - t2:.3f}s")

t4 = time.time()
conv2d64_out = conv2d_forward(y_test,pooled,kernels_64)
t5 = time.time()
print(f"conv2d_forward: {t5 - t4:.3f}s")

normalized = relu(conv2d64_out)
t6 = time.time()
print(f"relu:           {t6 - t5:.3f}s")

pooled = maxpooling2dbutfast(normalized)
t7 = time.time()
print(f"maxpooling2d:   {t7 - t6:.3f}s")

# (64, 5, 5) -> (1600,) per image
flattened = [flatten(img) for img in pooled]
t8 = time.time()
print(f"flatten:        {t8 - t7:.3f}s")

# randomly zero out 50% of values, scale survivors
dropped = [dropout(img) for img in flattened]
t9 = time.time()
print(f"dropout:        {t9 - t8:.3f}s")

# (1600,) -> (10,) raw logits per image
logits = [dense(img, dense_w, dense_b) for img in dropped]
t10 = time.time()
print(f"dense:          {t10 - t9:.3f}s")

# (10,) -> (10,) probabilities per image
predictions = [softmax(img) for img in logits]
t11 = time.time()
print(f"softmax:        {t11 - t10:.3f}s")

print(f"total:          {t11 - t0:.3f}s")

# check output for first image
print(f"\nprediction: {np.argmax(predictions[0])}, actual: {y_test[0]}")
print(f"probabilities: {predictions[0]}")

