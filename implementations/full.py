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

# 32X1X3X3
kernels_32 = np.random.randn(32, 1, 3, 3) * 0.1 
# 9X32
kernels_32 = kernels_32.reshape(32,9).T

#wtf 32 chanels?
kernels_64 = np.random.randn(64, 32, 3, 3) * 0.1 
# 9X32X64
kernels_64 = kernels_64.reshape(64,32,9).T

#kernel
# first -> 9X32
#second -> 9X32X64
#x_test
# first -> 28X28
#second -> 32X13X13


def conv2d_forward(y_test,x_test,kernel):
    x_flattened = []
    #in the first iteration for Mat Mul the inner dimentions are 9 (3X3) because we do one filter
    #in the second iteration we have 32 chanels on input so it is 288 (32X3X3)
    print(x_test[0].shape)
    for i in range(len(x_test)):
        s = x_test[i].strides 

        #to make convolutions efficient MatMuls we want to do a magic (memeory) trick with pointers
        # first -> 28X28 -> 26X26X3X3 stride 1 so there is overlap
        #second -> 32X13X13 -> 11X11X3X3

        flattened1 = np.lib.stride_tricks.as_strided(
            x_test[i],
            (x_test[0].shape[0]-2,x_test[0].shape[1]-2,3,3),
            (s[0], s[1], s[0], s[1])
        )

        #we want to flatten it completely for MatMul
        # first -> 26X26X3X3 -> 676X9
        #second -> 11X11X3X3 -> 121X9
        x_flattened.append(flattened1.reshape(
            (x_test[0].shape[0]-2)*(x_test[0].shape[1]-2)
            ,9)
        )
    
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

        



t0 = time.time()
print(x_test[0].shape) # 28X28
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
print(pooled[0].shape) # 32X13X13
conv2d64_out = conv2d_forward(y_test,pooled,kernels_64)
t5 = time.time()
print(f"conv2d_forward: {t5 - t4:.3f}s")

print(f"total:          {t3 - t0:.3f}s")

