from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
from pyinstrument import Profiler
import keras

np.set_printoptions(threshold=np.inf)     



# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

test_data_list = []

# the purpose of im2col is not to reduce arithmetic complexity
# complexity stays the exact same
# but by organizing the memory to be a MatMul rather than a convolution you can tap into highly optimized hardware and libraries
# the CPU(SIMD) likes long contigious memory and BLAS have been custom tuned for this


for i in range(len(x_test)):
    #26 by 26 patches of 3x3, you need to repeat strides so it knows the offset for everything
    #sride_tricks is pointer redirection so it does not re-copy data
    s = x_test[i].strides 
    flattened1 = np.lib.stride_tricks.as_strided(x_test[i],(26,26,3,3),(s[0], s[1], s[0], s[1]))
    #this actually coppies memory as kernel is 3x3 and stride is 1 so there is overlap
    #reshape collapses from the right so 3 then 3 then 26 then 26
    flattened2 = flattened1.reshape(676,9)
    test_data_list.append((y_test[i], flattened2))


#32 filter of 1 channel 3x3 kernel
kernels_32 = np.random.randn(32, 1, 3, 3) * 0.1 
kernels_32 = kernels_32.reshape(32,9).T

#literally so slow, the heat death of the universe will come before this
def conv2d_forward(input):
    conv2d32_out = []
    counter = 1
    for label, image in input: #iterate over all image
        print(counter)
        counter = counter + 1
        #hyper optimized BLAS code with @
        #uses SIMD
        new = (image @ kernels_32).T.reshape(32,26,26)
        conv2d32_out.append((label,new))

    return conv2d32_out

profiler = Profiler()
profiler.start()

conv2d32_out = conv2d_forward(test_data_list)
profiler.stop()
profiler.print()


# iterate over all images
# first image is your input
#iterate over all filters
#convolve all filter
#output from one filter is the input to the next
    


