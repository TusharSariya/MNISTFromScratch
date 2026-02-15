from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
from pyinstrument import Profiler
import keras



# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#add a chanel dimention, 1 chanel
x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

test_data_list = []

for i in range(len(x_test)):
    test_data_list.append((y_test[i], x_test[i]))

kernels_32 = np.random.randn(32, 1, 3, 3) * 0.1 



#literally so slow, the heat death of the universe will come before this
def conv2d_forward(input):
    conv2d32_out = []
    counter = 1
    for label, image in input: #iterate over all image
        print(counter)
        counter = counter + 1
        new = np.zeros((32,26, 26)) # convolutions for all 32 filters for each image
        for k in range(32): #iterate over all filters
            #weight = conv1.weight[k]
            weight = kernels_32[k]
            for y in range(1,27): #convolve over y
                for x in range(1,27): #convolve over x
                    total = 0
                    for j in range(-1,2): # y range
                        for i in range(-1,2): # x range
                            total += weight[0][j+1][i+1]*image[y+j][x+i]
                    new[k][y-1][x-1] = total
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
    


