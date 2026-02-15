from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn


# Load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
test_data = datasets.MNIST("data", train=False, transform=transform)
print("-----------------------------")
print("1")
print(test_data) # all data
print("-----------------------------")
print("2")
print(test_data[0]) # -> tensor, int
print("-----------------------------")
print("3")
print(test_data[0][0]) # -> tensor
print("-----------------------------")
print("4")
print(test_data[0][0][0]) # -> first (and only) chanel
print("-----------------------------")
print("5")
print(test_data[0][0][0][0].shape) 
print(test_data[0][0][0][0]) # first row
print("-----------------------------")
print("6")
print(test_data[0][0][0][0][0].shape) 
print(test_data[0][0][0][0][0]) # first px

test_data_list = []

for i in range(len(test_data)):
    img, label = test_data[i]
    test_data_list.append((label, img[0]))

kernels_32 = np.random.randn(32, 1, 3, 3) * 0.1 
print("-----------------------------")
print("7")
print(kernels_32)

conv1 = nn.Conv2d(1, 32, 3)
print("-----------------------------")
print("8")
print(conv1)
print(conv1.weight[0])



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

conv2d32_out = conv2d_forward(test_data_list)
print("-----------------------------")
print("9")
print(conv2d32_out)


# iterate over all images
# first image is your input
#iterate over all filters
#convolve all filter
#output from one filter is the input to the next
    


