import numpy as np


np.random.seed(1) #seed random

#input
# 4X3
streetlights = np.array([
    [1,0,1],
    [0,1,1],
    [0,0,1],
    [1,1,1]]
)

#output
#1X4 -> 4X1
walkstop = np.array([[1,1,0,1]]).T

#hidden layer
#3X4
weights_0 = 2*np.random.random((3,4))-1 # uniform distrobution of positive/negative 1

#samesize as output
#4X1
weights_1 = 2*np.random.random((4,1)) - 1 # uniform distrobution of positive/negative 1

#any shape, on hidden layers
def relu(arr):
    return (arr > 0) * arr

#only final layer
def error(pred):
    return (pred - walkstop) ** 2

#only applies to final layer
def derrevative(prev,pred,goal):
    return prev.dot((pred - goal))

def weight(weight,derrevative, alpha=0.1):
    return weight - (alpha * derrevative)


#dot product is row first multiplied by column second (kind of) and added together to get a prediciton
#effectively we multiply the first set of weights with  with the first set of lights
#but it must have three rows to dot product effectively
#it can have any number of columns
#hidden layers are 'magic'

err = np.ones((4,1)) * 999
idx = 0

while err.mean() > 0.001 and idx < 1000:
    idx += 1
    #--------------------------------------------------------
    #forwards pass

    # streetlights dot weights_0 = intermediary
    # 4X3 dot 3X4 = 4X4

    #dropout_mask = np.random.random(streetlights.shape) > 0.5                                                                                                               
    #streetlights_dropout = streetlights * dropout_mask  

    output_0 = streetlights.dot(weights_0)


    #use relu to make activations non linear
    relu_output_0 = relu(output_0)

    #intermediary dot weights_1 = prediction
    # 4x4 dot 4X1 = 4x1
    prediction = relu_output_0.dot(weights_1)

    #-------------------------------------------------------
    #backwards pass

    #error
    err = error(prediction)
    print(f"iter {idx}, error: {err.mean():.4f}")


    #final layer
    #as the final layer has a known output we can compare results to it
    derr_1 = derrevative(relu_output_0.T, prediction,walkstop)
    weights_1_new = weight(weights_1,derr_1)

    #hidden layers
    #previous layers do not have ground truth, so the eror delta must be propogated
    #delta is how wrong you are
    #4X1
    delta_1 = prediction - walkstop
    # 4X1 dot 4X1.T -> 4X1 dot 1X4 -> 4X4
    #find the error that each neuron in the hidden layer
    delta_0 = delta_1.dot(weights_1.T)
    #relu again
    delta_0 *= (output_0 > 0)
    #4X3.T -> 3X4 dot 4X4 = 3X4
    #gradient for weights_0
    derr_0 = streetlights.T.dot(delta_0)

    weights_0_new = weight(weights_0,derr_0)
    weights_1 = weights_1_new
    weights_0 = weights_0_new
