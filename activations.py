import numpy as np
"""
contains relu, tanh and signmoid activations 
"""
def relu(x):
    return np.maximum(0, x)

def LeakyRelu(x):
    return np.maximum(0.01*x, x)

def relu_grad(x):
    return float( x> 0 )

def sigmoid(x):
    return 1.0/(1.0+np.exp(x))

def sigmoid_grad(x):
    return sigmoid(x) * (1-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - tanh(x)**2

def Softmax(x, axis = 0):
    t = np.exp(x)
    return t / t.sum(axis = axis, keepdims = True)



