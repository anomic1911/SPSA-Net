#!/usr/bin/env python
import numpy as np
from model import Net

# Hyperparams
input_dim = 15
latent_dim = 100
output_dim = 15

inputs = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                   [9.779], [6.182], [7.59], [2.167], [7.042],
                   [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

targets = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

num_epochs = 1000
model = Net(input_dim, latent_dim, output_dim)

for epoch in range(num_epochs):
    preds = model.train(inputs, targets)
    if epoch%10==0:
        print(epoch, "\tMSELoss is: ", np.sum((preds - targets)**2))
