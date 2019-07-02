import numpy as np
from activations import tanh
# from optimizer import Optimizer
class Net:
    def __init__(self, input_dim, latent_dim, output_dim,  a=0.01, c=0.1, alpha=0.8, gamma=0.5):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, weights):

        out = np.dot(weights[0].T, inputs)
        # out = tanh(out)
        # out = np.dot(weights[1].T, out)
        return out
        

    def train(self, inputs, targets, num_epochs, t_max= 200): 
        # initialize layer weights
        weights = []
        np.random.seed(1)
        weights.append(np.random.randn(self.input_dim, self.output_dim))
        # weights.append(np.random.randn(self.latent_dim, self.output_dim))
        # self.W.append(np.random.randn(latent_dim, latent_dim))
        # self.W.append(np.random.randn(latent_dim, output_dim))

        for epoch in range(num_epochs):
            for l in range(len(weights)):
                W_p = np.copy(weights)
                W_m = np.copy(weights)
                for t in range(1,t_max):
                    a_t = self.a / t**self.alpha
                    c_t = self.c / t**self.gamma
                    delta = np.random.binomial(1, p=0.5, size=(weights[l].shape)) * 2. - 1
                    # perturb weights in plus directions
                    W_p[l] = W_p[l] + c_t * delta
                    # compute predictions according to W_p and then compute loss using perturbed weight
                    preds= self.forward(inputs, W_p)
                    loss_p = self.loss( preds, targets)
                    # perturb weights in minus directions
                    W_m[l] = W_m[l] - c_t * delta
                    # compute predictions according to W_m and then compute loss using perturbed weight
                    preds= self.forward(inputs,W_m)
                    loss_m = self.loss( preds, targets)
                    # Compute approximation of the gradient
                    g_hat = (loss_p - loss_m) / (2 * c_t * delta)
                    if loss_m - loss_p != 0 :
                        print("gothca", loss_m - loss_p, 0.01*np.mean(g_hat))
                    # print("a_t * g_hat mean is: ", np.mean(a_t * g_hat))
                    weights[l] = weights[l] - a_t * g_hat
            if(epoch % 1 == 0):
                preds = self.forward(inputs, weights)
                print("RMSE Loss is: ", np.sum((self.forward(inputs, weights) - targets)**2))
        return weights

    def loss(self, preds, targets):
        return np.sum((preds-targets)**2)
