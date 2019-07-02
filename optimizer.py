import numpy as np
class Optimizer:
    def __init__(self, weights,  a=0.01, c=0.01, alpha=1.0, gamma=0.4):
        self.weights = weights
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma

    def step(self, preds, targets, t_max=25):
        W_p = np.copy(self.weights)
        W_m = np.copy(self.weights)
        for l in range(len(self.weights)):
            for t in range(1,t_max):
                a_t = self.a / t**self.alpha
                c_t = self.c / t**self.gamma
                delta = np.random.binomial(1, p=0.5, size=(self.weights[l].shape)) * 2. - 1
                # perturb weights in both directions
                W_p[l] = self.weights[l] + c_t * delta
                W_m[l] = self.weights[l] - c_t * delta
                # print("sent W_p = ", W_p)
                loss_p = self.loss( preds, targets)
                loss_m = self.loss( preds, targets)
                g_hat = (loss_p - loss_m) / (2 * c_t * delta)
                self.weights[l] = self.weights[l] - a_t * g_hat
    def loss(self, preds, targets):
        return np.sum((preds-targets)**2)
