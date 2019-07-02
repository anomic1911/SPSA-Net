import numpy as np
np.random.seed(786)


class Net:
    def __init__(self, input_dim, latent_dim, output_dim,  a=1, c=10**(-4), alpha=0.8, gamma=0.2):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        # initialize layer weights
        self.weights = []
        self.weights.append(np.random.randn(self.input_dim, self.latent_dim))
        self.weights.append(np.random.randn(self.latent_dim, self.output_dim))

    def train(self, inputs, targets, t_max=200):
        for l in range(len(self.weights)):
            w_p = np.copy(self.weights)
            w_m = np.copy(self.weights)
            for t in range(1, t_max):
                a_t = self.a / t**self.alpha
                b_t = self.c / t**self.gamma
                delta = np.random.binomial(1, p=0.5, size=self.weights[l].shape) * 2. - 1
                # perturb weights in plus directions
                w_p[l] = w_p[l] + b_t * delta
                # compute predictions according to W_p and then compute loss using perturbed weight
                preds = self.forward(inputs, w_p)
                loss_p = self.loss(preds, targets)
                # perturb weights in minus directions
                w_m[l] = w_m[l] - b_t * delta
                # compute predictions according to W_m and then compute loss using perturbed weight
                preds = self.forward(inputs, w_m)
                loss_m = self.loss(preds, targets)
                # Compute approximation of the gradient
                g_hat = (loss_p - loss_m) / (2 * b_t * delta)
                # print("Logging Gradient descent update", np.max(a_t * g_hat))
                self.weights[l] = self.weights[l] - a_t * g_hat
            # print(l,"Logging weights:", self.weights[l], '\n', self.weights[l].shape)
        return self.forward(inputs, self.weights)

    @staticmethod
    def forward(inputs, w):
        out = np.dot(w[0].T, inputs)
        out = np.tanh(out)
        out = np.dot(w[1].T, out)
        return out

    @staticmethod
    def loss(preds, targets):
        return np.sum((preds-targets)**2)
