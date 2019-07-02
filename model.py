import numpy as np
np.random.seed(786)

class Net:
    def __init__(self, input_dim, latent_dim, output_dim,  a=0.01, b=0.01, alpha=0.602, gamma=0.101):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.a = a
        self.b = b
        self.alpha = alpha
        self.gamma = gamma
        # initialize layer weights
        self.weights = []
        self.biases = []
        self.weights.append( np.random.randn(self.input_dim, self.latent_dim) )
        self.weights.append( np.random.randn(self.latent_dim, self.output_dim) )
        self.biases.append( np.random.randn(self.input_dim) )
        self.biases.append( np.random.randn(self.output_dim) )
        
    def forward(self, inputs, W, b):
        out = np.dot(W[0].T, inputs) + b[0]
        out = np.tanh(out)
        out = np.dot(W[1].T, out) + b[1]
        return out

    def train(self, inputs, targets, t_max= 100): 
        for l in range(len(self.weights)):
            W_p = np.copy(self.weights)
            W_m = np.copy(self.weights)
            b_p = np.copy(self.biases)
            b_m = np.copy(self.biases)
            for t in range(1,t_max):
                a_t = self.a / (1+t+500)**self.alpha
                b_t = self.b / (1+t)**self.gamma
                delta = np.random.binomial(1, p=0.5, size=(self.weights[l].shape)) * 2. - 1
                delta2 = np.random.binomial(1, p=0.5, size=(self.biases[l].shape)) * 2. - 1
                # perturb weights in plus directions
                W_p[l] = W_p[l] + b_t * delta
                b_p[l] = b_p[l] + b_t * delta2
                # compute predictions according to W_p and then compute loss using perturbed weight
                preds = self.forward(inputs, W_p, b_p)
                loss_p = self.loss( preds, targets)
                # perturb weights in minus directions
                W_m[l] = W_m[l] - b_t * delta
                b_m[l] = b_m[l] - b_t * delta2
                # compute predictions according to W_m and then compute loss using perturbed weight
                preds = self.forward(inputs, W_m, b_m)
                loss_m = self.loss( preds, targets)
                # Compute approximation of the gradient
                g_hat = (loss_p - loss_m) / (2 * b_t * delta)
                g_hat2 = (loss_p - loss_m) / (2 * b_t * delta2)
                breaking_bad = True
                clip_max = np.ones(self.weights[l].shape)*5
                clip_min = np.ones(self.weights[l].shape)*(-5)
                W_new = self.weights[l]
                while breaking_bad:
                    
                    if((W_new - a_t * g_hat < clip_max).all()  or (W_new - a_t * g_hat > clip_min).all()):
                        self.weights[l] = self.weights[l] - a_t * g_hat
                        breaking_bad = False
                    else:
                        a_t = a_t / 2
                    W_new = self.weights[l] - a_t * g_hat
                # print("Logging Gradient descent update", np.max(a_t * g_hat))                                    
                self.biases[l] = self.biases[l] - a_t * g_hat2
        # print("Logging weights:", self.weights)
        # print("Logging Gradient descent update", np.max(a_t * g_hat))
        return(self.forward(inputs, self.weights, self.biases))

    def loss(self, preds, targets):
        return np.sum((preds-targets)**2)
