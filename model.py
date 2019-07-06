import numpy as np
np.random.seed(786)


class Net:
    def __init__(self, input_dim, latent_dim, output_dim,  a=1e-6, b=1e-2, alpha=0.602, gamma=0.101):
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
        self.weights.append(np.random.randn(self.input_dim, self.latent_dim))
        self.weights.append(np.random.randn(self.latent_dim, self.latent_dim))
        self.weights.append(np.random.randn(self.latent_dim, self.latent_dim))
        self.weights.append(np.random.randn(self.latent_dim, self.output_dim))
        
        # self.biases.append(np.zeros([1,self.latent_dim]))
        # self.biases.append(np.zeros([1,self.latent_dim]))
        # self.biases.append(np.zeros([1,self.latent_dim]))
        # self.biases.append(np.zeros([1,self.output_dim]))


    def train(self, inputs, targets, t_max=100):
        for l in range(len(self.weights)):
            w_p = np.copy(self.weights)
            w_m = np.copy(self.weights)
            # b_p = self.biases.copy()
            # b_m = self.biases.copy()
            for t in range(1, t_max):
                saved_weights = np.copy(self.weights[l])
                preds = self.forward(inputs, self.weights)
                loss_old = self.loss(preds, targets)

                a_t = self.a / (1+t+50000)**self.alpha
                b_t = self.b / (1+t)**self.gamma
                delta = np.random.binomial(1, p=0.5, size=self.weights[l].shape) * 2. - 1
                delta2 = np.random.binomial(1, p=0.5, size=self.weights[l].shape) * 2. - 1
                # perturb weights in plus directions
                w_p[l] = w_p[l] + b_t * delta
                # b_p[l] = b_p[l] + b_t * delta2
                # compute predictions according to W_p and then compute loss using perturbed weight
                preds = self.forward(inputs, w_p)
                loss_p = self.loss(preds, targets)
                # perturb weights in minus directions
                w_m[l] = w_m[l] - b_t * delta
                # b_m[l] = b_m[l] - b_t * delta2
                # compute predictions according to W_m and then compute loss using perturbed weight
                preds = self.forward(inputs, w_m)
                loss_m = self.loss(preds, targets)
                # Compute approximation of the gradient
                g_hat = (loss_p - loss_m) / (2 * b_t * delta)
                # g_hat2 = (loss_p - loss_m) / (2 * b_t * delta2)
                # For updating weights
                breaking_bad = True
                clip_max = np.ones(self.weights[l].shape)*5
                clip_min = np.ones(self.weights[l].shape)*(-5)
                this_ak = (self.weights[l]*0 + 1)*a_t
                W_new = self.weights[l]
                while breaking_bad:
                    out_of_bounds = np.where ( np.logical_or ( \
                        W_new - this_ak*g_hat > clip_max, 
                        W_new - this_ak*g_hat < clip_min ) )[0]
                    W_new = self.weights[l] - this_ak*g_hat
                    if len ( out_of_bounds ) == 0:
                        self.weights[l] = self.weights[l] - this_ak*g_hat
                        breaking_bad = False
                    else:
                        this_ak[out_of_bounds] = this_ak[out_of_bounds]/2.
                preds = self.forward(inputs, self.weights)
                loss_new = self.loss (preds, targets)
                
                if np.abs ( loss_new - loss_old ) > 5:
                    self.weights[l] = saved_weights
                    continue
                else:
                    loss_old = loss_new

        return self.forward(inputs, self.weights)

    @staticmethod
    def forward(inputs, w):
        out = inputs.dot(w[0])
        out = np.tanh(out)
        out = out.dot(w[1]) 
        out = np.tanh(out)
        out = out.dot(w[2]) 
        out = np.tanh(out)
        out = out.dot(w[3]) 
        return out

    @staticmethod
    def loss(preds, targets):
        return np.sum((preds-targets)**2)

    def getPreds(self, inputs):
        out = inputs.dot(self.weights[0])
        out = np.tanh(out)
        out = out.dot(self.weights[1])
        out = np.tanh(out)
        out = out.dot(self.weights[2])
        out= np.tanh(out)
        out = out.dot(self.weights[3])
        return out