import numpy as np
from prml import nn

"""
@keneyr: here, neural networks have also been applied to unsupervised learning where they have been used for dimensionality reduction
"""
class Autoencoder(nn.Network):
    def __init__(self, *args):
        self.n_unit = len(args)
        super().__init__()
        for i in range(self.n_unit - 1):
            # weight
            self.parameter[f"w_encode{i}"] = nn.asarray(np.random.randn(args[i], args[i + 1]))
            # bias
            self.parameter[f"b_encode{i}"] = nn.asarray(np.zeros(args[i + 1]))
            # weight
            self.parameter[f"w_decode{i}"] = nn.asarray(np.random.randn(args[i + 1], args[i]))
            # bias
            self.parameter[f"b_decode{i}"] = nn.asarray(np.zeros(args[i]))


    def transform(self, x):
        h = x
        for i in range(self.n_unit - 1):
            h = nn.tanh(h @ self.parameter[f"w_encode{i}"] + self.parameter[f"b_encode{i}"])
        return h.value
    
    """
    four-layer autoassociativc network,
    F_1: D --> M, 4 -> 3 -> 2
    F_2: M --> D, 2 -> 3 -> 4
    """
    def forward(self, x):
        h = x
        for i in range(self.n_unit - 1):
            h = nn.tanh(h @ self.parameter[f"w_encode{i}"] + self.parameter[f"b_encode{i}"])
        for i in range(self.n_unit - 2, 0, -1):
            h = nn.tanh(h @ self.parameter[f"w_decode{i}"] + self.parameter[f"b_decode{i}"])
        x_ = h @ self.parameter["w_decode0"] + self.parameter["b_decode0"]
        return x_

    def fit(self, x, n_iter=100, learning_rate=1e-3):
        optimizer = nn.optimizer.Adam(self.parameter, learning_rate)
        for _ in range(n_iter):
            self.clear()
            x_ = self.forward(x)
            # by using maximum log likelihood function to get optimal parameters.
            log_likelihood = nn.Gaussian(x_, 1.).log_pdf(x)
            optimizer.maximize(log_likelihood)
