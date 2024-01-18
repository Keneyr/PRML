import numpy as np


class PCA(object):

    def __init__(self, n_components):
        """
        construct principal component analysis

        Parameters
        ----------
        n_components : int
            number of components
        """
        assert isinstance(n_components, int)
        self.n_components = n_components
    """
    @keneyr: fit the data, get the parameters of this model. 
        Method1: maximum log likelyhood function, the the close-form solution
        Method2: EM algorithm, iterative to get the solution
    """
    def fit(self, X, method="eigen", iter_max=100):
        """
        maximum likelihood estimate of pca parameters
        x ~ \int_z N(x|Wz+mu,sigma^2)N(z|0,I)dz

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input data
        method : str
            method to estimate the parameters
            ["eigen", "em"]
        iter_max : int
            maximum number of iterations for em algorithm

        Attributes
        ----------
        mean : (n_features,) ndarray
            sample mean of the data
        W : (n_features, n_components) ndarray
            projection matrix
        var : float
            variance of observation noise
        C : (n_features, n_features) ndarray
            variance of the marginal dist N(x|mean,C)
        Cinv : (n_features, n_features) ndarray
            precision of the marginal dist N(x|mean, C)
        """
        method_list = ["eigen", "em"]
        if method not in method_list:
            print("availabel methods are {}".format(method_list))
        self.mean = np.mean(X, axis=0)
        getattr(self, method)(X - self.mean, iter_max)

    """
    @keneyr: maximum log likelyhood function, get the eigen values, eigen vectors, etc.
    p(x) = N(x | mu, C)
    """
    def eigen(self, X, *arg):
        sample_size, n_features = X.shape
        # N > D
        if sample_size >= n_features:
            # the covariance matrix S
            cov = np.cov(X, rowvar=False)
            # eigen values, eigen vectors of S
            values, vectors = np.linalg.eigh(cov)
            # the remained components except the components we want
            index = n_features - self.n_components
        else:
            cov = np.cov(X)
            values, vectors = np.linalg.eigh(cov)
            """
            @keneyr: N < D: the number of observed data N is less than the dimention M of data x_n
                the new eigen vectors:
                u_i = 1 / sqrt(N \lambda_i) X^T v_i
            """
            vectors = (X.T @ vectors) / np.sqrt(sample_size * values)
            index = sample_size - self.n_components
        self.I = np.eye(self.n_components)
        """
        @keneyr: index == 0 means there is no reduction of dimensionality, means C = S
        from maximum loglikelihood function, we get \sigma_ML
        """
        if index == 0:
            self.var = 0
        else:
            self.var = np.mean(values[:index])
        # from maximum loglikelihood function, we get W_ML
        self.W = vectors[:, index:].dot(np.sqrt(np.diag(values[index:]) - self.var * self.I))
        # get the matrix M, useful when calculating p(x) and p(z|x)
        self.__M = self.W.T @ self.W + self.var * self.I
        self.C = self.W @ self.W.T + self.var * np.eye(n_features)
        # get the inverse of C, useful for calcu p(x)
        if index == 0:
            self.Cinv = np.linalg.inv(self.C)
        else:
            self.Cinv = np.eye(n_features) / np.sqrt(self.var) - self.W @ np.linalg.inv(self.__M) @ self.W.T / self.var

    """
    @keneyr: EM Algorithm, computationally much more efficient than conventional PCA in spaces of high dimensionality
    """
    def em(self, X, iter_max):
        self.I = np.eye(self.n_components)
        # choose initial value for W, \sigma^2 parameters.
        self.W = np.eye(np.size(X, 1), self.n_components)
        self.var = 1.
        for i in range(iter_max):
            W = np.copy(self.W)
            stats = self._expectation(X)
            self._maximization(X, *stats)
            if np.allclose(W, self.W):
                break
        self.C = self.W @ self.W.T + self.var * np.eye(np.size(X, 1))
        self.Cinv = np.linalg.inv(self.C)
    
    """
    @keneyr: E.Step, get the expectation function Q of complete-data log likelihood function
    Q = ..
    E[z_n] = ..
    E[z_nz_n^T] = ..
    """
    def _expectation(self, X):
        self.__M = self.W.T @ self.W + self.var * self.I
        Minv = np.linalg.inv(self.__M)
        Ez = X @ self.W @ Minv
        Ezz = self.var * Minv + Ez[:, :, None] * Ez[:, None, :]
        return Ez, Ezz
    
    """
    @keneyr: M.Step, maximize Q function with respect to W, \sigma^2
    W_new = ..
    \sigma^2_new = ..
    """
    def _maximization(self, X, Ez, Ezz):
        self.W = X.T @ Ez @ np.linalg.inv(np.sum(Ezz, axis=0))
        self.var = np.mean(
            np.mean(X ** 2, axis=1)
            - 2 * np.mean(Ez @ self.W.T * X, axis=1)
            + np.trace((Ezz @ self.W.T @ self.W).T) / np.size(X, 1))
    
    # posterior distribution: p(z|x), reduce dimension
    def transform(self, X):
        """
        project input data into latent space
        p(Z|X) = N(Z|(X-mu)WMinv, sigma^-2M)

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input data

        Returns
        -------
        Z : (sample_size, n_components) ndarray
            projected input data
        """
        return np.linalg.solve(self.__M, ((X - self.mean) @ self.W).T).T

    """
    @keneyr: whiten the data to give it zero mean and unit covariance. x = Py, y = P^T x
    """
    def fit_transform(self, X, method="eigen"):
        """
        perform pca and whiten the input data

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input data

        Returns
        -------
        Z : (sample_size, n_components) ndarray
            projected input data
        """
        self.fit(X, method)
        return self.transform(X)

    # p(x) = N(x | mu, C), Gaussian probability
    def proba(self, X):
        """
        the marginal distribution of the observed variable

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input data

        Returns
        -------
        p : (sample_size,) ndarray
            value of the marginal distribution
        """
        d = X - self.mean
        return (
            np.exp(-0.5 * np.sum(d @ self.Cinv * d, axis=-1))
            / np.sqrt(np.linalg.det(self.C))
            / np.power(2 * np.pi, 0.5 * np.size(X, 1)))
