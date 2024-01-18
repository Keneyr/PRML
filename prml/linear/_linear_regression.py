import numpy as np

from prml.linear._regression import Regression


class LinearRegression(Regression):
    """Linear regression model.

    y = X @ w
    t ~ N(t|X @ w, var)
    """

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """Perform least squares fitting.

        Parameters
        ----------
        x_train : np.ndarray
            training independent variable (N, D)
        y_train : np.ndarray
            training dependent variable (N,)
        """
        
        """
        p.linalg.pinv: Compute the (Moore-Penrose) pseudo-inverse of a matrix
        """
        self.w = np.linalg.pinv(x_train) @ y_train
        """
        np.square(): Return the element-wise square of the input
        np.mean(): Returns the average of the array elements
        The average is taken over the flattened array by default, 
        otherwise over the specified axis
        """
        self.var = np.mean(np.square(x_train @ self.w - y_train))

    def predict(self, x: np.ndarray, return_std: bool = False):
        """Return prediction given input.

        Parameters
        ----------
        x : np.ndarray
            samples to predict their output (N, D)
        return_std : bool, optional
            returns standard deviation of each predition if True

        Returns
        -------
        y : np.ndarray
            prediction of each sample (N,)
        y_std : np.ndarray
            standard deviation of each predition (N,)
        """
        y = x @ self.w
        if return_std:
            # var: variance
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std
        return y
