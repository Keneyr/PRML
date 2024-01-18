import numpy as np

from prml.linear._regression import Regression


class RidgeRegression(Regression):
    """Ridge regression model.
    Tikhonov regularization:
    w* = argmin |t - X @ w|_2^2 + alpha * |w|_2^2
    """

    def __init__(self, alpha: float = 1.):
        """Initialize ridge linear regression model.

        Parameters
        ----------
        alpha : float, optional
            Coefficient of the prior term, by default 1.
        """
        self.alpha = alpha

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """Maximum a posteriori estimation of parameter.

        Parameters
        ----------
        x_train : np.ndarray
            training data independent variable (N, D)
        y_train : np.ndarray
            training data dependent variable (N,)
        """
        # np.eye: Return a 2-D array with ones on the diagonal and zeros elsewhere.
        eye = np.eye(np.size(x_train, 1))
        """
        np.linalg.solve(A,b): Solve a linear matrix equation, or system of linear scalar equations. Ax = b
        https://en.wikipedia.org/wiki/Ridge_regression#Tikhonov_regularization
        pseudoinverse
        """
        self.w = np.linalg.solve(
            self.alpha * eye + x_train.T @ x_train,
            x_train.T @ y_train,
        )

    def predict(self, x: np.ndarray):
        """Return prediction.

        Parameters
        ----------
        x : np.ndarray
            samples to predict their output (N, D)

        Returns
        -------
        np.ndarray
            prediction of each input (N,)
        """
        return x @ self.w
