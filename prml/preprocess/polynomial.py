import itertools
import functools
import numpy as np


class PolynomialFeature(object):
    """
    polynomial features

    transforms input array with polynomial features

    Example
    =======
    x =
    [[a, b],
    [c, d]]

    y = PolynomialFeatures(degree=2).transform(x)
    y =
    [[1, a, b, a^2, a * b, b^2],
    [1, c, d, c^2, c * d, d^2]]
    """

    def __init__(self, degree=2):
        """
        construct polynomial features

        Parameters
        ----------
        degree : int
            degree of polynomial
        """
        assert isinstance(degree, int)
        self.degree = degree

    """
    return the basis function values of x, e.g.:
    x = 2, the basis function values are 2^0, 2^1, 2^2, ... , 2^11
    """
    def transform(self, x):
        """
        transforms input array with polynomial features

        Parameters
        ----------
        x : (sample_size, n) ndarray
            input array

        Returns
        -------
        output : (sample_size, 1 + nC1 + ... + nCd) ndarray
            polynomial features
        """
        if x.ndim == 1:
            """
            creates an axis with length 1, e.g.:
                [1,2,3]'s shape is (3,)  with [:, None] operation, the shape becomes (3,1), which is
                [[1]
                [2]
                [3]]
            """
            x = x[:, None]
        x_t = x.transpose()
        features = [np.ones(len(x))] # no features is a list
        for degree in range(1, self.degree + 1):
            """
            list(combinations_with_replacement("ABC", 2))--> "AA", "AB", "AC", "BB", "BC", "CC"
            """
            for items in itertools.combinations_with_replacement(x_t, degree):
                """
                Apply function of two arguments cumulatively to the items of iterable, 
                from left to right, so as to reduce the iterable to a single value, e.g.:
                    functools.reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates ((((1+2)+3)+4)+5)
                """
                features.append(functools.reduce(lambda x, y: x * y, items))
        return np.asarray(features).transpose()
