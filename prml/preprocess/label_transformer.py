import numpy as np


class LabelTransformer(object):
    """
    Label encoder decoder

    Attributes
    ----------
    n_classes : int
        number of classes, K
    """

    def __init__(self, n_classes:int=None):
        self.n_classes = n_classes

    @property
    def n_classes(self):
        return self.__n_classes

    @n_classes.setter
    def n_classes(self, K):
        self.__n_classes = K
        self.__encoder = None if K is None else np.eye(K)

    @property
    def encoder(self):
        return self.__encoder

    def encode(self, class_indices:np.ndarray):
        """
        encode class index into one-of-k code

        Parameters
        ----------
        class_indices : (N,) np.ndarray
            non-negative class index
            elements must be integer in [0, n_classes)

        Returns
        -------
        (N, K) np.ndarray
            one-of-k encoding of input
        """
        if self.n_classes is None:
            """
            np.max(): find the maximum value along a specified axis or in a given array, e.g.
                arr = np.array([1, 3, 5, 2, 4])
                max_value = np.max(arr)
                print(max_value)  # Output: 5
            """
            self.n_classes = np.max(class_indices) + 1

        return self.encoder[class_indices]

    def decode(self, onehot:np.ndarray):
        """
        decode one-of-k code into class index

        Parameters
        ----------
        onehot : (N, K) np.ndarray
            one-of-k code

        Returns
        -------
        (N,) np.ndarray
            class index
        """

        """
        np.argmax(): returns the indices of the maximum values along a specified axis in an array, e.g.
            arr = np.array([1, 3, 5, 2, 4])
            index_of_max_value = np.argmax(arr)
            print(index_of_max_value)  # Output: 2 (index of the maximum value '5')
        """
        return np.argmax(onehot, axis=1)
