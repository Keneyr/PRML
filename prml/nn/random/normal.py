import numpy as np
from scipy.stats import truncnorm
from prml.nn.array.array import asarray


def normal(mean, std, size):
    return asarray(np.random.normal(mean, std, size))

"""
numpy.random.truncnorm(a, b, loc=0, scale=1, size=None) : generate random samples from a truncated normal distribution
rvs():  generic random variates sampling function that is often used to generate random samples from a specified distribution. 
The distribution can be specified using various parameters depending on the distribution type.
"""
def truncnormal(min, max, scale, size):
    return asarray(truncnorm(a=min, b=max, scale=scale).rvs(size))
