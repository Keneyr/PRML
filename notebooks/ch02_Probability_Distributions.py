# %% [markdown]
# # 2. Probability Distributions

# %%
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

from prml.rv import (
    Bernoulli,
    Beta,
    Categorical,
    Dirichlet,
    Gamma,
    Gaussian,
    MultivariateGaussian,
    MultivariateGaussianMixture,
    StudentsT,
    Uniform
)

np.random.seed(1234)

# %% [markdown]
# ## 2.1 Binary Variables

# %%
model = Bernoulli()
model.fit(np.array([0., 1., 1., 1.]))
print(model)

# %% [markdown]
# ### 2.1.1 The beta distributions

# %%
x = np.linspace(0, 1, 100)
for i, [a, b] in enumerate([[0.1, 0.1], [1, 1], [2, 3], [8, 4]]):
    plt.subplot(2, 2, i + 1)
    beta = Beta(a, b)
    plt.xlim(0, 1)
    plt.ylim(0, 3)
    plt.plot(x, beta.pdf(x))
    plt.annotate("a={}".format(a), (0.1, 2.5))
    plt.annotate("b={}".format(b), (0.1, 2.1))
plt.show()

# %%
beta = Beta(2, 2)
plt.subplot(2, 1, 1)
plt.xlim(0, 1)
plt.ylim(0, 2)
plt.plot(x, beta.pdf(x))
plt.annotate("prior", (0.1, 1.5))

model = Bernoulli(mu=beta)
model.fit(np.array([1]))
plt.subplot(2, 1, 2)
plt.xlim(0, 1)
plt.ylim(0, 2)
plt.plot(x, model.mu.pdf(x))
plt.annotate("posterior", (0.1, 1.5))

plt.show()

# %%
print("Maximum likelihood estimation")
model = Bernoulli()
model.fit(np.array([1]))
print("{} out of 10000 is 1".format(model.draw(10000).sum()))

print("Bayesian estimation")
model = Bernoulli(mu=Beta(1, 1))
model.fit(np.array([1]))
print("{} out of 10000 is 1".format(model.draw(10000).sum()))

# %% [markdown]
# ## 2.2 Multinomial Variables

# %%
model = Categorical()
# row vector, the first vector x = (0,1,0)
model.fit(np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]]))
print(model)

# %% [markdown]
# ### 2.2.1 The Dirichlet distribution

# %%
mu = Dirichlet(alpha=np.ones(3))
model = Categorical(mu=mu)
print(model)

model.fit(np.array([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]]))
print(model)

# %% [markdown]
# ## 2.3 The Gaussian Distribution

# %%
uniform = Uniform(low=0, high=1)
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.xlim(0, 1)
plt.ylim(0, 5)
plt.annotate("N=1", (0.1, 4.5))
plt.hist(uniform.draw(10000), bins=20, density=True)

plt.subplot(1, 3, 2)
plt.xlim(0, 1)
plt.ylim(0, 5)
plt.annotate("N=2", (0.1, 4.5))
plt.hist(0.5 * (uniform.draw(10000) + uniform.draw(10000)), bins=20, density=True)

plt.subplot(1, 3, 3)
plt.xlim(0, 1)
plt.ylim(0, 5)
sample = 0
for _ in range(10):
    sample = sample + uniform.draw(10000)
plt.annotate("N=10", (0.1, 4.5))
plt.hist(0.1 * sample, bins=20, density=True)

plt.show()

# %% [markdown]
# ### 2.3.4 Maximum Likelihood for the Gaussian

# %%
# Draw random samples from a normal (Gaussian) distribution, loc: mean, scale: Standard deviation
X = np.random.normal(loc=1., scale=2., size = (100,2))
gaussian = MultivariateGaussian()
gaussian.fit(X)
print(gaussian)

plt.scatter(X[:, 0], X[:, 1], facecolor="none", edgecolor="steelblue")

x, y = np.meshgrid(
    np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
"""

"""
p = gaussian.pdf(
    np.array([x, y]).reshape(2, -1).T).reshape(100, 100)
"""
Plot contour lines, p: The height values over which the contour is drawn. 
Color-mapping is controlled by cmap, norm, vmin, and vmax.
"""
plt.contour(x, y, p)
plt.show()

# %% [markdown]
# ### 2.3.6 Bayesian inference for the Gaussian

# %%
mu = Gaussian(0, 0.1)
model = Gaussian(mu, 0.1)
print(model)

x = np.linspace(-1, 1, 100)
plt.plot(x, model.mu.pdf(x), label="N=0")

model.fit(np.random.normal(loc=0.8, scale=0.1, size=1))
plt.plot(x, model.mu.pdf(x), label="N=1")

model.fit(np.random.normal(loc=0.8, scale=0.1, size=1))
plt.plot(x, model.mu.pdf(x), label="N=2")

model.fit(np.random.normal(loc=0.8, scale=0.1, size=8))
plt.plot(x, model.mu.pdf(x), label="N=10")

plt.xlim(-1, 1)
plt.ylim(0, 5)
plt.legend()
plt.show()

# %%
x = np.linspace(0, 2, 100)
for i, [a, b] in enumerate([[0.1, 0.1], [1, 1], [2, 3], [4, 6]]):
    plt.subplot(2, 2, i + 1)
    gamma = Gamma(a, b)
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.plot(x, gamma.pdf(x))
    plt.annotate("a={}".format(a), (1, 1.6))
    plt.annotate("b={}".format(b), (1, 1.3))
plt.show()

# %%
tau = Gamma(a=1, b=1)
model = Gaussian(mu=0, tau=tau)
print(model)

model.fit(np.random.normal(scale=1.414, size=100))
print(model)

# %% [markdown]
# ### 2.3.7 Student's t-distribution

# %%
X = np.random.normal(size=20)
X = np.concatenate([X, np.random.normal(loc=20., size=3)])
plt.hist(X.ravel(), bins=50, density=1., label="samples")

students_t = StudentsT()
gaussian = Gaussian()

gaussian.fit(X)
students_t.fit(X)

print(gaussian)
print(students_t)

x = np.linspace(-5, 25, 1000)
plt.plot(x, students_t.pdf(x), label="student's t", linewidth=2)
plt.plot(x, gaussian.pdf(x), label="gaussian", linewidth=2)
plt.legend()
plt.show()

# %% [markdown]
# ### 2.3.9 Mixture of Gaussians

# %%
x1 = np.random.normal(size=(100, 2))
x1 += np.array([-5, -5])
x2 = np.random.normal(size=(100, 2))
x2 += np.array([5, -5])
x3 = np.random.normal(size=(100, 2))
x3 += np.array([0, 5])
# np.vstack(): Stack arrays in sequence vertically (row wise)
X = np.vstack((x1, x2, x3))

model = MultivariateGaussianMixture(n_components=3)
model.fit(X)
print(model)

x_test, y_test = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
X_test = np.array([x_test, y_test]).reshape(2, -1).transpose()
probs = model.pdf(X_test)
Probs = probs.reshape(100, 100)
plt.scatter(X[:, 0], X[:, 1])
plt.contour(x_test, y_test, Probs)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()

# %%



