# %% [markdown]
# # 9. Mixture Models and EM

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
#%matplotlib inline

from prml.clustering import KMeans
from prml.rv import (
    MultivariateGaussianMixture,
    BernoulliMixture
)

np.random.seed(2222)

# %% [markdown]
# ## 9.1 K-means Clustering

# %%
# training data
x1 = np.random.normal(size=(100, 2))
x1 += np.array([-5, -5])
x2 = np.random.normal(size=(100, 2))
x2 += np.array([5, -5])
x3 = np.random.normal(size=(100, 2))
x3 += np.array([0, 5])
x_train = np.vstack((x1, x2, x3))

x0, x1 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
x = np.array([x0, x1]).reshape(2, -1).T

# %%
kmeans = KMeans(n_clusters=3)
kmeans.fit(x_train)
cluster = kmeans.predict(x_train)
plt.scatter(x_train[:, 0], x_train[:, 1], c=cluster)
plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], s=200, marker='X', lw=2, c=['purple', 'cyan', 'yellow'], edgecolor="white")
plt.contourf(x0, x1, kmeans.predict(x).reshape(100, 100), alpha=0.1)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# %% [markdown]
# ## 9.2 Mixture of Gaussians

# %%
gmm = MultivariateGaussianMixture(n_components=3)
gmm.fit(x_train)
p = gmm.classify_proba(x_train)

plt.scatter(x_train[:, 0], x_train[:, 1], c=p)
plt.scatter(gmm.mu[:, 0], gmm.mu[:, 1], s=200, marker='X', lw=2, c=['red', 'green', 'blue'], edgecolor="white")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.gca().set_aspect("equal")
plt.show()

# %% [markdown]
# ### 9.3.3 Mixtures of Bernoulli distributions

# %%
x, y = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
x_train = []
for i in [0, 1, 2, 3, 4]:
    x_train.append(x[np.random.choice(np.where(y == str(i))[0], 200)])
x_train = np.concatenate(x_train, axis=0)
x_train = (x_train > 127).astype(np.float)

# %%
bmm = BernoulliMixture(n_components=5)
bmm.fit(x_train)

plt.figure(figsize=(20, 5))
for i, mean in enumerate(bmm.mu):
    plt.subplot(1, 5, i + 1)
    plt.imshow(mean.reshape(28, 28), cmap="gray")
    plt.axis('off')
plt.show()

# %%



