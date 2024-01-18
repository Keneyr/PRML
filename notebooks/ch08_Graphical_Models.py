# %% [markdown]
# # 8. Graphical Models

# %%
#%matplotlib inline
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from prml import bayesnet as bn


np.random.seed(1234)

# %%
b = bn.discrete([0.1, 0.9])
f = bn.discrete([0.1, 0.9])

g = bn.discrete([[[0.9, 0.8], [0.8, 0.2]], [[0.1, 0.2], [0.2, 0.8]]], b, f)

# %%
print("b:", b)
print("f:", f)
print("g:", g)

# %%
g.observe(0)

# %%
print("b:", b)
print("f:", f)
print("g:", g)

# %%
b.observe(0)

# %%
print("b:", b)
print("f:", f)
print("g:", g)

# %% [markdown]
# ### 8.3.3 Illustration: Image de-noising

# %%
x, _ = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
x = x[0]
binarized_img = (x > 127).astype(np.int).reshape(28, 28)
plt.imshow(binarized_img, cmap="gray")

# %%
indices = np.random.choice(binarized_img.size, size=int(binarized_img.size * 0.1), replace=False)
noisy_img = np.copy(binarized_img)
noisy_img.ravel()[indices] = 1 - noisy_img.ravel()[indices]
plt.imshow(noisy_img, cmap="gray")

# %%
markov_random_field = np.array([
        [[bn.discrete([0.5, 0.5], name=f"p(z_({i},{j}))") for j in range(28)] for i in range(28)], 
        [[bn.DiscreteVariable(2) for _ in range(28)] for _ in range(28)]])
a = 0.9
b = 0.9
pa = [[a, 1 - a], [1 - a, a]]
pb = [[b, 1 - b], [1 - b, b]]
for i, j in itertools.product(range(28), range(28)):
    bn.discrete(pb, markov_random_field[0, i, j], out=markov_random_field[1, i, j], name=f"p(x_({i},{j})|z_({i},{j}))")
    if i != 27:
        bn.discrete(pa, out=[markov_random_field[0, i, j], markov_random_field[0, i + 1, j]], name=f"p(z_({i},{j}), z_({i+1},{j}))")
    if j != 27:
        bn.discrete(pa, out=[markov_random_field[0, i, j], markov_random_field[0, i, j + 1]], name=f"p(z_({i},{j}), z_({i},{j+1}))")
    markov_random_field[1, i, j].observe(noisy_img[i, j], proprange=0)

# %%
for _ in range(10000):
    i, j = np.random.choice(28, 2)
    markov_random_field[1, i, j].send_message(proprange=3)
restored_img = np.zeros_like(noisy_img)
for i, j in itertools.product(range(28), range(28)):
    restored_img[i, j] = np.argmax(markov_random_field[0, i, j].proba)
plt.imshow(restored_img, cmap="gray")

# %%



