import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit as tictoc

plt.style.use("seaborn-v0_8-whitegrid")

"""Exploring Fancy Indexing"""
rng = np.random.default_rng(seed=1701)
x = rng.integers(100, size=10)

# Traditional access
# print([x[3], x[7], x[4]])

# Fancy indexing
index = [3, 7, 4]
# print(x[index])

# Shape of index array
index_2D = np.array([[3, 7], [4, 5]])
# print(x[index_2D])

# Multiple dimensions
X = np.arange(12).reshape((3, 4))
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])

# print(X[row, col])

# print(X)
# print(col)
# print(row[:, np.newaxis])
# print(X[row[:, np.newaxis], col])

"""Combined Indexing"""
# print(X)
# print(X[2, [2, 0, 1]])

# print(X)
# print(X[1:, [2, 0, 1]])

mask = np.array([True, False, True, False])
# print(X)
# print(X[row[:, np.newaxis], mask])

"""Example: Selecting Random Points"""
mean = [0, 0]
cov = [[1, 2], [2, 5]]
X = rng.multivariate_normal(mean, cov, 100)
# print(X.shape)

# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

index_random = np.random.choice(X.shape[0], 20, replace=False)
selection = X[index_random]
# print(selection.shape)

# plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
# plt.scatter(
#     selection[:, 0], selection[:, 1], facecolor="none", edgecolors="black", s=200
# )
# plt.show()

"""Modifying Values with Fancy Indexing"""
x = np.arange(10)
i = np.array([2, 1, 8, 4])

x[i] = 99
# print(x)

x[i] -= 10
# print(x)

x = np.zeros(10)
x[[0, 0]] = [4, 6]
# print(x)

i = [2, 3, 3, 4, 4, 4]
x[i] += 1
# print(x)

repeat_add = np.add.at(x, i, 1)
# print(x)

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
i = np.array([0, 5, 8])
reduced = np.add.reduceat(x, i)
# print(reduced)

"""searchsorted"""
arr = np.array([1, 2, 3, 4, 5])
v = np.array([0.5, 3.5, 10.0])

sorted_arr = np.searchsorted(arr, v)
# print(sorted_arr)

"""Example: Binning Data"""
x = rng.normal(size=100)

bins = np.linspace(-5, 5, 20)
counts = np.zeros_like(bins)

i = np.searchsorted(bins, x)

np.add.at(counts, i, 1)

# plt.plot(bins, counts, drawstyle='steps')
# plt.show()

# plt.hist(x, bins, histtype='step')
# plt.show()

# numpy_histogram_time = tictoc(lambda: np.histogram(x, bins), number=10000)
# print(f"NumPy histogram ({len(x)} points): {numpy_histogram_time} seconds")

# custom_histogram_time = tictoc(lambda: np.add.at(counts, np.searchsorted(bins, x), 1), number=10000)
# print(f"Custom histogram ({len(x)} points): {custom_histogram_time} seconds")

x = rng.normal(size=10000)

# numpy_histogram_time = tictoc(lambda: np.histogram(x, bins), number=10000)
# print(f"NumPy histogram ({len(x)} points): {numpy_histogram_time} seconds")

# custom_histogram_time = tictoc(lambda: np.add.at(counts, np.searchsorted(bins, x), 1), number=10000)
# print(f"Custom histogram ({len(x)} points): {custom_histogram_time} seconds")