import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

l = [3, 1, 4, 1, 5, 9, 2, 6]
# print(sorted(l))
l.sort()  # act in place

string = "python"
# print(sorted(string))

"""Fast Sorting in NumPy: np.sort and np.argsort"""
x = np.array([2, 1, 4, 3, 5])
# print(np.sort(x))

x = np.array([2, 1, 4, 3, 5])
x.sort()  # act in place
# print(x)

x = np.array([2, 1, 4, 3, 5])
i = np.argsort(x)
# print(i)

# print(x[i])  # fancy indexing

"""Sorting Along Rows or Columns"""
rng = np.random.default_rng(seed=42)
X = rng.integers(10, size=(4, 6))

# print(X, np.sort(X), sep="\n\n")  # default rows
# print(X, np.sort(X, axis=0), sep='\n\n')
# print(X, np.sort(X, axis=1), sep='\n\n')

"""Partial Sorts: Partitioning"""
x = np.array([7, 2, 3, 1, 6, 5, 4])

# print(x, np.partition(x, 3), sep='\n')
# print(X, np.partition(X, 2, axis=1), sep="\n\n")

"""Example: k-Nearest Neighbors"""
X = rng.random(size=(10, 2))

x = X[:, 0]
y = X[:, 1]

plt.scatter(x, y, s=200)
# plt.show()

# square distance for 2 points = (x2-x1)^2 + (y2-y1)^2
dist_sq = np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=-1)

####################################### simplify #
points1 = X[:, np.newaxis]
points2 = X[np.newaxis, :]

# print(points1[0, 0])  # [0.64386512 0.82276161]
# print(points2[0, 0])  # [0.64386512 0.82276161]

difference_point_1 = points1[0, 0] - points1[0, 0]
# print(difference_point_1)  # [0 0]

sqr_difference_point_1 = difference_point_1**2
# print(sqr_difference_point_1)  # [0 0]

dist_sq_point_1 = sqr_difference_point_1.sum(-1)
# print(dist_sq_point_1)
##################################################

# make sure by diagonal
diagonal = dist_sq.diagonal()
# print(diagonal) # zeros like points dist_itself

# sort indeces distance on rows(every row is a point)
nearest = np.argsort(dist_sq, axis=1)  # expencive work
# print(nearest[:, :3])

# partition(best fit work)
k = 2
nearest_partition = np.argpartition(dist_sq, k + 1, axis=1)
# print(nearest_partition[:, : k + 1])

# plot lines
plt.scatter(X[:, 0], X[:, 1], s=100)
for i in range(x.shape[0]):
    for j in nearest_partition[i, : k + 1]:
        # plot a line from X[i] to X[j]
        plt.plot(*zip(X[j], X[i]), color="black")
# plt.show()