import numpy as np
import timeit

name = ["Alice", "Bob", "Cathy", "Doug"]
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

data = np.zeros(
    4, dtype={"names": ("name", "age", "weight"), "formats": ("U10", "i4", "f8")}
)

# print(data.dtype)

data["name"] = name
data["age"] = age
data["weight"] = weight

# print(data["name"])
# print(data[0])
# print(data[-1]['name'])
# print(data[data["age"] < 30]['name'])

"""Exploring Structured Array Creation"""
arr = np.dtype(
    {"names": ("name", "age", "weight"), "formats": ((np.str_, 10), int, np.float32)}
)
# print(arr)

arr = np.dtype([("name", "S10"), ("age", "i4"), ("weight", "f8")])
# print(arr)

arr = np.dtype("S10,i4,f8")
# print(arr)

"""More Advanced Compound Types"""
tp = np.dtype([("id", "i8"), ("mat", "f8", (3, 3))])
X = np.zeros(1, dtype=tp)
# print(X[0])
# print(X["mat"][0])

"""Record Arrays: Structured Arrays with a Twist"""
rec = data.view(np.recarray)
# print(rec.age)

print(timeit.timeit(lambda: data["age"]))
print(timeit.timeit(lambda: rec["age"]))
print(timeit.timeit(lambda: rec.age))