"""Example: Counting Rainy DaysL"""
import numpy as np
import matplotlib.pyplot as plt
from vega_datasets import data

rainfall_mm = np.array(
    data.seattle_weather().set_index("date")["precipitation"]["2015"]
)

# plt.style.use("seaborn-v0_8-whitegrid")
# plt.hist(rainfall_mm, 40)
# plt.show()

"""Comparison Operators as Ufuncs"""
x = np.array([1, 2, 3, 4, 5])

# print(x < 3)  # = np.less(x, 3)
# print(x > 3)
# print(x <= 3)
# print(x >= 3)  # = np.greater_equal(x, 3)
# print(x == 3)
# print(x != 3)  # = np.not_equal(x, 3)

# print((2 * x) == (2**x))

rng = np.random.default_rng(seed=1701)
x = rng.integers(10, size=(3, 4))
# print(x, x < 6, sep='\n')

"""Working with Boolean Arrays"""
# Counting Entries
count_less_6 = np.count_nonzero(x < 6)
# print(count_less_6)

sum_less_6 = np.sum(x < 6)
# print(sum_less_6)

sum_less_6_row = np.sum(x < 6, axis=1)
# print(sum_less_6_row)

any_greater_8 = np.any(x > 8)
# print(any_greater_8)

all_equal_6 = np.all(x == 6)
# print(all_equal_6)

all_less_8_row = np.all(x < 8, axis=1)
# print(all_less_8_row)

# Boolean Operators
rainfall_insight_1 = np.sum((rainfall_mm > 10) & (rainfall_mm < 20))
rainfall_insight_1 = np.sum(~((rainfall_mm <= 10) | (rainfall_mm >= 20)))
# print(rainfall_insight_1)

# insights
# print("Number days without rain:  ", np.sum(rainfall_mm == 0))
# print("Number days with rain:     ", np.sum(rainfall_mm != 0))
# print("Days with more than 10 mm: ", np.sum(rainfall_mm > 10))
# print("Rainy days with < 5 mm:    ", np.sum((rainfall_mm > 0) & (rainfall_mm < 5)))

"""Boolean Arrays as Masks"""
arr_x_less_5 = x[x < 5]
# print(arr_x_less_5)

index = np.arange(len(rainfall_mm))

# boolean operations
rainy = rainfall_mm > 0
summer = (index > 172) & (index < 262)

# print("Median precip on rainy days in 2015 (mm):   ", np.median(rainfall_mm[rainy]))
# print("Median precip on summer days in 2015 (mm):  ", np.median(rainfall_mm[summer]))
# print("Maximum precip on summer days in 2015 (mm): ", np.max(rainfall_mm[summer]))
# print("Median precip on non-summer rainy days (mm):",np.median(rainfall_mm[rainy & ~summer]))

"""Using the Keywords and/or Versus the Operators &/|"""
# print(bool(42), bool(0))
# print(bool(42 and 0))
# print(bool(42 or 0))

# print(bin(42))
# print(bin(59))
# print(bin(42 & 59))
# print(bin(42 | 59))

a = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
b = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
# print(a | b)
# print(a or b) # ambiguous

x = np.arange(10)
# print((x > 4) & (x < 8))
# print((x > 4) & (x < 8)) # ambiguous