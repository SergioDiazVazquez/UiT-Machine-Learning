import numpy as np

def dot_product(x, a):
    y = np.sum(x * a, axis=1)
    return y

x = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

a = np.array([10, 20])

print(dot_product(x, a))