import numpy as np
y = np.array([0, 0, 0])
x = np.array([  
    [1,2],
    [3,4], 
    [5,6]
])
a = np.array([10, 20])

y = x @ a
print(y)