import numpy as np
from tools import vecs_to_nums
# Example 2D array
a = np.array([[np.nan, np.nan, np.nan],
              [4, 5, 6],
              [7, 8, 9]])

b = np.array([1,2,3,4,5,6])


# Vector to find
print(list(map(int,vecs_to_nums(a - np.array([1,1,1])[None, :],3))))

