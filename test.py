import numpy as np
from tools import vecs_to_nums

n = 6
k =5
q = 3
total = 1
for i in range(k):
    total *=  (pow(q,n) - pow(q,i))/(pow(q,k) - pow(q,i))
print(total)