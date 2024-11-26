import numpy as np

def num_to_vec(x : int, n: int) -> np.ndarray:
    vector = np.zeros(n,dtype=np.int8)
    for i in range(n-1,-1,-1):
        vector[i] = x% 3
        x //= 3
    return vector

def vecs_to_nums(vec, n):
   
    powers = np.array([3 ** j for j in range(n - 1, -1, -1)], dtype=int)  # [n]
    num = np.einsum(
            'nk,k->n',
            (vec % 3), powers)
    return num


def nums_to_vecs(nums,n):
    vectors = np.empty((len(nums), n),dtype=np.int8)
    for i, num in enumerate(nums):
        vectors[i] = num_to_vec(num,n)
    return vectors



def get_capset_from_line_count(lines_count):
    """from line count gets capset, returns it in enumerated form"""
    capset = []
    for vec_num, line_count in enumerate(lines_count):
        if line_count != 0:
            continue
        capset.append(vec_num)
    return capset