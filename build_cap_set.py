import numpy as np
from random import shuffle

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
            (vec) % 3, powers)
    return num

def ground_up(n : int,skip_values : set[int] = set(),max_size : int = None):
    """
        Ground up finding of capsets algorythm. Does it randomly, expect different capsets every time. \n
        n : int - the dimension of the vectors \n
        skip_values : container that supports in keyword - the enumeration of vectors which are forbidden from being in the capset \n
        max_size : int - maximum size of cap set returned \n
        returns numpy array of vectors in the capset.
    """
    if max_size is None:
        max_size == 3 ** n
    is_blocked = np.full(shape=3 ** n, fill_value=False, dtype=bool)
    vectors = np.empty((3**n,n),dtype=np.int8)
    vectors_length = 0
    order = list(range(3**n))
    shuffle(order)
    for i in order:
        if is_blocked[i] or i in skip_values:
            continue
        new_vec = num_to_vec(i,n)
        if vectors_length >= 1:
            blocking = vecs_to_nums(- vectors[:vectors_length, :] - new_vec[None, :],n)
            is_blocked[blocking] = True
        is_blocked[i] = True
        vectors[vectors_length] = new_vec
        vectors_length += 1

        if vectors_length == max_size:
            return vectors[:vectors_length]

    return vectors[:vectors_length]

def is_cap_set(vectors: np.ndarray) -> bool:
  """Returns whether `vectors` form a valid cap set.

  Checking the cap set property naively takes O(c^3 n) time, where c is the size
  of the cap set. This function implements a faster check that runs in O(c^2 n).

  Args:
    vectors: [c, n] array containing c n-dimensional vectors over {0, 1, 2}.
  """
  _, n = vectors.shape

  # Convert `vectors` elements into raveled indices (numbers in [0, 3^n) ).
  powers = np.array([3 ** j for j in range(n - 1, -1, -1)], dtype=int)  # [n]
  raveled = np.einsum('in,n->i', vectors, powers)  # [c]

  # Starting from the empty set, we iterate through `vectors` one by one and at
  # each step check that the vector can be inserted into the set without
  # violating the defining property of cap set. To make this check fast we
  # maintain a vector `is_blocked` indicating for each element of Z_3^n whether
  # that element can be inserted into the growing set without violating the cap
  # set property.
  is_blocked = np.full(shape=3 ** n, fill_value=False, dtype=bool)
  for i, (new_vector, new_index) in enumerate(zip(vectors, raveled)):
    if is_blocked[new_index]:
      return False  # Inserting the i-th element violated the cap set property.
    if i >= 1:
      # Update which elements are blocked after the insertion of `new_vector`.
      blocking = np.einsum(
          'nk,k->n',
          (- vectors[:i, :] - new_vector[None, :]) % 3, powers)
      is_blocked[blocking] = True
    is_blocked[new_index] = True  # In case `vectors` contains duplicates.
  return True  # All elements inserted without violating the cap set property.


def felipe_algorythm(n,skip_values):
    """
    for a field with q = 3 n = n, returns an array of which vectors have blocks when the vectors in skip_values are removed
    """
    blocking_count = np.full(shape = 3**n, fill_value=0,dtype=np.int16)
    vectors =  np.empty((3**n,n),dtype=np.int8)
    values_to_skip = skip_values
    k = 0
    for i in range(3**n):
        new_vec = num_to_vec(i,n)
        if i in values_to_skip:
           continue
        if i >= 1:
            powers = np.array([3 ** j for j in range(n - 1, -1, -1)], dtype=int)  # [n]

            blocking = np.einsum(
            'nk,k->n',
            (- vectors[:k, :] - new_vec[None, :]) % 3, powers)
            blocking_count[blocking] += 1
        vectors[k] = new_vec
        k += 1
    return blocking_count


def nums_to_vecs(nums,n):
    vectors = np.empty((len(nums), n),dtype=np.int8)
    for i, num in enumerate(nums):
        vectors[i] = num_to_vec(num,n)
    return vectors


def get_rid_of_lines(n):
   
    initial_values = nums_to_vecs(list(range(20)),4)
    directions = [np.array([1,1,1,1]),np.array([1,2,3,1]),np.array([1,0,0,0])]
    print(initial_values)
    vecs =  np.array([i*np.ones(n) + initial_values[i//3] for i in range(3*20)],dtype=int)
    skip_values = vecs_to_nums(vecs,n)

    print(sorted(skip_values),len(set(skip_values)))
    print(a := felipe_algorythm(n,skip_values=skip_values))

def get_rid_of_capset(n):
    a = ground_up(n,[],3)
    skip_values = vecs_to_nums(a,n)
    print(felipe_algorythm(4,skip_values))
    b = ground_up(n,skip_values)
    print(is_cap_set(a))
    print(vecs_to_nums(b,n))
    skip_values = np.concatenate((vecs_to_nums(a,n),vecs_to_nums(b,n))) 
    
n= 4
#print(is_cap_set(nums_to_vecs(skip_values,n)))
#skip_values=[24,74,17,32,49,7,64,54,39]
skip_values = [24,74,17]
print("skip_values", len(skip_values))
a = felipe_algorythm(n,skip_values=skip_values)
print(a)
location = np.where(a == 38)[0]
location.flags.writeable = False
unique, counts = np.unique(a,return_counts=True)
print(dict(zip(unique,counts)))
print(set(location) - set(skip_values))
