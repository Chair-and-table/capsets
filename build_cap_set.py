import numpy as np
from random import shuffle
from is_cap_set import is_cap_set
import heapq

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


def felipe_algorythm(n,skip_values):
    """
    For a field with q = 3 n = n, ignoring all vectors in skip_values where skip_values contains enumerated vectors,
    returns an array of integers, where the index represents an enumerated point and the value represents how many lines go through
    that point

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
    """not important"""
   
    initial_values = nums_to_vecs(list(range(20)),4)
    directions = [np.array([1,1,1,1]),np.array([1,2,3,1]),np.array([1,0,0,0])]
    print(initial_values)
    vecs =  np.array([i*np.ones(n) + initial_values[i//3] for i in range(3*20)],dtype=int)
    skip_values = vecs_to_nums(vecs,n)

    print(sorted(skip_values),len(set(skip_values)))
    print(a := felipe_algorythm(n,skip_values=skip_values))

def find_next_togetridof(lines_count,amount, skip_values=[]):
    """
    Given:
    lines_count - list of integers describing at each index i how many lines go through enumerated point i
    amount - how many points are to be removed
    skip_values - points which already have been removed
    
    Returns:
    An array of enumerated points. These points are the ones that have the most lines going through them,
    and are not found in teh skip_values array. If the choice is ambiguous, it removes {amount} points such that
    no three of those points lie a line
    
    """
    next_values = np.empty(amount,dtype=int)
    amount_we_have = 0

    # a little bit of python magic
    # easiest way to understand it is with an example:
    # it will turn [10,11,20,31,0] into [(10,0),(11,1),(20,2),(31,3),(0,4)] and then sort it in reverse order
    # based on the first value of each tuple. so it will become [(31,3),(20,2),(11,1),(10,0),(0,4)]
    indexed_sorted_list = sorted(list(zip(lines_count,range(0,len(lines_count)))),key=lambda x: x[0],reverse=True)
    i = 0
    
    if len(skip_values) % (3 ** (amount - 1)) == 0:
        # the dimension of the object removed from the capset is equal to amount - 1.
        # if we have already removed all points of the object from
        # the capset then we must generate a new capset from what is remaining.
        return vecs_to_nums(ground_up(n, skip_values=skip_values, max_size=amount),n)
    

    while amount > amount_we_have:

        if i == len(indexed_sorted_list):
            return next_values[:amount_we_have]
        
        if indexed_sorted_list[i][1] not in skip_values:
            next_values[amount_we_have] = indexed_sorted_list[i][1]
            amount_we_have += 1
        i += 1
    return next_values

def get_capset_from_line_count(lines_count, skip_values=[]):
    """from line count gets capset, returns it in enumerated form"""
    capset = []
    for vec_num, lines_count in enumerate(lines_count):
        if lines_count != 0:
            continue
        if vec_num in skip_values:
            continue
        capset.append(vec_num)
    return capset

def get_rid_of_capset_method(n,capset_size,verbose=False):
    """Apply the method of getting rid of capsets from the set. In practice this is just removing {capset_size-1}-hyperplanes
    from the set"""

    len_skip_values = 0
    itteration_number = 1
    skip_values = np.empty(3**n,dtype=np.int64)
    vectors_to_get_rid_of = vecs_to_nums(ground_up(n,[],capset_size),n)
    logs = ""
    logs += f"n = {n} capset_size = {capset_size} \n"
    while len_skip_values < 61: #61 is almost completely arbitrary, i was just trying different numbers
        
        #updating the vectors that need to be removed from the set
        #all vectors are enumerated.
        skip_values[len_skip_values:len_skip_values + len(vectors_to_get_rid_of)] = vectors_to_get_rid_of
        len_skip_values += len(vectors_to_get_rid_of)
        

        logs += f"Itteration number:  {itteration_number}\n"
        logs += f"Values that are skipped:  {skip_values[:len_skip_values]}\n"
        logs += f"Amount of skipped values: {len_skip_values}\n"

        # for each vector i, get the amount of lines going through it.
        lines_count = felipe_algorythm(n,skip_values=skip_values[:len_skip_values])

        logs += f"lines_count  {lines_count}\n"
        logs += "\n"

        itteration_number += 1

        vectors_to_get_rid_of = find_next_togetridof(lines_count,capset_size,skip_values[:len_skip_values])

    capset = get_capset_from_line_count(lines_count,skip_values)
    logs += f"Final capset: \n {capset}"
    if verbose:
        with open("logs.txt","a") as f:
            f.write(logs)
            f.write("\n\n\n")
    return capset

n= 4

def main():
    capset = get_rid_of_capset_method(4,2,verbose=True)
    print("Capset", capset)
    print(is_cap_set(nums_to_vecs(capset,n)))
#print(is_cap_set(nums_to_vecs(skip_values,n)))
#skip_values=[24,74,17,32,49,7,64,54,39]
if __name__ == "__main__":
    main()

""" location.flags.writeable = False
unique, counts = np.unique(a,return_counts=True)
print(dict(zip(unique,counts)))
print(set(location) - set(skip_values)) """
