import numpy as np
from tools import vecs_to_nums
from is_cap_set import is_cap_set
from build_cap_set import ground_up
from random import shuffle
from itertools import chain
from tools import num_to_vec

#points_not_allowed = np.empty((3**8, 8))

#skip_values = np.ones(3 ** N)
smallestcapsetn3 = np.array(
    [np.array([0, 0, 0 ]),
     np.array([1, 0, 1]),
     np.array([1, 0 ,2]),
     np.array([1, 1, 0]),
     np.array([2, 1, 1]),
     np.array([2, 1, 2]),
     np.array([1, 2, 0]),
     np.array([2, 2, 1]),
     np.array([2, 2, 2])
     ]
     )



def ground_up(n : int,start_values = [], line_counts : list[int] = [],max_size : int = None, condition = lambda line_counts,i, amount : False):
    """
        Ground up finding of capsets algorythm. Does it randomly, expect different capsets every time. \n
        n : int - the dimension of the vectors \n
        skip_values : is ACTUALLY the line count, if the line count is negative we know the value is skipped \n
        max_size : int - maximum size of cap set returned \n
        start_values : int - this has to be a capset. the program will attempt to make the capset given larger \n
        condition : function that takes in line_counts, index, and the amount, and returns a true or false value. the capset will be generated such that the line counts 
        of all points in the capset DO NOT satisfy that condition.
        returns numpy array of vectors in the capset.
    """
    if max_size is None:
        max_size == 3 ** n
    
    is_blocked = np.full(shape=3 ** n, fill_value=False, dtype=bool)
    vectors = np.empty((3**n,n),dtype=np.int8)
    vectors_length = 0
    order = list(range(3**n))
    shuffle(order)


    for i in chain(start_values,order):
        if is_blocked[i]: 
            continue

        if (len(line_counts) > 0 and (line_counts[i] < 0 or condition(line_counts,i,max_size))):
            continue

        new_vec = num_to_vec(i,n)
        if vectors_length >= 1:
            blocking = vecs_to_nums(- vectors[:vectors_length, :] - new_vec[None, :],n)
            line_counts[blocking] = -1
            
        line_counts[i] = -1
        vectors[vectors_length] = new_vec
        vectors_length += 1

        if vectors_length == max_size:
            return vectors[:vectors_length]

    return vectors[:vectors_length]




def find_valid_next(point,n,smallestcapset,N):
    points = vecs_to_nums((-point-smallestcapset)%3,N)
    p = np.random.randint(0,3,N)
    p[:N-n] = 0
    p[N-n] = 2
    while vecs_to_nums((p[None,:]),N)[0] in points:
        p = np.random.randint(0,3,N)
        p[:N-n] = 0
        p[N-n] = 2
    return p

def give_base_capset(N):

    smallestcapsetn3 = np.array(
    [np.array([0, 0, 0 ]),
     np.array([1, 0, 1]),
     np.array([1, 0 ,2]),
     np.array([1, 1, 0]),
     np.array([2, 1, 1]),
     np.array([2, 1, 2]),
     np.array([1, 2, 0]),
     np.array([2, 2, 1]),
     np.array([2, 2, 2])
     ]
     )

    smallestcapset = np.zeros((9,N), dtype=int)
    for i,a in enumerate(smallestcapset):
        a[-1],a[-2],a[-3] = smallestcapsetn3[i][-1],smallestcapsetn3[i][-2],smallestcapsetn3[i][-3]
    final_capset = np.empty((9 + 2* (N-3),N),dtype=int)
    final_capset[:9] = smallestcapset

    def make_base_capset_attempt2(n,N):
        if n == 3:
            return 9

        final_capset_length = make_base_capset_attempt2(n-1,N) 
        point = np.random.randint(0,3,N)
        point[:N-n] = 0
        point[N-n] = 1
        p = find_valid_next(point,n,smallestcapset,N)
        final_capset[final_capset_length] = point
        final_capset[final_capset_length+1] = p
        final_capset_length += 2
        return final_capset_length
    make_base_capset_attempt2(N,N)
    return final_capset

def make_base_capset(n):
    """make a capset which contains every single hyperplane"""
    if n == 3:
        # return size 9 capset
        return ground_up(4, line_counts=skip_values, max_size=9)
    print(skip_values)
    capset = make_base_capset(n-1)
    print(skip_values)
    point = np.random.randint(0,3,4)
    while skip_values[vecs_to_nums(point[None,:],N)[0]] < 0:
        point = np.random.randint(0,3,4)
    

    skip_values[vecs_to_nums(point[None,:], N)] = -1
    skip_values[vecs_to_nums((-point[None,:]-capset) % 3, N)] = -1
    skip_values[vecs_to_nums((capset),N)] = -1

    capset2 = make_base_capset(n-1)

    final_capset[0] = point
    final_capset[1:len(capset)+1] = capset
    final_capset[len(capset)+1:] = capset2
    print(is_cap_set(final_capset[1:]))
    return final_capset
#a = make_base_capset(4)
if __name__ == "__main__":
    N=4
    final_capset = give_base_capset(N)
    print(final_capset.__repr__())
    points = (is_cap_set(final_capset))
