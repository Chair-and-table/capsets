import numpy as np
from random import shuffle, randint
from is_cap_set import is_cap_set
from multiprocessing import Pool
from tools import *


debugglobal = ""

def ground_up(n : int,start_values = [], skip_values : list[int] = [],max_size : int = None):
    """
        Ground up finding of capsets algorythm. Does it randomly, expect different capsets every time. \n
        n : int - the dimension of the vectors \n
        skip_values : is ACTUALLY the line count, if the line count is negative we know the value is skipped \n
        max_size : int - maximum size of cap set returned \n
        start_values : int - this has to be a capset. the program will attempt to make the capset given larger \n
        returns numpy array of vectors in the capset.
    """
    if max_size is None:
        max_size == 3 ** n
    is_blocked = np.full(shape=3 ** n, fill_value=False, dtype=bool)
    vectors = np.empty((3**n,n),dtype=np.int8)
    vectors_length = 0
    order = list(range(3**n))
    shuffle(order)


    for value in start_values:
        if is_blocked[value]:
            raise Exception("What are you doing it says in the doc string start_values should be a capset")
        
        if skip_values and skip_values[value] < 0:
            raise Exception("Why did you put values that are in skip_values into start values?")


        new_vec = num_to_vec(value,n)
        if vectors_length >= 1:
            blocking = vecs_to_nums(- vectors[:vectors_length, :] - new_vec[None, :],n)
            is_blocked[blocking] = True

        is_blocked[value] = True
        vectors[vectors_length] = new_vec
        vectors_length += 1

    for i in order:
        if is_blocked[i] or (len(skip_values) > 0 and skip_values[i] < 0):
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


def felipe_algorythm(n : int,skip_vectors: np.ndarray[np.ndarray[int]],skip_vectors_len : int , prev_line_count: np.ndarray[int], point_removed_enum : int) -> np.ndarray[int]:
    """
    For a field with q = 3 n = n, ignoring all vectors in skip_values where skip_values contains enumerated vectors,
    returns an array of integers, where the index represents an enumerated point and the value represents how many lines go through
    that point
    """
    # if linecounts[i] < 0 -> vector i has been removed. 
    # Not possible for point_removed_enum to be repeated, or for the amount of lines going through it to be 0
    prev_line_count[point_removed_enum] *= -1
    
    point_removed = num_to_vec(point_removed_enum, n)
    prev_line_count -= 1
    prev_line_count[vecs_to_nums((-skip_vectors[:skip_vectors_len] - point_removed[None, :]) % 3,n)] += 1

    return prev_line_count


    

def find_next_togetridof(n,lines_count : np.ndarray,amount : int,skip_vectors_len : int,randomchance =0):
    """
    Given:

    lines_count - list of integers describing at each index i how many lines go through enumerated point i
    amount - how many points are to be removed
    skip_vectors - points which already have been removed
    Returns:
    An array of enumerated points. These points are the ones that have the most lines going through them,
    and are not found in the skip_vectors array. If the choice is ambiguous, it removes {amount} points such that
    no three of those points lie on a line
    
    """
    global debugglobal

    if skip_vectors_len % (3 ** (amount - 1)) == 0:
        # the dimension of the object removed from the capset is equal to amount - 1.
        # if we have already removed all points of the object from
        # the capset then we must generate a new capset from what is remaining.
        potential_vectors =  vecs_to_nums(ground_up(n, skip_values=lines_count, max_size=amount),n)
        attempts = 0
        there_exists_zero = True

        # this bit of the code is a bit iffy
        # definetly room for improvement here
        while there_exists_zero:
            there_exists_zero = False
            attempts += 1
            for vector_enum in potential_vectors:
                
                # This is to prevent an obscure bug. If you return a list of lenght amount of values and one of the values
                # has a line count going through it that is less than or equal to amount -1 its possible for the removal of previous
                # values to cause other values to suddenly have 0 lines going through them. This triggers the stop condition, resulting
                # in early termination
                if 0 < lines_count[vector_enum] <= amount - 1:
                    debugglobal = "returned from double inequality"
                    return [vector_enum] 
                

                if lines_count[vector_enum] == 0:
                    there_exists_zero = True

            # this is to prevent infinte loops. If you couldn't find a capset where all the points have lines going through them after 15 attempts, it probably doesn't exist
            if attempts == 10:
                count = 0
                for line_count in lines_count:
                    if line_count > 0:
                        count += 1
                    if count < amount:
                        debugglobal = "returned from the attempts check"
                        return [np.argmax(lines_count)]

            potential_vectors = vecs_to_nums(ground_up(n, skip_values=lines_count, max_size=amount),n)
        
        debugglobal = "returned from after the while loop"
        return potential_vectors
    
    if randint(1,100) <= randomchance:
        potential_vector =  randint(0,len(lines_count) - 1)
        if lines_count[potential_vector] > 0:
            debugglobal = "returned from random chance"
            return [potential_vector]
    # return index of maximum value
    debugglobal = "returned from the very end"
    return [np.argmax(lines_count)]




def get_rid_of_capset_method(n : int,capset_size: int,randomchance : int =0) -> np.ndarray[np.ndarray[int]]:
    """
    Apply the method of getting rid of capsets from the set. In practice this is just removing {capset_size-1}-hyperplanes
    from the set. If randomchance is greater than 0, some points will be removed randomly.

    n - dimension of vectors.

    capset_size - the size of the capsets that are removed when the choice of value removal is ambiguous.

    randomchance - percentchance that when a point is going to be removed, a random one is removed instead.
    For optimisation purposes, the actual chance is below the chance specified and actually decreases as the algorythm 
    gets closer to finishing.\n

    Returns: 

    2d numpy array of vectors size n which form a capset of the field n = n q = 3.
    """ 


    set_size = 3**n
    skip_vectors = np.empty((set_size,n), dtype=int)
    skip_vectors_len = 0
 
    lines_count = np.full(set_size, (set_size - 1)/2, dtype=int)

    # lines_count[i] is greater than or equal to 0 if vector i is in our subset, and less than 0 if it's been removed
    vectors_to_get_rid_of = vecs_to_nums(ground_up(n,skip_values=lines_count, max_size=capset_size),n)


    while set_size - skip_vectors_len:
        #updating the vectors that need to be removed from the set
        #all vectors are enumerated.
        for vector_enum in vectors_to_get_rid_of:

            # If the best vector to get rid of has 0 lines going through, all others must also have 0 lines going through them
            # i.e. capset.
            if lines_count[vector_enum] == 0:
                if set_size - skip_vectors_len <= 12:
                    print(debugglobal)
                capset = get_capset_from_line_count(lines_count)
                return capset
            
            # updating points removed
            skip_vectors[skip_vectors_len] = num_to_vec(vector_enum,n)
            skip_vectors_len += 1

            # for each vector i, get the amount of lines going through it with the point being removed
            lines_count = felipe_algorythm(n,skip_vectors=skip_vectors,skip_vectors_len=skip_vectors_len, prev_line_count=lines_count, point_removed_enum=vector_enum)
            
            
        # finds the next best vector to get rid of
        vectors_to_get_rid_of = find_next_togetridof(n,lines_count,capset_size, skip_vectors_len,randomchance=randomchance)
    
    # at this point, we should have no points left to remove. All that is left to do is use the lines_count to figure out what vectors
    # are in our capset.
    capset = get_capset_from_line_count(lines_count)

    return capset


def run_a_bunch(n):

    capset1max = get_rid_of_capset_method(4,3)
    capset2max = ground_up(n,start_values=capset1max)
    for _ in range(100):
        capset1 = get_rid_of_capset_method(4,3)
        capset2 = ground_up(n,start_values=capset1)
        if len(capset2max) < len(capset2):
            capset2max = capset2
            capset1max = capset1
    print(capset1max)
    print(capset2max)
    print(is_cap_set(capset2max),len(capset2max))

def run(params : list):
    """
    params: n, sample_number, filename, random_chance, capset_size

    it'll write the results to filename.
    """
    n,sample_number, file,random_chance,capset_size = params
    with open(file, "w") as f:
        f.write(f"n: {n} capset_size: {capset_size} random_chance: {random_chance}\n")
    capsets = []
    for i in range(sample_number):
        with open(file, "a") as f:
            capset = get_rid_of_capset_method(n,capset_size=capset_size,randomchance=random_chance)
            f.write(f"Begining capset: {capset}\n")
            f.write(f"Capset length:  {len(capset)}\n")
            capset2 = ground_up(n, start_values=capset)
            f.write(str(capset2))
            f.write("\n")
            f.write(f"Len of capset  {len(capset2)}\n")
            capsets.append(str(len(capset2)))
    with open(file, "a") as f:
        f.write("	".join(capsets))
        f.write("\n")


def main():
    params = []
    n = 8
    capset_size = 3
    sample_size = 20

    for i in range(11):
        params.append([n,sample_size, f"logs{i}.txt", i * 10,   capset_size])

    with Pool() as pool:
        list(pool.imap_unordered(run, params))

#print(is_cap_set(nums_to_vecs(skip_values,n)))
#skip_values=[24,74,17,32,49,7,64,54,39]
if __name__ == "__main__":
    main()
""" location.flags.writeable = False
unique, counts = np.unique(a,return_counts=True)
print(dict(zip(unique,counts)))
print(set(location) - set(skip_values)) """