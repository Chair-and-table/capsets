import numpy as np
from tools import vecs_to_nums
from is_cap_set import is_cap_set

#points_not_allowed = np.empty((3**8, 8))

#skip_values = np.ones(3 ** N)





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


#a = make_base_capset(4)
if __name__ == "__main__":
    N=8
    final_capset = give_base_capset(N)
    print(final_capset)
    points = (is_cap_set(final_capset))
