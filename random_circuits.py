import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from numba import jit
from functools import lru_cache


def entanglement_entropy(stabilisers, A):
    """entanglement entropy of a region A.
    If onesided - set i or j to None. inclusive, exclusive
    Args:
        A (Tuple[Tuple[int]] or Tuple[int]): A region, or a list of regions.
        Will be interpreted as a set of pairs (i, j), delimiting the intervals that should be joined to make the interval of interest."""
    n = stabilisers.shape[0]

    ## list of intervals. Don't check whether they overlap.
    # Interval manipulations
    stab_ind = reduce(list.__add__, [list(range(*ij)) for ij in A])
    destab_ind = reduce(
        list.__add__, [list(range(*(np.array(ij) + n))) for ij in A]
    )
    size = np.sum([np.abs(i - j) for i, j in A])

    # Stabilizer manipulations
    Psi = stabilisers[:, stab_ind+destab_ind].T
    ent = gf2_rank(Psi) - size
    return ent

#def row_to_int(row):
#    return int("".join(row.astype(str)), 2)

def row_to_int(row):
    #return int("".join(row.astype(str)), 2)
    return sum(1<<i for i, b in enumerate(row) if b)

def rows_to_ints(mat):
    ints = [row_to_int(row) for row in mat]
    return ints

def gf2_rank_ints(rows):
    """
    Find rank of a matrix over GF2.
    https://stackoverflow.com/questions/56856378/fast-computation-of-matrix-rank-over-gf2

    The rows of the matrix are given as nonnegative integers, thought
    of as bit-strings.

    This function modifies the input list. Use gf2_rank(rows.copy())
    instead of gf2_rank(rows) to avoid modifying rows.
    """
    rank = 0
    while rows:
        pivot_row = rows.pop()
        if pivot_row:
            rank += 1
            lsb = pivot_row & -pivot_row
            for index, row in enumerate(rows):
                if row & lsb:
                    rows[index] = row ^ pivot_row
    return rank

def gf2_rank(mat):
    return gf2_rank_ints(rows_to_ints(mat))

def SAB(mat):
    L = mat.shape[0]
    SAB = np.zeros((L, L), np.uint8)
    for i in range(L):
        for j in range(L):
            if not i<=j:
                SAB[i, j] = entanglement_entropy(mat, ((i, i+1), (j, j+1)))

    SAB = SAB+SAB.T
    L = mat.shape[0]
    for i in range(L):
        SAB[i, i] = entanglement_entropy(mat, ((i, i+1),))

    return SAB
    

if __name__=='__main__':
    ## Split the big data into 10 bits
    #arr = np.load('data/all_tableaus.npy')
    #n_samples = arr.shape[0]
    #for i, m in enumerate(range(0, n_samples, n_samples//10)):
    #    np.save(f'data/{i+1}.npy', arr[m*n_samples//10:(m+1)*n_samples//10, :, :])
    #print(arr.shape)

    import time
    k = 8
    K = None
    arr = np.load(f"data/all_tableaus.npy")[:None, :, :]
    SABs = np.zeros((arr.shape[0], arr.shape[1], arr.shape[1]))
    t = time.time()
    total_expected = 0
    for i in range(SABs.shape[0]):
        if i == 1:
            total_expected = (time.time()-t)*SABs.shape[0]
        print(f"{i}/{SABs.shape[0]} :{round(time.time()-t, 2)}/{round(total_expected, 2)}    \r", sep='', end='', flush=True)
        SABs[i] = SAB(arr[i])
    print('\n', time.time()-t)
    np.save(f'data/SAB.npy', SABs)
