import numpy as np
from math import log

def find_threshold_entropy(x, y, m,
                           idx,
                            n_classes, min_leaf):
    """
    Find the threshold for continuous attribute values that maximizes
    information gain.

    Argument min_leaf sets the minimal number of data instances on each side
    of the threshold. If there is no threshold within that limits with positive
    information gain, the function returns (0, 0).

    Args:
        x: attribute values
        y: class values
        idx: arg-sorted indices of x (and y)
        n_classes: the number of classes
        min_leaf: the minimal number of instances on each side of the threshold

    Returns:
        (highest information gain, the corresponding optimal threshold)
    """
    # print("x = ", list(x))
    # print("y = ", list(y))
    # print("m = ", list(m))
    # print("idx = ", list(idx))
    # print("n_classes = ", n_classes)
    # print("min_leaf = ", min_leaf)
    
    distr = np.zeros(2 * n_classes, dtype=np.uint32)
    offset_distr = np.zeros(2 * n_classes, dtype=np.uint32)
    # i, j
    # entro, class_entro, best_entro
    # p, curr_y
    best_cut = None
    N = idx.shape[0]

    # Initial split (min_leaf on the left)
    if N <= min_leaf:
        return 0, 0
    
    # min_leaf = 1 # we have to check all possoble splits
    
    for i in range(0):  # one will be added in the loop
        distr[n_classes + int(y[idx[i]])] += 1
    for i in range(0, N):
        distr[int(y[idx[i]])] += 1
    
    # print("distr", distr)
    
    # Compute class entropy
    class_entro = N * log(N)
    for j in range(n_classes):
        p = distr[j] + distr[j + n_classes]
        if p:
            class_entro -= p * log(p)
    best_entro = class_entro
    
    # Loop through
    for i in range(0, N - 1):
        curr_y = int(y[idx[i]])
        distr[curr_y] -= 1
        distr[n_classes + curr_y] += 1
        i_offset = 0
        offset = (-m[idx[i]] + m[idx[i+1]]) * 0.5 # TODO dodaj spremenljivko self.uncertainty_multiplyer
        cut = (x[idx[i]] + x[idx[i+1]]) * 0.5
        if offset < 0:  # glede na to a se offset zveče al zmanša maš 2 variante al se premekne levo al desno nov treshold
            while i + i_offset > 0 and cut + offset < x[idx[i + i_offset]]:
                i_offset -= 1
        if offset > 0:
            while i + i_offset + 1 < N and cut + offset > x[idx[i + i_offset + 1]]:
                i_offset += 1
                
        if i + i_offset < min_leaf - 1 or i + i_offset > N - min_leaf - 1:
            continue
        
        for j in range(0, len(distr)): #coppy distr
            offset_distr[j] = distr[j]
        if i_offset > 0:
            for j in range(1, i_offset+1):
                curr_y = int(y[idx[i + j]])
                offset_distr[curr_y] -= 1
                offset_distr[n_classes + curr_y] += 1
        if i_offset < 0:
            for j in range(i_offset, 0):
                curr_y = int(y[idx[i + j]])
                offset_distr[curr_y] += 1
                offset_distr[n_classes + curr_y] -= 1
        
        if x[idx[i + i_offset]] != x[idx[i + i_offset + 1]]:
            entro = (i + i_offset + 1) * log(i + i_offset + 1) + (N - (i + i_offset) - 1) * log(N - (i + i_offset) - 1)
            for j in range(2 * n_classes):
                if offset_distr[j]:
                    entro -= offset_distr[j] * log(offset_distr[j])
            if entro < best_entro:
                best_entro = entro
                best_cut = cut + offset
    
    # print(best_cut)
    if best_cut is None:
        return 0, 0
    
    return (class_entro - best_entro) / N / log(2), best_cut


x =  [0.6315830194020284, 0.6005169976989805, 0.685445512338765, 0.6646589970581247, 0.8702451499838711, 0.7900349589151867]
y =  [3.0, 3.0, 3.0, 2.0, 3.0, 3.0]
m =  [0.15964653, 0.000455345, 0.11042741, 0.14302266, 0.006831112, 0.09479967]
idx =  [1, 0, 3, 2, 5, 4]
n_classes =  4
min_leaf =  3
x = np.array(x)
y = np.array(y)
m = np.array(m)
idx = np.array(idx)

best_score, best_cut = find_threshold_entropy(x, y, m,
                           idx,
                            n_classes, min_leaf)

print(best_score, best_cut)
print(x > best_cut)

