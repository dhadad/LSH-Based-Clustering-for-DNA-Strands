import numpy as np
import time
import random


# Naive Solution


def naive_clstring(all_reads, n=14):
    """
    Attempt to split the sequences to clusters via a trivial method: use the first n characters
    in a sequence as an index, then decide two sequences match based on this prefix being equal
    or not.
    :param all_reads: the whole input, in the form of an array of strings
    :param n: integer, the prefix length to be looked at as a key for the clustering
    :return C_til, dict of clusters. In the form of C_til[rep] = [reads assigned to the cluster]
    """
    time_start = time.time()
    prefix_to_ind = {}
    for i in range(len(all_reads)):
        if all_reads[i][:n] in prefix_to_ind:
            prefix_to_ind[all_reads[i][:n]].append(i)
        else:
            prefix_to_ind[all_reads[i][:n]] = [i]

    C_til = {i: [i] for i in range(len(all_reads))}
    for indexes in prefix_to_ind.values():
        C_til[indexes[0]] = indexes

    print("time for naive approach: {}".format(time.time() - time_start))
    return C_til


# Helper Functions


def qgram_val(sub_seq):
    """
    Calculate the value of a Q-gram
    :param sub_seq: sub string of the original sequence, of length q
    :return: integer, representing the value of the Q-gram
    """
    vals = {"A": 0, "C": 1, "G": 2, "T": 3}
    tot = 0
    for pos in range(len(sub_seq)):
        tot += (4 ** pos) * vals[sub_seq[pos]]
    return tot


def seq_numset(seq, q):
    """
    Convert a sequence into a set of numbers
    :param seq: an original line from the input file
    :param q: length of the divided sub-sequences (Q-grams)
    :return: array of integers, each one is the value of a Q-gram
    """
    arr = []
    for i in range(len(seq) - q + 1):
        arr.append(qgram_val(seq[i:i + q]))
    return arr


def mh_sig(numset, perm):
    """
    Obtain a MH signature for a sequence, based on given representation of it as a number set
    the given permutation and the definition for MH signature in the article
    :param numset: array of integers, each one is a Q-gram value (so its length is the
        original sequence's length minus q)
    :param perm: array, permutation of {0,..., 4**q}
    :return: MH signature in the form of a single integer
    """
    return min([perm[num] for num in numset])


def lsh_sig(numset, perms):
    """
    Obtain a LSH signature for a sequence, converted to its representation as a set of numbers
    :param numset: array of integers, each one is a Q-gram value (so its length is the
        original sequence's length minus q)
    :param perms: array of arrays, each: permutation of {0,..., 4**q}
    :return: an array of length equal to the nubmer of permutations given. each element is the
        MH signature of the sequence calculated with the permutation with the suitable index.
    """
    lsh = []
    for perm in perms:
        lsh.append(mh_sig(numset, perm))
    return lsh


# The LSH Clustering Algorithm


def _numsets(all_reads, q):
    """
    Generate the numbers sets for all the sequences
    :param all_reads: array of all the input sequences
    :param q: length of the divided sub-sequences (Q-grams)
    :return: a dictionary, mapping a number set for each sequence in the input,
        while the key is the index of the sequence in all_reads
    """
    time_start = time.time()
    numsets = {}
    for i in range(len(all_reads)):
        numsets[i] = seq_numset(all_reads[i], q)
    print("time to create number set for each sequence: {}".format(time.time() - time_start))
    return numsets


def _lsh_sigs(numsets, m, top):
    """
    Calculate the LSH signature of all the sequnces in the input
    :param numsets: array of arrays, each one is a set of number representing a sequence
    :param m: size of the LSH signature
    :param top: the largest possible number in the sets
    :return: array of LSH signatures (each is an array itself)
    """
    time_start = time.time()
    # generate m permutations
    perms = [np.random.permutation(top) for _ in range(m)]
    # LSH signature tuple (size m, instead of k, as the original paper suggests) for each sequence
    lsh_sigs = [lsh_sig(numsets[i], perms) for i in range(len(numsets))]
    print("time to create LSH signatures for each sequence: {}".format(time.time() - time_start))
    return lsh_sigs


def _add_pair(elem_1, elem_2, C_til):
    """
    Insert a pair of two sequences indices into C_til. In case both didn't appeared yet, we'll
    treat the first one as the center of the possible cluster.
    :param elem_1, elem_2: the indices of two sequences in all_reads.
    :param C_til: array of clusters. In the form of C_til[rep] = [reads assigned to the cluster]
    """
    if len(C_til[elem_2]) > 1 >= len(C_til[elem_1]):
        C_til[elem_2].extend(C_til[elem_1])
        C_til[elem_1] = []
    else:
        C_til[elem_1].extend(C_til[elem_2])
        C_til[elem_2] = []


def lsh_clstering(all_reads, q, k, m, L):
    """
    Run the full clustering algorithm: create the number sets for each sequence, then generate a LSH
    signature for each, and finally iterate L times looking for matching pairs, to be inserted to the
    same cluster.
    :param all_reads: array of strings, each: a DNA sequence from the input
    :param m: size of the LSH signature
    :param q: length of the divided sub-sequences (Q-grams)
    :param k: number of MH signatures in a LSH signature
    :param L: number of iterations of the algorithm
    :return C_til, dict of clusters. In the form of C_til[rep] = [reads assigned to the cluster]
    """
    numsets = _numsets(all_reads, q)
    lsh_sigs = _lsh_sigs(numsets, m, 4 ** q)
    C_til = {i: [i] for i in range(len(all_reads))}
    for itr in range(L):
        time_start = time.time()
        pairs = set()
        sigs = []
        buckets = {}
        # first, choose random k elements of the LSH signature
        # then, by giving a weight to each one, their sum will act as the new signature
        indexes = random.sample(range(m), k)
        for lsh in lsh_sigs:
            sig = sum(lsh[indexes[i]] * ((4 ** q) ** i) for i in range(k))
            sigs.append(sig)

        # buckets[sig] = [indexes (from all_reads) of (hopefully) similar sequences]
        for i in range(len(all_reads)):
            if sigs[i] in buckets:
                buckets[sigs[i]].append(i)
            else:
                buckets[sigs[i]] = [i]

        # from each bucket we'll keep pairs. the first element will be announced as center
        for elems in buckets.values():
            if len(elems) <= 1:
                continue
            for elem in elems[1:]:
                pairs.add((elems[0], elem))

        # pairs = sorted(list(pairs))
        for pair in pairs:
            _add_pair(pair[0], pair[1], C_til)

        print("time for iteration {} in the algorithm: {}".format(itr + 1, time.time() - time_start))

    return C_til


# Accuracy Calculation


def rep_in_C(read, C_reps):
    lower = 0
    upper = len(C_reps) - 1
    while lower <= upper:
        mid = lower + int((upper - lower) / 2)
        if read == (C_reps[mid][0]):
            return C_reps[mid][1]
        if read > (C_reps[mid][0]):
            lower = mid + 1
        else:
            upper = mid - 1
    return -1


def comp_clstrs(alg_clstr, org_clstr, gamma, reads_err):
    num_exist = 0
    if len(alg_clstr) > len(org_clstr):
        return 0
    for i in range(0, len(alg_clstr)):
        flg_exist = 0
        for j in range(0, len(org_clstr)):
            if reads_err[alg_clstr[i]] == org_clstr[j]:
                flg_exist = 1
                num_exist += 1
                break
        if flg_exist == 0:
            return 0
    if num_exist < gamma * len(org_clstr):
        return 0
    return 1


def calc_acrcy(clustering, C_dict, C_reps, gamma, reads_err):
    acrcy = 0
    for i in range(0, len(clustering)):
        if len(clustering[i]) >= 1:
            acrcy += comp_clstrs(clustering[i],
                                 C_dict[rep_in_C(reads_err[clustering[i][0]], C_reps)], gamma, reads_err)
    return acrcy


# Reading The Data


reads_cl = []  # the whole input
with open(r'C:\Users\Adar\Documents\git_repos\yupyter\evyat.txt') as f:
    for line in f:
        reads_cl.append(line.strip())
cnt = 0
reads = []  # representatives
for i in range(0, len(reads_cl)):
    if reads_cl[i] != "":
        if reads_cl[i][0] == "*":
            cnt += 1
            rep = reads_cl[i - 1]
            reads.append(rep)

'''
Construct the setup for a run.
C_reps = [(Read, Cluster rep of the cluster to which the read belongs to)]
C_dict = {Cluster rep: All the Reads that belong to that cluster}
'''
C_reps = []
C_dict = {}
rep = reads_cl[0]
for i in range(1, len(reads_cl)):
    if reads_cl[i] != "":
        if reads_cl[i][0] == "*":
            if len(C_reps) > 0:
                # the last sequence is to be placed in a different cluster
                C_dict[rep].pop()
                C_reps.pop()
            rep = reads_cl[i - 1]
            C_dict[rep] = []
        else:
            C_dict[rep].append(reads_cl[i])
            C_reps.append((reads_cl[i], rep))
C_reps.sort(key=lambda x: x[0])

reads_err = [0] * (len(C_reps))
for i in range(0, len(C_reps)):
    reads_err[i] = C_reps[i][0]
random.shuffle(reads_err)


# Test the clustering algorithm


#C_til = lsh_clstering(all_reads=reads_err, q=7, k=5, m=50, L=32)
C_til = naive_clstring(reads_err)
acrcy1 = calc_acrcy(C_til, C_dict, C_reps, 0.6, reads_err) / len(reads)
acrcy2 = calc_acrcy(C_til, C_dict, C_reps, 0.7, reads_err) / len(reads)
acrcy3 = calc_acrcy(C_til, C_dict, C_reps, 0.8, reads_err) / len(reads)
acrcy4 = calc_acrcy(C_til, C_dict, C_reps, 0.9, reads_err) / len(reads)
acrcy5 = calc_acrcy(C_til, C_dict, C_reps, 0.95, reads_err) / len(reads)
acrcy6 = calc_acrcy(C_til, C_dict, C_reps, 0.99, reads_err) / len(reads)
acrcy7 = calc_acrcy(C_til, C_dict, C_reps, 1, reads_err) / len(reads)
print("Accuracy:", acrcy1, acrcy2, acrcy3, acrcy4, acrcy5, acrcy6, acrcy7)
