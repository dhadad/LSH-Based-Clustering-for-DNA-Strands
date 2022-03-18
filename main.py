import numpy as np
import time
import random
import plotly.express as px
import pandas as pd
from functools import cached_property
import multiprocessing as mp
from array import array

# Global Constants
INDEX_LEN = 11
BASE_VALS = {"A": 0, "C": 1, "G": 2, "T": 3}


def cache_symmetric(func):
    """
    The decorator is to be used when we wish to avoid recalculating values already seen.
    (The difference from functools' @cache is in being able to notice also the symmetric version of the key).
    """
    cache = dict()
    def outer_func(s1, s2):
        if (s1, s2) in cache:
            return cache[(s1, s2)]
        else:
            result = func(s1, s2)
            cache[(s1, s2)] = result
            cache[(s2, s1)] = result
            return result
    return outer_func


@cache_symmetric
def edit_dis(s1, s2):
    """
    Fully calculate the edit distance between two sequences. O(n^2) using dynamic programming.
    :param s1, s2: the two strings to get the distance between.
    """
    if not s1 or not s2:
        return float('inf')
    tbl = {}
    for i in range(len(s1) + 1):
        tbl[i, 0] = i
    for j in range(len(s2) + 1):
        tbl[0, j] = j
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            tbl[i, j] = min(tbl[i, j - 1] + 1, tbl[i - 1, j] + 1, tbl[i - 1, j - 1] + cost)
    return tbl[i, j]


class LSHCluster:
    def __init__(self, all_reads, q, k, m, L, rand_subs=True):
        """
        Initiate an object dedicated for clustering the DNA sequences in 'all_reads'.
        The function just makes ready the necessary data structures. For the starting the clustering
        process use the 'run' method.
        :param all_reads: array of strings, each: a DNA sequence from the input
        :param m: size of the LSH signature
        :param q: length of the divided sub-sequences (Q-grams)
        :param k: number of MH signatures in a LSH signature
        :param L: number of iterations of the algorithm
        :param rand_subs: determine whether the sub arrays (of the num sets) used for creating an LSH
            signature in each iteration will consist of adjacent items or random ones.
            The original article suggest the random option.
            if False is given, it's recommended to choose L s.t: k * L = m
        """
        self.all_reads = all_reads
        self.q = q
        self.k = k
        self.m = m
        self.L = L
        self.top = 4 ** q
        self.rand_subs = rand_subs
        self.sc = LSHCluster.Score(len(all_reads))

        # array of clusters: C_til[rep] = [reads assigned to the cluster]
        self.C_til = {idx: [idx] for idx in range(len(all_reads))}

        # array for tracking the sequence with the highest score in the cluster
        self.max_score = [(idx, 0) for idx in range(len(all_reads))]

        # mapping between a sequence's index to it's parent's index
        self.parent = [idx for idx in range(len(all_reads))]

    class Score:
        """
        The Score class is to be used to keep track of the score given to the sequences. The idea is to give a higher
        score to a sequence that seem to be "more related" to to its cluster. That will be determined based on the
        number of sequences in the cluster we consider to be "close" to it (based on the condition used make pairs).
        """

        def __init__(self, n):
            if n <= 0:
                raise ValueError("Invalid number of items")
            self.seen = dict()
            self.values = [0 for _ in range(n)]

        def update(self, elem_1, elem_2):
            """
            Update the score of the elements we are about to assaign to the same cluster.
            :param elem_1, elem_2: the indices of two sequences in all_reads.
            :param scores: array of positive integers, with the grade we give to each sequence. higher it is,
                higher the chance we'll choose it to represent the cluster.
            """
            if (elem_1, elem_2) in self.seen or (elem_2, elem_1) in self.seen:
                return
            self.values[elem_1] += 1
            self.values[elem_2] += 1
            self.seen[(elem_1, elem_2)] = True

    @staticmethod
    def index(seq):
        """
        Get the index out of a given sequence. The convention is to use INDEX_LEN chars index.
        :param seq: the DNA strand. string.
        :return: The first INDEX_LEN chars of the string.
        """
        return seq[:INDEX_LEN]

    def rep_find(self, inp):
        """
        Obtain the representative of the cluster a given sequence is related to.
        In the beginning, for each sequence the "parent" is itself.
        Not using cache as values can be updated as the algorithm progress.
        In addition, updates the items in the path from 'inp' to its topmost ancestor, in order to achieve
        amortized complexity of log*(n) when getting the parent of an element.
        :param inp: the unique index of the sequence
        :return: the parent's index.
        """
        temp = inp
        path = [inp]
        while self.parent[temp] != temp:
            temp = self.parent[temp]
            path.append(temp)
        # update parent for items in path:
        for idx in path:
            self.parent[idx] = temp
        return temp

    @staticmethod
    def qgram_val(sub_seq):
        """
        Calculate the value of a Q-gram
        :param sub_seq: sub string of the original sequence, of length q
        :return: integer, representing the value of the Q-gram
        """
        tot = 0
        for pos in range(len(sub_seq)):
            tot += (4 ** pos) * BASE_VALS[sub_seq[pos]]
        return tot

    @staticmethod
    def jaccard_similarity(numset_1, numset_2):
        """
        Approximate the edit distance of two sequences using Jaccard simillarity.
        :param numset_1, numset_2: two arrays of integers. each one represents one of the sequences we
            we wish to estimate the distance for. The numbers are the value of the Q-grams which the
            sequence consists of. They are obtained using '_numsets' function.
        :return: float, from 0 to 1.
        """
        intersection = len(list(set(numset_1).intersection(numset_2)))
        union = (len(numset_1) + len(numset_2)) - intersection
        return float(intersection) / union

    @staticmethod
    def qgram_distance(numset_1, numset_2):
        """
        Approximate the edit distance of two sequences using Q-gram distance.
        :param numset_1, numset_2: two arrays of integers. each one represents one of the sequences we
            we wish to estimate the distance for. The numbers are the value of the Q-grams which the
            sequence consists of. They are obtained using '_numsets' function.
        :return: float, from 0 to 1.
        """
        union = set()
        for k in numset_1:
            union.add(k)
        for k in numset_2:
            union.add(k)
        dis = 0
        for k in union:
            val_1 = k if k in numset_1 else 0
            val_2 = k if k in numset_2 else 0
            dis += abs(val_1 - val_2)
        return dis

    def seq_numset(self, idx):
        """
        Convert a sequence into a set of numbers
        :param idx: index of the sequence within the 'all_reads' array
        :return: array of integers, each one is the value of a Q-gram
        """
        arr = []
        seq = self.all_reads[idx]
        for idx in range(len(seq) - self.q + 1):
            arr.append(LSHCluster.qgram_val(seq[idx:idx + self.q]))
        return arr

    @staticmethod
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
            lsh.append(min([perm[num] for num in numset]))  # append a MH signature
        return lsh

    @cached_property
    def numsets(self):
        """
        Generate the numbers sets for all the sequences
        :return: a dictionary, mapping a number set for each sequence in the input,
            while the key is the index of the sequence in all_reads
        """
        time_start = time.time()
        res = {}
        for idx in range(len(self.all_reads)):
            res[idx] = self.seq_numset(idx)
        print("time to create number set for each sequence: {}".format(time.time() - time_start))
        return res

    @cached_property
    def lsh_sigs(self):
        """
        Calculate the LSH signature of all the sequences in the input
        """
        # calculate numsets, if not done before
        numsets = self.numsets
        time_start = time.time()
        # generate m permutations
        perms = [np.random.permutation(self.top) for _ in range(self.m)]
        # LSH signature tuple (size m, instead of k, as the original paper suggests) for each sequence
        res = [LSHCluster.lsh_sig(numsets[idx], perms) for idx in range(len(numsets))]
        print("time to create LSH signatures for each sequence: {}".format(time.time() - time_start))
        return res

    def _add_pair(self, elem_1, elem_2):
        """
        "Adding a pair" is interpreted as merging the clusters of the two sequences given. If both are in
        the same cluster already, no effect. Otherwise: the union of the two clusters will have as its "center"
        the minimal parent's index of the two input sequences.
        :param elem_1, elem_2: the indices of two sequences in all_reads.
        """
        p1 = self.rep_find(elem_1)
        p2 = self.rep_find(elem_2)
        if p1 != p2:
            # update clusters:
            center, merged = min(p1, p2), max(p1, p2)
            self.C_til[center].extend(self.C_til[merged])
            self.C_til[merged] = []

            # update parents:
            self.parent[merged] = center

            # update max_score:
            if self.max_score[center][1] < self.max_score[merged][1]:
                self.max_score[center] = self.max_score[merged]
                self.max_score[merged] = tuple()

    def run(self):
        """
        Run the full clustering algorithm: create the number sets for each sequence, then generate a LSH
        signature for each, and finally iterate L times looking for matching pairs, to be inserted to the
        same cluster.
        """
        for itr in range(self.L):
            time_start = time.time()
            pairs = set()
            sigs = []
            buckets = {}

            # choose random k elements of the LSH signature
            if self.rand_subs:
                indexes = random.sample(range(self.m), self.k)
            else:
                indexes = [num for num in range(self.k * itr, self.k * itr + self.k)]
            for idx in range(len(self.all_reads)):
                # represent the sig as a single integer
                sig = sum(int(self.lsh_sigs[idx][indexes[i]]) * (self.top ** i) for i in range(self.k))
                # sigs.append((self.index(self.all_reads[idx]), sig))
                sigs.append(sig)

            # buckets[sig] = [indexes (from all_reads) of (hopefully) similar sequences]
            for i in range(len(self.all_reads)):
                if sigs[i] in buckets:
                    buckets[sigs[i]].append(i)
                else:
                    buckets[sigs[i]] = [i]

            # from each bucket we'll keep pairs. the first element will be announced as center
            for elems in buckets.values():
                if len(elems) <= 1:
                    continue
                for elem in elems[1:]:
                    jac = LSHCluster.jaccard_similarity(self.numsets[elems[0]], self.numsets[elem])
                    # q_dis = LSHCluster.qgram_distance(self.numsets[elems[0]], self.numsets[elem])
                    if jac >= 0.38 or (jac >= 0.25 and
                       edit_dis(LSHCluster.index(self.all_reads[elem]), LSHCluster.index(self.all_reads[elems[0]])) <= 3):
                        pairs.add((elems[0], elem))

            for pair in pairs:
                self.sc.update(pair[0], pair[1])
                self._add_pair(pair[0], pair[1])

            print("time for iteration {} in the algorithm: {}".format(itr + 1, time.time() - time_start))
        return self.C_til

    def index_clstring(self):
        """
        Clustering based on the sequences' index only. Kind of a naive approach to the problem. Recommended only
        as part of a full algorithm and not as a single step.
        """
        time_start = time.time()
        pairs = set()
        buckets = {}

        for idx in range(len(self.all_reads)):
            if LSHCluster.index(self.all_reads[idx]) in buckets:
                buckets[LSHCluster.index(self.all_reads[idx])].append(idx)
            else:
                buckets[LSHCluster.index(self.all_reads[idx])] = [idx]

        # from each bucket we'll keep pairs. the first element will be announced as center
        for elems in buckets.values():
            if len(elems) <= 1:
                continue
            for elem in elems[1:]:
                jac = LSHCluster.jaccard_similarity(self.numsets[elems[0]], self.numsets[elem])
                # q_dis = LSHCluster.qgram_distance(self.numsets[elems[0]], self.numsets[elem])
                if jac >= 0.3:
                    pairs.add((elems[0], elem))

        for pair in pairs:
            self.sc.update(pair[0], pair[1])
            self._add_pair(pair[0], pair[1])

        print("time for index based approach: {}".format(time.time() - time_start))
        return self.C_til

    def relable(self, multi=False):
        """
        Used AFTER we executed 'run'. The purpose is to handle single sequences (meaning, clusters of size 1).
        We'll iterate over those sequences, and look for the cluster whose most highly ranked sequence is similar to
        that single sequence.
        :param multi: determine whether to use multiprocessing.
        :return: the updated C_til
        """
        time_start = time.time()
        singles = [center for center, clstr in self.C_til.items() if len(clstr) == 1]
        if len(singles) == 0:
            return self.C_til
        clstr_reps = [center for center, clstr in self.C_til.items() if len(clstr) > 1]
        if multi:
            with mp.Pool(mp.cpu_count()-1) as pool:
                matches = [pool.apply_async(self.relable_single, args=(single, clstr_reps, ))
                           for single in singles]
                pool.close()
                pool.join()
                matches = [match.get() for match in matches]
        else:
            matches = [self.relable_single(single, clstr_reps) for single in singles]
        for idx in range(len(singles)):
            if matches[idx] is not None:
                self._add_pair(matches[idx], singles[idx])
        print("time for relabeling step: {}".format(time.time() - time_start))
        return self.C_til

    def relable_single(self, single, clstr_reps):
        """
        Does the relabeling of a single sequence. To be used separetely for each unique single sequence.
        :param single: the index of the single sequence we wish to find a cluster for.
        :param clstr_reps: array of the representatives of the clusters having more than 1 element.=
        :return the representative of the cluster to have the given single sequence inserted to.
            if no suitable cluster is to be found, None will be returned.
        """
        for rep in clstr_reps:
            # get the index of the sequence with the highest score (from the current cluster)
            best = self.max_score[rep][0]
            jac = LSHCluster.jaccard_similarity(self.numsets[single], self.numsets[best])
            # q_dis = LSHCluster.qgram_distance(self.numsets[single], self.numsets[best])
            if jac >= 0.2:
                return rep
        return None

    def common_substr_step(self, w=3, r=4, repeats=3):
        """
        Step of clustering the sequences using a sub string shared by several sequences. Will be done via getting a
        random permutation of size 'w', then looking for it inside the input sequences. If exists, we will use a
        subs string of size 'r' starting with it. Otherwise, a prefix of size 'l' will be used instead.
        :param w: size for the permutation we will search for inside the sequences.
        :param r: the size of common sub string.
        :param repeats: number of iterations, as it's an iterative algorithm.
        :return: the updated C_til
        """
        def cmn_substr(x, a, w, l):
            ind = x.find(a)
            if ind == -1:
                ind = 0
            return x[ind:min(len(x), ind + w + l)]
        time_start = time.time()
        for _ in range(repeats):
            a = ''.join(random.choice('ACGT') for _ in range(w))
            common_substr_hash = [0] * len(self.all_reads)
            for idx in range(len(self.all_reads)):
                common_substr_hash[idx] = (idx, cmn_substr(self.all_reads[idx], a, w, r))
            common_substr_hash.sort(key=lambda x: x[1])
            for idx in range(len(common_substr_hash) - 1):
                if common_substr_hash[idx][1] == common_substr_hash[idx + 1][1]:
                    jac = LSHCluster.jaccard_similarity(self.numsets[common_substr_hash[idx][0]],
                                                        self.numsets[common_substr_hash[idx + 1][0]])
                    if jac >= 0.32:
                        self._add_pair(common_substr_hash[idx][0], common_substr_hash[idx + 1][0])
        print("common substr step took: {}".format(time.time() - time_start))
        return self.C_til

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
    for i in clustering.keys():
        if len(clustering[i]) >= 1:
            acrcy += comp_clstrs(clustering[i],
                                 C_dict[rep_in_C(reads_err[clustering[i][0]], C_reps)], gamma, reads_err)
    return acrcy


# Reading The Data

if __name__ == '__main__':
    reads_cl = []  # the whole input
    dataset = r'C:\Users\Adar\Documents\git_repos\yupyter\index11\500\evyat.txt'
    # dataset = r'C:\Users\Adar\Documents\git_repos\yupyter\evyat.txt'
    with open(dataset) as f:
        print("using dataset: {}".format(dataset))
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
    acrcy_dict1 = {}
    acrcy_dict2 = {}
    acrcy_dict3 = {}
    acrcy_dict4 = {}
    acrcy_dict5 = {}
    acrcy_dict6 = {}
    acrcy_dict7 = {}

    time_acrcy_dict = {}
    time_itr_dict = {}
    monitor_acry = False
    begin = time.time()
    lsh = LSHCluster(all_reads=reads_err, q=6, k=3, m=50, L=32)
    C_til = lsh.run()
    # C_til = lsh_clstering(all_reads=reads_err, q=6, k=3, m=50, L=16, rand_subs=False)
    # C_til = naive_clstring(reads_err)
    print("time for whole process: {}".format(time.time() - begin))

    # info regarding the num of single sequences, in contrast to bigger clusters

    if monitor_acry:
        keys = acrcy_dict1.keys()
        values1 = acrcy_dict1.values()
        values2 = acrcy_dict2.values()
        values3 = acrcy_dict3.values()
        values4 = acrcy_dict4.values()
        values5 = acrcy_dict5.values()
        values6 = acrcy_dict6.values()
        values7 = acrcy_dict7.values()

        df = pd.DataFrame()

        df["keys"] = keys
        df["0.6"] = values1
        df["0.7"] = values2
        df["0.8"] = values3
        df["0.9"] = values4
        df["0.95"] = values5
        df["0.99"] = values6
        df["1.0"] = values7

        fig = px.line(df, x=df["keys"], y=['0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '1.0'])
        fig.show()

        keys = time_itr_dict.keys()
        values = time_itr_dict.values()

        px.line(x=keys, y=values)
    else:
        clstrs = dict(filter(lambda elem: len(elem[1]) > 1, C_til.items()))
        singles = [center for center, clstr in C_til.items() if len(clstr) == 1]
        print("num of clsts bigger than 1: {}, num of single seqs: {}".format(len(clstrs), len(singles)))
        acrcy1 = calc_acrcy(C_til, C_dict, C_reps, 0.6, reads_err) / len(reads)
        acrcy2 = calc_acrcy(C_til, C_dict, C_reps, 0.7, reads_err) / len(reads)
        acrcy3 = calc_acrcy(C_til, C_dict, C_reps, 0.8, reads_err) / len(reads)
        acrcy4 = calc_acrcy(C_til, C_dict, C_reps, 0.9, reads_err) / len(reads)
        acrcy5 = calc_acrcy(C_til, C_dict, C_reps, 0.95, reads_err) / len(reads)
        acrcy6 = calc_acrcy(C_til, C_dict, C_reps, 0.99, reads_err) / len(reads)
        acrcy7 = calc_acrcy(C_til, C_dict, C_reps, 1, reads_err) / len(reads)
        print("Accuracy:", acrcy1, acrcy2, acrcy3, acrcy4, acrcy5, acrcy6, acrcy7)
        print("Relabeling 1:")
        C_til = lsh.relable()
        clstrs = dict(filter(lambda elem: len(elem[1]) > 1, C_til.items()))
        singles = [center for center, clstr in C_til.items() if len(clstr) == 1]
        print("num of clsts bigger than 1: {}, num of single seqs: {}".format(len(clstrs), len(singles)))
        acrcy1 = calc_acrcy(C_til, C_dict, C_reps, 0.6, reads_err) / len(reads)
        acrcy2 = calc_acrcy(C_til, C_dict, C_reps, 0.7, reads_err) / len(reads)
        acrcy3 = calc_acrcy(C_til, C_dict, C_reps, 0.8, reads_err) / len(reads)
        acrcy4 = calc_acrcy(C_til, C_dict, C_reps, 0.9, reads_err) / len(reads)
        acrcy5 = calc_acrcy(C_til, C_dict, C_reps, 0.95, reads_err) / len(reads)
        acrcy6 = calc_acrcy(C_til, C_dict, C_reps, 0.99, reads_err) / len(reads)
        acrcy7 = calc_acrcy(C_til, C_dict, C_reps, 1, reads_err) / len(reads)
        print("Accuracy:", acrcy1, acrcy2, acrcy3, acrcy4, acrcy5, acrcy6, acrcy7)
        print("Extra index step:")
        C_til = lsh.index_clstring()
        clstrs = dict(filter(lambda elem: len(elem[1]) > 1, C_til.items()))
        singles = [center for center, clstr in C_til.items() if len(clstr) == 1]
        print("num of clsts bigger than 1: {}, num of single seqs: {}".format(len(clstrs), len(singles)))
        acrcy1 = calc_acrcy(C_til, C_dict, C_reps, 0.6, reads_err) / len(reads)
        acrcy2 = calc_acrcy(C_til, C_dict, C_reps, 0.7, reads_err) / len(reads)
        acrcy3 = calc_acrcy(C_til, C_dict, C_reps, 0.8, reads_err) / len(reads)
        acrcy4 = calc_acrcy(C_til, C_dict, C_reps, 0.9, reads_err) / len(reads)
        acrcy5 = calc_acrcy(C_til, C_dict, C_reps, 0.95, reads_err) / len(reads)
        acrcy6 = calc_acrcy(C_til, C_dict, C_reps, 0.99, reads_err) / len(reads)
        acrcy7 = calc_acrcy(C_til, C_dict, C_reps, 1, reads_err) / len(reads)
        print("Accuracy:", acrcy1, acrcy2, acrcy3, acrcy4, acrcy5, acrcy6, acrcy7)