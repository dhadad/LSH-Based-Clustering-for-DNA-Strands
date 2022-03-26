import numpy as np
import time
import random
import plotly.express as px
import pandas as pd
from functools import cached_property
import multiprocessing as mp
from collections import Counter
from array import array

# Global Constants
INDEX_LEN = 11
BASE_VALS = {"A": 0, "C": 1, "G": 2, "T": 3}
NUM_HEIGHEST = 15
NORMAL_CLSTR_SIZE = 140
ADJ_DIFF_FACTOR = 26

# The LSH Clustering Algorithm


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

        # array for tracking the sequences with the highest score in the cluster
        self.max_score = [[(idx, 0)] for idx in range(len(all_reads))]

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
            Update the score of the elements we are about to assign to the same cluster.
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
    def sorensen_dice(numset_1, numset_2, bag=False):
        """
        Approximate the edit distance of two sequences using Sorensen-Dice.
        :param numset_1, numset_2: two arrays of integers. each one represents one of the sequences we
            we wish to estimate the distance for. The numbers are the value of the Q-grams which the
            sequence consists of. They are obtained using '_numsets' function.
        :param bag: boolean. if True, we'll use bag semantics (allow duplicates). otherwise, we won't pay
            attention to the number of appearances of each item.
        :return: float, from 0 to 1.
        """
        if not bag:
            set_1, set_2 = set(numset_1), set(numset_2)
            intersection = len(set_1.intersection(set_2))
            return 2 * float(intersection) / (len(set_1) + len(set_2))
        else:
            bag_1, bag_2 = Counter(numset_1), Counter(numset_2)
            intersection = len(list((bag_1 & bag_2).elements()))
            return 2 * float(intersection) / (len(numset_1) + len(numset_2))

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

    def _add_pair(self, elem_1, elem_2, update_maxscore=True):
        """
        "Adding a pair" is interpreted as merging the clusters of the two sequences given. If both are in
        the same cluster already, no effect. Otherwise: the union of the two clusters will have as its "center"
        the minimal parent's index of the two input sequences.
        Also, the function is in charge of updating the 'max_score' struct.
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
            if not update_maxscore:
                return
            both_max_score = self.max_score[center] + self.max_score[merged]  # two lists
            found_1, found_2 = False, False
            for j in range(len(both_max_score)):
                if both_max_score[j][0] == elem_1:
                    both_max_score[j] = (elem_1, self.sc.values[elem_1])
                    found_1 = True
                elif both_max_score[j][0] == elem_2:
                    both_max_score[j] = (elem_2, self.sc.values[elem_2])
                    found_2 = True
            if not found_1:
                both_max_score.append((elem_1, self.sc.values[elem_1]))
            if not found_2:
                both_max_score.append((elem_2, self.sc.values[elem_2]))

            both_max_score = sorted(both_max_score, reverse=True, key=lambda t: t[1])[:NUM_HEIGHEST]
            self.max_score[merged] = [tuple()]
            self.max_score[center] = both_max_score

    def update_maxscore(self, rep):
        """
        Re-calculate the 'max_score' value for a single given cluster.
        :param rep: the cluster's representative's index.
        """
        clstr_sc = [(seq_ind, self.sc.values[seq_ind]) for seq_ind in self.C_til[rep]]
        self.max_score[rep] = sorted(clstr_sc, reverse=True, key=lambda x: x[1])[:NUM_HEIGHEST]

    def run(self):
        """
        Run the full clustering algorithm: create the number sets for each sequence, then generate a LSH
        signature for each, and finally iterate L times looking for matching pairs, to be inserted to the
        same cluster.
        :return: the updated C_til
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
                    sd = LSHCluster.sorensen_dice(self.numsets[elems[0]], self.numsets[elem])
                    if sd >= 0.38 or (sd >= 0.3 and edit_dis(LSHCluster.index(self.all_reads[elem]),
                                                             LSHCluster.index(self.all_reads[elems[0]])) <= 3):
                        pairs.add((elems[0], elem))

            for pair in pairs:
                self.sc.update(pair[0], pair[1])
                self._add_pair(pair[0], pair[1])

            print("time for iteration {} in the algorithm: {}".format(itr + 1, time.time() - time_start))

        return self.C_til

    def splitter(self, split_singles=False, update_maxscore=True):
        """
        The method's aim is to go through all the clusters and to split clusters where it is highly likely we
        merged two true clusters into one.
        To be used only after self.run() finished executing.
        :param update_maxscore: boolean, determine whether to update the self.max_score structure. There's no point
            suffering from the overhead if the splitter will be used a last step in the algorithm. Otherwise, you'd
            probably want the self.max_score to store the real values, for having self._add_pair() working properly.
        :return: the updated C_til
        """
        print("Splitter:")
        tot = time.time()
        cnt = 0
        for rep in self.C_til.keys():
            if len(self.C_til[rep]) < 3:
                continue
            # arrange the sequences according to their score, sorted
            best = self.max_score[rep][0][0]
            axis = [(seq_ind, self.sorensen_dice(self.numsets[best], self.numsets[seq_ind]))
                    for seq_ind in C_til[rep] if seq_ind != best]
            axis.sort(key=lambda x: x[1])

            # detect if the axis consist of two separate groups
            avg_diff = float(0)
            for ind in range(len(axis) - 1):
                avg_diff += axis[ind + 1][1] - axis[ind][1]
            avg_diff /= (len(axis) - 1)
            print("avg diff: {}".format(avg_diff))
            max_diff, ind_max_diff = 0, -1
            for ind in range(len(axis) - 1):
                cur_diff = axis[ind + 1][1] - axis[ind][1]
                if cur_diff > max_diff:
                    max_diff = cur_diff
                    ind_max_diff = ind
            print("max diff = {}".format(max_diff))
            if ind_max_diff == -1 or (not split_singles and (ind_max_diff == 0 or ind_max_diff == len(axis) - 2)):
                print("wanted to split a single. avoid")
                continue

            # splitting
            if max_diff > ADJ_DIFF_FACTOR * avg_diff and len(C_til[rep]) >= NORMAL_CLSTR_SIZE:

                # FOR DEBUG: learn about the cluster we split:
                org_clstr = C_dict[rep_in_C(reads_err[self.C_til[rep][0]], C_reps)]
                alg_clstr = C_til[rep]
                print("     cmp failed! clstr too big. len alg_clstr = {}, len org clstr = {}".format(len(alg_clstr), len(org_clstr)))
                for a in alg_clstr:
                    if reads_err[a] not in org_clstr:
                        print("     len: {}".format(len(C_dict[rep_in_C(reads_err[a], C_reps)])))

                cnt += 1
                clstr_1 = [axis[ind][0] for ind in range(ind_max_diff + 1)]
                clstr_2 = [axis[ind][0] for ind in range(ind_max_diff + 1, len(axis))]
                if rep in clstr_1:
                    C_til[rep] = clstr_1
                    C_til[clstr_2[0]] = clstr_2
                    other_rep = clstr_2[0]
                else:
                    C_til[rep] = clstr_2
                    C_til[clstr_1[0]] = clstr_1
                    other_rep = clstr_1[0]
                if update_maxscore:
                    self.update_maxscore(rep)
                    self.update_maxscore(other_rep)

        print("Total time for splitting {} clusters: {}".format(cnt, time.time() - tot))
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
                sd = LSHCluster.sorensen_dice(self.numsets[elems[0]], self.numsets[elem])
                if sd >= 0.42 or (sd >= 0.36 and edit_dis(LSHCluster.index(self.all_reads[elem]),
                                                          LSHCluster.index(self.all_reads[elems[0]])) <= 3):
                    pairs.add((elems[0], elem))

        for pair in pairs:
            self.sc.update(pair[0], pair[1])
            self._add_pair(pair[0], pair[1])

        print("time for index based approach: {}".format(time.time() - time_start))
        return self.C_til

    def relable(self, r=0, multi=False):
        """
        Used AFTER we executed 'run'. The purpose is to handle single sequences (meaning, clusters of size 1).
        We'll iterate over those sequences, and look for the cluster whose most highly ranked sequence is similar to
        that single sequence.
        :param r: 0 for the using the sequence with the highest score to represent a cluster. 1 for the second highest,
            and so on. if the cluster is smaller than 'r', we won't use it.
        :param multi: determine whether to use multiprocessing.
        :return: the updated C_til
        """
        time_start = time.time()
        singles = [center for center, clstr in self.C_til.items() if len(clstr) == 1]
        if len(singles) == 0: return self.C_til
        clstr_reps = [center for center, clstr in self.C_til.items() if len(clstr) > 1]
        for single in singles:
            for rep in clstr_reps:
                # get the index of the sequence with the highest score (from the current cluster)
                if len(self.max_score[rep]) > r and len(self.max_score[rep][r]) > 0:
                    best = self.max_score[rep][r][0]
                    sd = LSHCluster.sorensen_dice(self.numsets[single], self.numsets[best])
                    if sd >= 0.31 or (sd >= 0.28 and edit_dis(LSHCluster.index(self.all_reads[single]),
                                                              LSHCluster.index(self.all_reads[best])) <= 3):
                        self._add_pair(single, rep)
        print("time for relabeling step: {}".format(time.time() - time_start))
        return self.C_til

    def common_substr_step(self, w=3, t=4, repeats=220):
        """
        Step of clustering the sequences using a sub string shared by several sequences. Will be done via getting a
        random permutation of size 'w', then looking for it inside the input sequences. If exists, we will use a
        subs string of size 't' starting with it. Otherwise, a prefix of size 't' will be used instead.
        :param w: size for the permutation we will search for inside the sequences.
        :param t: the size of common sub string.
        :param repeats: number of iterations, as it's an iterative algorithm.
        :return: the updated C_til
        """

        def cmn_substr(x, a, w, t):
            ind = x.find(a)
            if ind == -1:
                ind = 0
            return x[ind:min(len(x), ind + w + t)]

        time_start = time.time()
        singles = [center for center, clstr in self.C_til.items() if len(clstr) == 1]
        if len(singles) == 0: return self.C_til
        for itr in range(repeats):
            print("itr: %s" % itr)
            a = ''.join(random.choice('ACGT') for _ in range(w))
            common_substr_hash = []
            for idx in singles:
                common_substr_hash.append((idx, cmn_substr(self.all_reads[idx], a, w, t)))
            common_substr_hash.sort(key=lambda x: x[1])

            for idx in range(len(common_substr_hash) - 1):
                if common_substr_hash[idx] is None:
                    continue
                elif common_substr_hash[idx][1] == common_substr_hash[idx + 1][1]:
                    sd = LSHCluster.sorensen_dice(self.numsets[common_substr_hash[idx][0]],
                                                  self.numsets[common_substr_hash[idx + 1][0]])
                    if sd >= 0.15:
                        self._add_pair(common_substr_hash[idx][0], common_substr_hash[idx + 1][0], update_maxscore=False)
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


def find_size_in_mine(str, clustering, reads_err):
    for i in clustering.keys():
        if reads_err[i] == str:
            return len(clustering[i])
        for j in clustering[i]:
            if reads_err[j] == str:
                return len(clustering[i])
    return 0


def comp_clstrs(alg_clstr, org_clstr, gamma, reads_err, clustering):
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
                                 C_dict[rep_in_C(reads_err[clustering[i][0]], C_reps)], gamma, reads_err, clustering)
    return acrcy

# Reading The Data


if __name__ == '__main__':
    reads_cl = []  # the whole input
    dataset = r'C:\Users\Adar\Documents\git_repos\yupyter\index11\3000\evyat.txt'
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
    print("time for base process: {}".format(time.time() - begin))

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
        print("*************************************************************")
        for r in range(NUM_HEIGHEST):
            print("Relabeling %s:" % r)
            C_til = lsh.relable(r=r)
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
            print("*************************************************************")
        print("Splitting clusters:")
        C_til = lsh.splitter()
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
        print("*************************************************************")