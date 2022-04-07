# IMPORTS

import time
import random
import multiprocessing as mp

try:
    from functools import cache
except ImportError:
    def cache(func):
        d = dict()

        def outer_func(*args):
            if args not in d:
                d[args] = func(*args)
            return d[args]
        return outer_func

# Global Constants

INDEX_LEN = 11
BASE_VALS = {"A": 0, "C": 1, "G": 2, "T": 3}
NUM_HEIGHEST = 4
ADJ_DIFF_FACTOR = 10

def symmetric(func):
    """
    The decorator is to be used when arguments order doesn't have a significance.
    """
    def outer_func(s1, s2):
        return func(min(s1, s2), max(s1, s2))
    return outer_func


@cache
@symmetric
def edit_dis(s1, s2):
    """
    Fully calculate the edit distance between two sequences. O(n^2) using dynamic programming.
    :param s1, s2: the two strings to get the distance between.
    """
    try:
        import Levenshtein      # pip install python-Lavenshtein
        return Levenshtein.distance(s1, s2)
    except ImportError:
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
    def __init__(self, all_reads, q, k, m, L, rand_subs=True, debug=False):
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
        self.jobs = mp.cpu_count() - 1
        print("number of cpus: {}".format(mp.cpu_count()))
        self.rand_subs = rand_subs
        self.sc = LSHCluster.Score(len(all_reads))
        self.debug = debug
        # array of clusters: C_til[rep] = [reads assigned to the cluster]
        self.C_til = {idx: [idx] for idx in range(len(all_reads))}

        # array for tracking the sequences with the highest score in the cluster
        self.max_score = [[(idx, 0)] for idx in range(len(all_reads))]

        # mapping between a sequence's index to it's parent's index
        self.parent = [idx for idx in range(len(all_reads))]

        self.numsets = self._numsets()
        self.lsh_sigs = self._lsh_sigs()

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
    def sorensen_dice(numset_1, numset_2):
        """
        Approximate the edit distance of two sequences using Sorensen-Dice.
        :param numset_1, numset_2: two arrays of integers. each one represents one of the sequences we
            we wish to estimate the distance for. The numbers are the value of the Q-grams which the
            sequence consists of. They are obtained using '_numsets' function.
        :return: float, from 0 to 1.
        """
        set_1, set_2 = set(numset_1), set(numset_2)
        intersection = len(set_1.intersection(set_2))
        return 2 * float(intersection) / (len(set_1) + len(set_2))

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

    def _numsets(self):
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

    def _lsh_sigs(self):
        """
        Calculate the LSH signature of all the sequences in the input
        """
        # calculate numsets, if not done before
        numsets = self.numsets
        time_start = time.time()
        # generate m permutations
        perms = []
        vals = [num for num in range(self.top)]
        for _ in range(self.m):
            random.shuffle(vals)
            perms.append(vals.copy())
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

            if self.debug:
                real_rep1 = rep_in_C(reads_err[elem_1], C_reps)
                real_rep2 = rep_in_C(reads_err[elem_2], C_reps)
                org_clstr_1 = C_dict[real_rep1]
                org_clstr_2 = C_dict[real_rep2]
                if reads_err[elem_2] not in org_clstr_1 or reads_err[elem_1] not in org_clstr_2:
                    print("wrong merge {} {}.\nseq1: {}\nseq2: {}\nreal_rep 1: {}\nreal_rep 2: {}\n"
                          "numset 1: {}\nnumset 2:{}\nfull sig 1:{}\nfull sig 2:{}\n************\n"
                          .format(min(elem_1, elem_2), max(elem_1, elem_2), self.all_reads[elem_1],
                                  self.all_reads[elem_2], real_rep1, real_rep2,
                                  self.numsets[elem_1], self.numsets[elem_2],
                                  sorted(self.lsh_sigs[elem_1]), sorted(self.lsh_sigs[elem_2])))
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

    def handle_some_buckets(self, buckets):
        # READ ONLY
        pairs = set()
        for elems in buckets.values():
            if len(elems) <= 1:
                continue
            for elem in elems[1:]:
                sd = LSHCluster.sorensen_dice(self.numsets[elems[0]], self.numsets[elem])
                if sd >= 0.38 or (sd >= 0.3 and edit_dis(LSHCluster.index(self.all_reads[elem]),
                                                         LSHCluster.index(self.all_reads[elems[0]])) <= 3):
                    pairs.add((elems[0], elem))
        return pairs

    def lsh_clustering(self, iters=0, k=0):
        """
        Run the full clustering algorithm: create the number sets for each sequence, then generate a LSH
        signature for each, and finally iterate L times looking for matching pairs, to be inserted to the
        same cluster.
        :return: the updated C_til
        """
        iters = self.L if iters == 0 else iters
        for itr in range(iters):
            time_start = time.time()
            pairs = set()
            sigs = []
            buckets = {}

            # choose random k elements of the LSH signature
            k = self.k if k == 0 else k
            indexes = random.sample(range(self.m), k)
            for idx in range(len(self.all_reads)):
                # represent the sig as a single integer
                sig = sum(int(self.lsh_sigs[idx][indexes[i]]) * (self.top ** i) for i in range(k))
                sigs.append(sig)
            print("{} LSH signatures calculated.".format(time.time() - time_start))

            # buckets[sig] = [indexes (from all_reads) of (hopefully) similar sequences]
            for i in range(len(self.all_reads)):
                if sigs[i] in buckets:
                    buckets[sigs[i]].append(i)
                else:
                    buckets[sigs[i]] = [i]

            # from each bucket we'll keep pairs. the first element will be announced as center
            tup_size = int(len(buckets) / self.jobs) + 1
            inputs = tuple(left for left in range(len(buckets)) if left % tup_size == 0)
            pool = mp.Pool(processes=self.jobs)
            bkts = []
            for left in inputs:
                right = min(left + tup_size, len(buckets))
                print("bucket left: {} right: {}".format(left, right))
                bkts.append(dict(list(buckets.items())[left: right]))
            bkts = tuple(bkts)
            sets = pool.map_async(self.handle_some_buckets, bkts)
            pool.close()
            pool.join()
            [pairs.update(s) for s in sets.get()]

            for pair in pairs:
                if self.debug:
                    print("tried merge {} {}. sigs: {} {}"
                          .format(min(pair[0], pair[1]), max(pair[0], pair[1]),
                                  tuple(sorted([self.lsh_sigs[pair[0]][indexes[a]] for a in range(self.k)])),
                                  tuple(sorted([self.lsh_sigs[pair[1]][indexes[a]] for a in range(self.k)]))))
                self.sc.update(pair[0], pair[1])
                self._add_pair(pair[0], pair[1])
            print("time for iteration {} in the algorithm: {}".format(itr + 1, time.time() - time_start))

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
        if not multi:
            for single in singles:
                for rep in clstr_reps:
                    # get the index of the sequence with the highest score (from the current cluster)
                    if len(self.max_score[rep]) > r and len(self.max_score[rep][r]) > 0:
                        best = self.max_score[rep][r][0]
                        sd = LSHCluster.sorensen_dice(self.numsets[single], self.numsets[best])
                        if sd >= 0.27 or (sd >= 0.22 and edit_dis(LSHCluster.index(self.all_reads[single]),
                                                                  LSHCluster.index(self.all_reads[best])) <= 3):
                            self._add_pair(single, rep)
        print("time for relabeling step: {}".format(time.time() - time_start))
        return self.C_til

    def small_clstrs_matching(self):
        """
        create couples out of singles
        """
        y = time.time()
        centers = [(center, len(clstr)) for center, clstr in self.C_til.items() if len(clstr) <= 5]
        centers.sort(key=lambda x: x[1])
        print("small clstrs: {}".format(len(centers)))
        for idx_1 in range(min(len(centers), 1250)):
            center_1 = centers[idx_1][0]
            best_1 = self.max_score[center_1][0][0] if len(self.max_score[center_1]) > 0 and len(
                self.max_score[center_1][0]) > 0 else center_1
            for idx_2 in range(idx_1):
                center_2 = centers[idx_2][0]
                best_2 = self.max_score[center_2][0][0] if len(self.max_score[center_2]) > 0 and len(
                    self.max_score[center_2][0]) > 0 else center_2
                sd = LSHCluster.sorensen_dice(self.numsets[best_1], self.numsets[best_2])
                if sd >= 0.16 or (sd >= 0.12 and edit_dis(LSHCluster.index(self.all_reads[best_1]),
                                                          LSHCluster.index(self.all_reads[best_2])) <= 3):
                    self._add_pair(center_1, center_2)
        print("time for small clstrs: {}".format(time.time() - y))
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
        relevant_centers = [center for center, clstr in self.C_til.items() if 1 <= len(clstr) <= 13]
        if len(relevant_centers) == 0: return self.C_til
        for itr in range(repeats):
            print("itr: %s" % itr)
            a = ''.join(random.choice('ACGT') for _ in range(w))
            singles = [random.choice(self.C_til[center]) for center in relevant_centers if len(self.C_til[center]) >= 1]
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
                    if sd >= 0.18:
                        self._add_pair(common_substr_hash[idx][0], common_substr_hash[idx + 1][0],
                                       update_maxscore=False)
        print("common substr step took: {}".format(time.time() - time_start))
        return self.C_til

    def run(self):
        begin = time.time()
        self.lsh_clustering()
        print_accrcy(self.C_til, C_dict, C_reps, reads_err, size)
        for r in range(NUM_HEIGHEST):
            print("Relabeling %s:" % r)
            lsh.relable(r=r)
            print_accrcy(self.C_til, C_dict, C_reps, reads_err, size)
        print("Onecore clusters:")
        print_accrcy(self.C_til, C_dict, C_reps, reads_err, size)
        print("More Iters:")
        lsh.lsh_clustering(iters=5)
        print_accrcy(self.C_til, C_dict, C_reps, reads_err, size)
        print("Total time: {}".format(time.time() - begin))

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


def comp_clstrs(alg_clstr, org_clstr, gamma, reads_err):
    num_exist = 0
    if len(alg_clstr) > len(org_clstr):
        if debug and int(gamma) >= 1:
            print("cmp failed! clstr too big. len alg_clstr = {}, len org clstr = {}"
                  .format(len(alg_clstr), len(org_clstr)))
            for a in alg_clstr:
                if reads_err[a] not in org_clstr:
                    print("len: {}, numsets: {}, {}".format(len(C_dict[rep_in_C(reads_err[a], C_reps)]),
                                                            lsh.numsets[a], lsh.numsets[alg_clstr[0]]))
        return 0
    for i in range(0, len(alg_clstr)):
        flg_exist = 0
        for j in range(0, len(org_clstr)):
            if reads_err[alg_clstr[i]] == org_clstr[j]:
                flg_exist = 1
                num_exist += 1
                break
        if flg_exist == 0:
            if debug:
                print("wrong clstr! {} supposed to be in clstr of size {}, instead, got: {}".
                      format(reads_err[alg_clstr[i]], len(org_clstr), len(alg_clstr)))
            return 0
    if num_exist < gamma * len(org_clstr):
        if debug:
            print("too small clstr: {}, {}".format(len(org_clstr), len(alg_clstr)))
        return 0
    return 1


def calc_acrcy(clustering, C_dict, C_reps, gamma, reads_err):
    acrcy = 0
    for i in clustering.keys():
        if len(clustering[i]) >= 1:
            acrcy += comp_clstrs(clustering[i],
                                 C_dict[rep_in_C(reads_err[clustering[i][0]], C_reps)], gamma, reads_err)
    return acrcy


def print_accrcy(C_til, C_dict, C_reps, reads_err, size):
    clstrs = dict(filter(lambda elem: len(elem[1]) > 1, C_til.items()))
    singles = [center for center, clstr in C_til.items() if len(clstr) == 1]
    print("Clusters > 1: {}, Singles: {}".format(len(clstrs), len(singles)))
    print("Accuracy:")
    accrcy = {gamma: calc_acrcy(C_til, C_dict, C_reps, gamma, reads_err) / size
              for gamma in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]}
    [print("{}: {}".format(key, value)) for key, value in accrcy.items()]
    print("*************************************************************")


# Reading The Data


if __name__ == '__main__':
    reads_cl = []  # the whole input
    dataset = r"./evyat500.txt"
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
    print("input has {} clusters".format(len(C_dict)))

    # Test the clustering algorithm
    size = len([center for center, clstr in C_dict.items() if len(clstr) > 0])
    print("size: {} total len: {}".format(size, len(reads)))
    debug = False
    lsh = LSHCluster(all_reads=reads_err, q=6, k=3, m=50, L=32, debug=debug)
    lsh.run()
