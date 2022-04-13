# **********************************
#   Imports & Auxiliary functions
# **********************************
import time
import random
import multiprocessing as mp
from functools import lru_cache
import numpy as np

try:
    from Levenshtein import distance
    print("INFO: using external library edit distance")
    @lru_cache
    def edit_dis(s1, s2):
        return distance(s1, s2)
except ImportError:
    print("INFO: using house made implementation of edit distance")
    @lru_cache
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


# **********************************
#   Globals
# **********************************
INDEX_LEN = 15
NUM_HEIGHEST = 4
ADJ_DIFF_FACTOR = 10
STOP_RELABLE = 0.01     # 1 precent
PRIORITZED = 0
NOT_PRIORITZED = 1
REF_PNTS = 12

# **********************************
#   Main class
# **********************************

class LSHCluster:
    def __init__(self, all_reads, q, k, m, L, debug=False):
        """
        Initiate an object dedicated for clustering the DNA sequences in 'all_reads'.
        The function just makes ready the necessary data structures. For the starting the clustering
        process use the 'run' method.
        :param all_reads: array of strings, each: a DNA sequence from the input
        :param m: size of the LSH signature
        :param q: length of the divided sub-sequences (Q-grams)
        :param k: number of MH signatures in a LSH signature
        :param L: number of iterations of the algorithm
        """
        self.all_reads = all_reads
        self.q = q
        self.k = k
        self.m = m
        self.L = L
        self.top = 4 ** q
        self.jobs = mp.cpu_count() - 1
        print("# CPU's to be used: {}".format(self.jobs))
        self.debug = debug
        self.duration = 0
        # array of clusters: C_til[rep] = [reads assigned to the cluster]
        self.C_til = {idx: [idx] for idx in range(len(all_reads))}

        # array for tracking the sequences with the highest score in the cluster
        self.max_score = [[(idx, 0)] for idx in range(len(all_reads))]

        # array for tracking the scores
        self.score = np.array([0 for _ in range(len(all_reads))])

        # mapping between a sequence's index to it's parent's index
        self.parent = np.array([idx for idx in range(len(all_reads))])

        # calculate singatures upon initializing the object
        self.perms = list()
        self.numsets = dict()
        self._numsets()
        self.lsh_sigs = dict()
        self._lsh_sigs()

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
    
    def _create_numset(self, tasks, results):
        """
        A single number set generation.
        :param tasks: queue with the indices of the sequences (as part of 'all_reads') we want to calculate a number set for.
        :param results: queue for storing the results (the pairs of an index and a number set (represented as a list))
        :return: a string
        """
        def _single_numset(seq, q):
            numset = []
            for idx in range(len(seq) - q + 1):
                sub = seq[idx:idx + q]
                tot = 0
                for pos in range(len(sub)):
                    if sub[pos] == 'C':
                        tot += 4 ** pos
                    elif sub[pos] == 'G':
                        tot += (4 ** pos) * 2
                    elif sub[pos] == 'T':
                        tot += (4 ** pos) * 3
                numset.append(tot)
            return numset
        
        while True:
            seq_idx = tasks.get()
            if seq_idx is None:
                tasks.task_done()
                break
            res = _single_numset(self.all_reads[seq_idx], self.q)
            tasks.task_done()
            results.put((seq_idx, res))
        return

    def _numsets(self):
        """
        Generate the numbers sets for all the sequences. Creates a dictionary, mapping a number set for
        each sequence in the input, while the key is the index of the sequence in all_reads
        """
        time_start = time.time()
        tasks = mp.JoinableQueue()
        results = mp.Queue()
        processes = []
        for _ in range(self.jobs):
            processes.append(mp.Process(target=self._create_numset, args=(tasks, results,)))
        [p.start() for p in processes]
        for idx in range(len(self.all_reads)):
            tasks.put(idx)
        for _ in range(self.jobs):
            tasks.put(None)     # poison pill

        tasks.join()
        for _ in range(len(self.all_reads)):
            idx, numset = results.get()
            self.numsets[idx] = numset
        self.duration += time.time() - time_start
        print("time to create number set for each sequence: {}".format(time.time() - time_start))

    def _create_lsh_sig(self, tasks, results):
        """
        Obtain a LSH signature for a sequence, converted to its representation as a set of numbers
        make use of 'numsets': an array of integers, each one is a Q-gram value (so its length is the
        original sequence's length minus q). make use of self.perms: array of arrays, each: permutation 
        of {0,..., 4**q}. 
        The result is an array of length equal to the nubmer of permutations given. each element is the
        MH signature of the sequence calculated with the permutation with the suitable index. It is inserted
        to the results queue.
        :param tasks: queue with the indices of the sequences (as part of 'all_reads') we want to calculate a signature for.
        :param results: queue for storing the results (the pairs of an index and a LSH signature (represented as a list))
        """
        while True:
            next_idx = tasks.get()
            if next_idx is None:
                tasks.task_done()
                break
            res = []
            for perm in self.perms:
                res.append(min([perm[num] for num in self.numsets[next_idx]]))  # append a MH signature
            tasks.task_done()
            results.put((next_idx, res))
        return

    def _lsh_sigs(self):
        """
        Calculate the LSH signature of all the sequences in the input
        """
        time_start = time.time()
        # generate m permutations.
        vals = [num for num in range(self.top)]
        for _ in range(self.m):
            random.shuffle(vals)
            self.perms.append(vals.copy())
        # LSH signature tuple (size m, instead of k, as the original paper suggests) for each sequence
        tasks, results, processes = mp.JoinableQueue(), mp.Queue(), list()
        for _ in range(self.jobs):
            processes.append(mp.Process(target=self._create_lsh_sig, args=(tasks, results,)))
        [p.start() for p in processes]
        for idx in range(len(self.all_reads)):
            tasks.put(idx)
        for _ in range(self.jobs):
            tasks.put(None)     # poison pill

        tasks.join()
        del self.perms      # no need to keep it
        for _ in range(len(self.all_reads)):
            idx, sig = results.get()
            self.lsh_sigs[idx] = sig
        self.duration += time.time() - time_start
        print("time to create LSH signatures for each sequence: {}".format(time.time() - time_start))

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
                                  self.all_reads[elem_2], real_rep1, real_rep2, self.numsets[elem_1],
                                  self.numsets[elem_2],
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
                    both_max_score[j] = (elem_1, self.score[elem_1])
                    found_1 = True
                elif both_max_score[j][0] == elem_2:
                    both_max_score[j] = (elem_2, self.score[elem_2])
                    found_2 = True
            if not found_1:
                both_max_score.append((elem_1, self.score[elem_1]))
            if not found_2:
                both_max_score.append((elem_2, self.score[elem_2]))

            both_max_score = sorted(both_max_score, reverse=True, key=lambda t: t[1])[:NUM_HEIGHEST]
            self.max_score[merged] = [tuple()]
            self.max_score[center] = both_max_score

    def update_maxscore(self, rep):
        """
        Re-calculate the 'max_score' value for a single given cluster.
        :param rep: the cluster's representative's index.
        """
        clstr_sc = [(seq_ind, self.score[seq_ind]) for seq_ind in self.C_til[rep]]
        self.max_score[rep] = sorted(clstr_sc, reverse=True, key=lambda x: x[1])[:NUM_HEIGHEST]

    def _handle_bucket(self, tasks, results):
        while True:
            sig = tasks.get()
            if sig is None:
                tasks.task_done()
                break
            res = set()
            if len(self.buckets[sig]) > 1:
                bucket_ref_points = random.choices(self.buckets[sig], k=REF_PNTS)
                for ref in bucket_ref_points:
                    for elem in self.buckets[sig]:
                        if elem == ref:
                            continue
                        sd = LSHCluster.sorensen_dice(self.numsets[ref], self.numsets[elem])
                        if sd >= 0.38:
                            res.add((ref, elem))
            tasks.task_done()
            results.put(res)
        return        

    def lsh_clustering(self, iters=0, k=0):
        """
        Run the full clustering algorithm: create the number sets for each sequence, then generate a LSH
        signature for each, and finally iterate L times looking for matching pairs, to be inserted to the
        same cluster.
        :return: the updated C_til
        """
        iters = self.L if iters == 0 else iters
        k = self.k if k == 0 else k
        for itr in range(iters):
            time_start = time.time()
            sigs = []
            self.buckets = {}
            pairs = set()
            # choose random k elements of the LSH signature
            indexes = random.sample(range(self.m), k)
            for idx in range(len(self.all_reads)):
                # represent the sig as a single integer
                sig = sum(int(self.lsh_sigs[idx][indexes[i]]) * (self.top ** i) for i in range(k))
                sigs.append(sig)
            print("{} LSH signatures calculated.".format(time.time() - time_start))

            # buckets[sig] = [indexes (from all_reads) of (hopefully) similar sequences]
            for i in range(len(self.all_reads)):
                if sigs[i] in self.buckets:
                    self.buckets[sigs[i]].append(i)
                else:
                    self.buckets[sigs[i]] = [i]

            tasks, results, processes = mp.JoinableQueue(), mp.Queue(), list()
            for _ in range(self.jobs):
                processes.append(mp.Process(target=self._handle_bucket, args=(tasks, results,)))
            [p.start() for p in processes]
            cnt = 0
            for sig in self.buckets.keys():
                if len(self.buckets[sig]) > 1:
                    cnt += 1
                    tasks.put(sig)
            for _ in range(self.jobs):
                tasks.put(None)     # poison pill

            tasks.join()
            for _ in range(cnt):
                pairs = results.get()
                for pair in pairs:
                    self.score[pair[1]] += 1
                    self.score[pair[0]] += 1
                    self._add_pair(pair[0], pair[1])    
            self.duration += time.time() - time_start
            print("time for iteration {} in the algorithm: {}".format(itr + 1, time.time() - time_start))

        return self.C_til

    def _relable_given_singles(self, tasks, results, r=0):
        """
        Preforms the relabeling process, by one of the worker threads.
        :param tasks: queue with the singles left for scanning a cluster to match them to.
        :param results: queue for storing the results (the pairs of a single and a cluster's representative)
        :param r: we store the the sequences with the best score in a cluster in order, so r=0 represents the
            best one, r=1 the one after it, and so on.
        """
        clstr_reps = [center for center, clstr in sorted(self.C_til.items(), key=lambda x: x[1]) if len(clstr) > 1]
        while True:
            next_single = tasks.get()
            if next_single is None:
                tasks.task_done()
                break
            found = False
            for rep in clstr_reps:
                # get the index of the sequence with the highest score (from the current cluster)
                if len(self.max_score[rep]) > r and len(self.max_score[rep][r]) > 0:
                    best = self.max_score[rep][r][0]
                    sd = LSHCluster.sorensen_dice(self.numsets[next_single], self.numsets[best])
                    if sd >= 0.27 or (sd >= 0.22 and edit_dis(LSHCluster.index(self.all_reads[next_single]),
                                                              LSHCluster.index(self.all_reads[best])) <= 3):
                        # print("insert to queue: {} {}".format(next_single, rep))
                        tasks.task_done()
                        results.put((next_single, rep))
                        found = True
                        break
            if not found:
                tasks.task_done()  # Nothing was chosen for this single
                results.put((next_single, -1))
        return

    def relable(self, r=0):
        """
        Used AFTER we executed 'lsh_clustering'. The purpose is to handle single sequences (meaning, clusters of size 1).
        We'll iterate over those sequences, and look for the cluster whose most highly ranked sequence is similar to
        that single sequence.
        :param r: 0 for the using the sequence with the highest score to represent a cluster. 1 for the second highest,
            and so on. if the cluster is smaller than 'r', we won't use it.
        :return: the precent of relabled singles out of the total number of singles we began with.
        """
        time_start = time.time()
        singles = [center for center, clstr in sorted(self.C_til.items(), key=lambda x: x[1]) if len(clstr) == 1]
        initial_singles = len(singles)
        if initial_singles == 0:
            return 0

        # scanning for pairs will be excuted in paralel
        tasks = mp.JoinableQueue()
        results = mp.Queue()
        processes = []
        for _ in range(self.jobs):
            processes.append(mp.Process(target=self._relable_given_singles, args=(tasks, results, r,)))
        [p.start() for p in processes]
        for single in singles:
            tasks.put(single)
        for _ in range(self.jobs):
            tasks.put(None)     # poison pill

        # inserting the pairs is done ins serial
        tasks.join()
        for _ in range(initial_singles):
            elem_1, elem_2 = results.get()
            if elem_2 == -1:
                continue
            self._add_pair(elem_1, elem_2)
        
        # information for deciding whether to continue to additional iterations
        final_singles = len([1 for clstr in self.C_til.values() if len(clstr) == 1])
        success_rate = float(initial_singles - final_singles) / initial_singles
        self.duration += time.time() - time_start
        print("Time for relabeling step {}: {}. Success rate: {}".format(r, time.time() - time_start, success_rate))
        return success_rate

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
                                       update_maxscore=True)
        self.duration += time.time() - time_start
        print("Common sub-string step took: {}".format(time.time() - time_start))
        return self.C_til

    def run(self, accrcy=True):
        """
        To be used in order to preform the whole algorithm flow.
        :param accrcy: True for printing accuracy results. False otherwise.
        """
        lsh_begin = time.time()
        self.lsh_clustering()
        print("Time for basic LSH clustring step: {}".format(time.time() - lsh_begin))
        if accrcy:
            print_accrcy(self.C_til, C_dict, C_reps, reads_err)
        '''
        print("First common sub-string step:")
        lsh.common_substr_step()
        if accrcy:
            print_accrcy(self.C_til, C_dict, C_reps, reads_err)
        '''
        relable_begin = time.time()
        for r in range(NUM_HEIGHEST):
            print("Relabeling %s:" % r)
            success = lsh.relable(r=r)
            if accrcy:
                print_accrcy(self.C_til, C_dict, C_reps, reads_err)
            if success < STOP_RELABLE:
                break
        print("Time for all the relabeling step: {}".format(time.time() - relable_begin))
        print("Second common sub-string step:")
        lsh.common_substr_step()
        if accrcy:
            print_accrcy(self.C_til, C_dict, C_reps, reads_err)
        print("Total time (include init): {}".format(self.duration))


# **********************************
#   Accuracy Calculation
# **********************************

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


def print_accrcy(C_til, C_dict, C_reps, reads_err):
    clstrs = dict(filter(lambda elem: len(elem[1]) > 1, C_til.items()))
    singles = [center for center, clstr in C_til.items() if len(clstr) == 1]
    size = len(clstrs) + len(singles)
    print("Clusters > 1: {}, Singles: {}".format(len(clstrs), len(singles)))
    print("Accuracy:")
    accrcy = {gamma: calc_acrcy(C_til, C_dict, C_reps, gamma, reads_err) / size
              for gamma in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]}
    [print("{}: {}".format(key, value)) for key, value in accrcy.items()]
    print("*************************************************************")


# **********************************
#   Reading The Data
# **********************************

if __name__ == '__main__':
    reads_cl = []
    dataset = r"./datasets/evyat100000.txt"
    with open(dataset) as f:
        print("Using dataset: {}".format(dataset))
        for line in f:
            reads_cl.append(line.strip())
    cnt = 0
    reads = []
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
    size = len([center for center, clstr in C_dict.items() if len(clstr) > 0])
    print("Input has: {} clusters. True size (neglecting empty clusters): {}".format(len(C_dict), size))
    debug = False
    lsh = LSHCluster(reads_err, q=6, k=3, m=50, L=32, debug=debug)
    lsh.run()
