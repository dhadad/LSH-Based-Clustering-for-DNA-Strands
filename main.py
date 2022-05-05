#!/usr/bin/env python3
# **********************************
#   Imports & Auxiliary functions
# **********************************
import argparse
import time
import random
import multiprocessing as mp
import queue

try:
    from Levenshtein import distance
    print("-INFO: using external library edit distance")
    def edit_dis(s1, s2):
        return distance(s1, s2)
except ImportError:
    print("-INFO: using our implementation of edit distance")
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
BASE_VALS = {"A": 0, "C": 1, "G": 2, "T": 3}
INDEX_LEN = 15
NUM_HEIGHEST = 5
ADJ_DIFF_FACTOR = 8
STOP_RELABEL = 0.03     # 3 percent
REF_PNTS = 12
QSIZE = 15000000        # 15 million
RESULTS_CHUNK = 3000
WORK_IN_BAD_ROUND = 4
ALLOWED_BAD_ROUNDS = 9
REDUCED_ITERS_FOR_LINE = 2 * 10 ** (-4)


# **********************************
#   Main class
# **********************************

class LSHCluster:
    def __init__(self, all_reads, q, k, m, L):
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
        self.top = 4 ** q  # upper boundary for items in numsets
        self.duration = 0  # sum of the all the calculations
        self.jobs = max(int(mp.cpu_count() * (2/3)), 1)
        print("-INFO: CPU's to be used: {}".format(self.jobs + 1))  # for 1 for main thread
        self.buckets = None

        self.max_reduced_iters = int(REDUCED_ITERS_FOR_LINE * len(self.all_reads))
        print("-INFO: maximum iterations if the reduced LSH clustring step: {}".format(self.max_reduced_iters))

        # array of clusters: C_til[rep] = [reads assigned to the cluster]
        self.C_til = {idx: [idx] for idx in range(len(all_reads))}

        # array for tracking the sequences with the highest score in the cluster
        self.max_score = [[(idx, 0)] for idx in range(len(all_reads))]

        # array for tracking the scores
        self.score = [0 for _ in range(len(all_reads))]

        # mapping between a sequence's index to it's parent's index
        self.parent = [idx for idx in range(len(all_reads))]

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
        return 2 * float(intersection) / (len(numset_1) + len(numset_2))

    def _single_numset(self, seq, q):
        """
        Creates a list of integers from 0 to 4**q representing the sequnce (instead of using alphabet of size 4)
        :param seq: integer, index of a sequnce
        :param q: the size of Q-grams.
        :returns: a list of integers, the numbers set representing the sequnce.
        """
        numset = []
        for idx in range(len(self.all_reads[seq]) - q + 1):
            sub = self.all_reads[seq][idx:idx + q]
            tot = 0
            for pos in range(len(sub)):
                tot += (4 ** pos) * BASE_VALS[sub[pos]]
            numset.append(tot)
        return numset

    def _numsets(self):
        """
        Generate the numbers sets for all the sequences. Creates a dictionary, mapping a number set for
        each sequence in the input, while the key is the index of the sequence in all_reads
        """

        def _create_numset(tasks, results):
            """
            A single number set generation.
            :param tasks: queue with the indices of the sequences (as part of 'all_reads') we want to
                calculate a number set for.
            :param results: queue for storing the results (the pairs of an index and a number set
                (represented as a list))
            """
            while True:
                seq_idx = tasks.get()
                if seq_idx is None:
                    tasks.task_done()
                    break
                res = self._single_numset(seq_idx, self.q)
                tasks.task_done()
                results.put((seq_idx, res))
            return

        time_start = time.time()
        tasks = mp.JoinableQueue()
        results = mp.Queue()
        processes = []
        for _ in range(self.jobs):
            processes.append(mp.Process(target=_create_numset, args=(tasks, results,)))
        [p.start() for p in processes]
        for idx in range(len(self.all_reads)):
            tasks.put(idx)
        for _ in range(self.jobs):
            tasks.put(None)  # poison pill
        liveprocs = list(processes)
        while liveprocs:
            try:
                while 1:
                    idx, numset = results.get_nowait()
                    self.numsets[idx] = numset
            except queue.Empty:
                pass
            if not results.empty():
                continue
            liveprocs = [p for p in liveprocs if p.is_alive()]  # implicit join
        self.duration += time.time() - time_start
        print("-INFO: time to create number set for each sequence: {}".format(time.time() - time_start))

    def _lsh_sigs(self):
        """
        Calculate the LSH signatures of all the sequences in the input.
        """

        def _create_lsh_sig(tasks, results):
            """
            Obtain a LSH signature for a sequence, converted to its representation as a set of numbers
            make use of 'numsets': an array of integers, each one is a Q-gram value (so its length is the
            original sequence's length minus q). make use of self.perms: array of arrays, each: permutation
            of {0,..., 4**q}.
            The result is an array of length equal to the nubmer of permutations given. each element is the
            MH signature of the sequence calculated with the permutation with the suitable index. It is inserted
            to the results queue.
            :param tasks: queue with the indices of the sequences (as part of 'all_reads') we want to calculate
                a signature for.
            :param results: queue for storing the results (the pairs of an index and a LSH signature
                (represented as a list))
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

        time_start = time.time()
        # generate m permutations.
        vals = [num for num in range(self.top)]
        for _ in range(self.m):
            random.shuffle(vals)
            self.perms.append(vals.copy())
        # LSH signature tuple (size m, instead of k, as the original paper suggests) for each sequence
        tasks, results, processes = mp.JoinableQueue(), mp.Queue(), list()
        for _ in range(self.jobs):
            processes.append(mp.Process(target=_create_lsh_sig, args=(tasks, results,)))
        [p.start() for p in processes]
        for idx in range(len(self.all_reads)):
            tasks.put(idx)
        for _ in range(self.jobs):
            tasks.put(None)  # poison pill
        liveprocs = list(processes)
        while liveprocs:
            try:
                while 1:
                    idx, sig = results.get_nowait()
                    self.lsh_sigs[idx] = sig
            except queue.Empty:
                pass
            if not results.empty():
                continue
            liveprocs = [p for p in liveprocs if p.is_alive()]  # implicit join
        del self.perms
        self.duration += time.time() - time_start
        print("-INFO: time to create LSH signatures for each sequence: {}".format(time.time() - time_start))

    def _add_pair(self, elem_1, elem_2, update_maxscore=True):
        """
        "Adding a pair" is interpreted as merging the clusters of the two given sequences. If both are in
        the same cluster already, no effect. Otherwise: the union of the two clusters will have as its "center"
        the minimal parent's index of the two input sequences.
        Also, the function is in charge of updating the 'max_score' struct.
        :param elem_1, elem_2: the indices of two sequences in all_reads.
        :param update_maxscore: update the lists of the items with highest score in the cluster.
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

    def lsh_clustering(self):
        """
        Run the full clustering algorithm: create the number sets for each sequence, then generate a LSH
        signature for each, and finally iterate L times looking for matching pairs, to be inserted to the
        same cluster.
        :return: the updated C_til
        """

        def _handle_bucket(tasks, results):
            """
            The function serves as the target for worker threads created by the 'lsh_clustering' function. Each thread
            is tasked with getting buckets from the queue, and for each bucket look for pairs of sequences satisfying a
            pre-determined condition.
            :param tasks: queue with signatures, serving as the buckets' keys.
            :param results: queue for storing the results (pairs of sequences)
            """
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
                            if sd >= 0.35 or (sd >= 0.3 and edit_dis(LSHCluster.index(self.all_reads[ref]),
                                                                     LSHCluster.index(self.all_reads[elem])) <= 3):
                                res.add((ref, elem))
                                if len(res) > RESULTS_CHUNK:
                                    results.put(res)
                                    res = set()
                tasks.task_done()
                results.put(res)
            return

        for itr in range(self.L):
            time_start = time.time()
            sigs = list()
            self.buckets = dict()
            # choose random k elements of the LSH signature
            indexes = random.sample(range(self.m), self.k)
            for idx in range(len(self.all_reads)):
                # represent the sig as a single integer
                sig = sum(int(self.lsh_sigs[idx][indexes[j]]) * (self.top ** j) for j in range(self.k))
                sigs.append(sig)
            # buckets[sig] = [indexes (from all_reads) of (hopefully) similar sequences]
            for idx in range(len(self.all_reads)):
                if sigs[idx] in self.buckets:
                    self.buckets[sigs[idx]].append(idx)
                else:
                    self.buckets[sigs[idx]] = [idx]
            tasks, results, processes = mp.JoinableQueue(), mp.Queue(maxsize=QSIZE), list()
            for _ in range(self.jobs):
                processes.append(mp.Process(target=_handle_bucket, args=(tasks, results,)))
            [p.start() for p in processes]
            for sig in self.buckets.keys():
                if len(self.buckets[sig]) > 1:
                    tasks.put(sig)
            for _ in range(self.jobs):
                tasks.put(None)  # poison pill

            liveprocs = list(processes)
            while liveprocs:
                try:
                    while 1:
                        pairs = results.get_nowait()
                        for pair in pairs:
                            self.score[pair[1]] += 1
                            self.score[pair[0]] += 1
                            self._add_pair(pair[0], pair[1])
                except queue.Empty:
                    pass
                if not results.empty():
                    continue
                liveprocs = [p for p in liveprocs if p.is_alive()]  # implicit join
            self.duration += time.time() - time_start
            print("-INFO: time for iteration {} in the algorithm: {}".format(itr + 1, time.time() - time_start))

    def lsh_clustering_new_draft(self):
        """
        Different approach for the main LSH clustering step. Mimics some of the ideas used in the reduced step. 
        Sort according to signatures instead of literally mapping to buckets. Also, compare adjacent sequences, 
        and not the whole bucket to a subset of sequnces with a high score.
        Improves speed in the cost of accuracy.
        :note: currently not in use.
        """
        for itr in range(self.L):
            time_start = time.time()
            sigs = list()
            # choose random k elements of the LSH signature
            indexes = random.sample(range(self.m), self.k)
            for idx in range(len(self.all_reads)):
                # represent the sig as a single integer
                sig = sum(int(self.lsh_sigs[idx][indexes[i]]) * (self.top ** i) for i in range(self.k))
                sigs.append((idx, sig))

            sigs.sort(key=lambda x: x[1])
            for a in range(len(sigs) - 1):
                if sigs[a][1] == sigs[a + 1][1]:
                    sd = LSHCluster.sorensen_dice(self.numsets[sigs[a][0]], self.numsets[sigs[a + 1][0]])
                    if sd >= 0.36 or (sd >= 0.3 and edit_dis(LSHCluster.index(self.all_reads[sigs[a][0]]),
                                                             LSHCluster.index(self.all_reads[sigs[a + 1][0]])) <= 3):
                        self.score[sigs[a][0]] += 1
                        self.score[sigs[a + 1][0]] += 1
                        self._add_pair(sigs[a][0], sigs[a + 1][0])

            self.duration += time.time() - time_start
            print("-INFO: time for iteration {} in the algorithm: {}".format(itr + 1, time.time() - time_start))

    def reduced_clustering(self):
        """
        Continue the clustering procedure by mimicing the flow of 'lsh_clustering', centering on the tackling singles.
        For this end, instead of iterating over all the sequences, we'll focus on singles and on constant number of
        representatives from the other clusters, thus having much shorter iterations, allowing more repeats.
        :return: the percent of relabeled singles out of the total number of singles we began with.
        :note: the implementation of this step is purely serial, as multiprocessing's overhead seemed too costly.
        """
        tot = time.time()
        initial_singles = sum([1 for clstr in self.C_til.values() if len(clstr) == 1])
        singles_round_end = initial_singles
        r = -1
        bad_rounds = 0
        for itr in range(self.max_reduced_iters):
            time_start = time.time()
            sigs = list()
            singles_round_start = sum([1 for clstr in self.C_til.values() if len(clstr) == 1])
            if singles_round_start == 0:
                break
            # reset data structures every 10 iterations
            if itr % 10 == 0:
                r = (r + 1) % NUM_HEIGHEST
                focus = [center for center, clstr in self.C_til.items() if len(clstr) == 1]
                if singles_round_start == 0:
                    break
                clstr_reps = [center for center, clstr in self.C_til.items() if len(clstr) > 1]
                for rep in clstr_reps:
                    if len(self.max_score[rep]) > r and len(self.max_score[rep][r]) > 0:
                        focus.append(self.max_score[rep][r][0])
            random.shuffle(focus)

            # choose random k elements of the LSH signature
            indexes = random.sample(range(self.m), self.k)
            for idx in focus:
                # represent the sig as a single integer
                sig = sum(int(self.lsh_sigs[idx][indexes[i]]) * (self.top ** i) for i in range(self.k))
                sigs.append((idx, sig))

            sigs.sort(key=lambda x: x[1])
            for a in range(len(sigs) - 1):
                if sigs[a][1] == sigs[a + 1][1]:
                    sd = LSHCluster.sorensen_dice(self.numsets[sigs[a][0]], self.numsets[sigs[a + 1][0]])
                    if sd >= 0.25 or (sd >= 0.22 and edit_dis(LSHCluster.index(self.all_reads[sigs[a][0]]),
                                                              LSHCluster.index(self.all_reads[sigs[a + 1][0]])) <= 3):
                        self.score[sigs[a][0]] += 1
                        self.score[sigs[a + 1][0]] += 1
                        self._add_pair(sigs[a][0], sigs[a + 1][0])
            del sigs

            singles_round_end = sum([1 for clstr in self.C_til.values() if len(clstr) == 1])
            working_rate = float(singles_round_start - singles_round_end) / singles_round_start
            print("-INFO: {} | {} s | rate: {} | r={} | first={} | end={} | diff={}".format(itr + 1,
                  time.time() - time_start, working_rate, r, singles_round_start, singles_round_end,
                    singles_round_start - singles_round_end))
            if singles_round_start - singles_round_end <= WORK_IN_BAD_ROUND:
                bad_rounds += 1
            else:
                bad_rounds = 0
            if bad_rounds >= ALLOWED_BAD_ROUNDS:
                print("-INFO: enough bad rounds in a row, finish secondary LSH step.")
                break

        success_rate = float(initial_singles - singles_round_end) / initial_singles if initial_singles != 0 else 0
        self.duration += time.time() - tot
        print("-INFO: time for a 'reduced clustering' stage: {}. Success rate: {}".format(time.time() - tot, success_rate))
        return success_rate

    def relabel(self, r=0):
        """
        Used AFTER we executed 'lsh_clustering'. The purpose is to handle single sequences (meaning, clusters of
        size 1). We'll iterate over those sequences, and look for the cluster whose most highly ranked sequence is
        similar to that single sequence.
        :param r: 0 for the using the sequence with the highest score to represent a cluster. 1 for the second highest,
            and so on. if the cluster is smaller than 'r', we won't use it.
        :return: the precent of relabled singles out of the total number of singles we began with.
        :note: currently not in use, as it becomes quite costly for large datasets
        """

        def _relabel_given_singles(tasks, results, r=0):
            """
            Preforms the relabeling process, by one of the worker threads.
            :param tasks: queue with the singles left for scanning a cluster to match them to.
            :param results: queue for storing the results (the pairs of a single and a cluster's representative)
            :param r: we store the the sequences with the best score in a cluster in order, so r=0 represents the
                best one, r=1 the one after it, and so on.
            """
            clstr_reps = [center for center, clstr in sorted(self.C_til.items(), key=lambda x: x[1], reverse=True) if
                          len(clstr) > 1]
            res = set()
            while True:
                next_single = tasks.get()
                if next_single is None:
                    if len(res) > 0:
                        results.put(res)
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
                            tasks.task_done()
                            res.add((next_single, rep))
                            if len(res) > RESULTS_CHUNK:
                                results.put(res)
                                res = set()
                            found = True
                            break
                if not found:
                    tasks.task_done()  # Nothing was chosen for this single
            return

        time_start = time.time()
        singles = [center for center, clstr in self.C_til.items() if len(clstr) == 1]
        initial_singles = len(singles)
        if initial_singles == 0:
            return 0

        # scanning for pairs will be executed in parallel
        tasks, results, processes = mp.JoinableQueue(), mp.Queue(maxsize=QSIZE), list()
        for _ in range(self.jobs):
            processes.append(mp.Process(target=_relabel_given_singles, args=(tasks, results, r,)))
        [p.start() for p in processes]
        for single in singles:
            tasks.put(single)
        for _ in range(self.jobs):
            tasks.put(None)  # poison pill

        liveprocs = list(processes)
        while liveprocs:
            try:
                while 1:
                    pairs = results.get_nowait()
                    for pair in pairs:
                        self.score[pair[1]] += 1
                        self.score[pair[0]] += 1
                        self._add_pair(pair[1], pair[0])
            except queue.Empty:
                pass
            if not results.empty():
                continue
            liveprocs = [p for p in liveprocs if p.is_alive()]  # implicit join

        # information for deciding whether to continue to additional iterations
        final_singles = len([1 for clstr in self.C_til.values() if len(clstr) == 1])
        success_rate = float(initial_singles - final_singles) / initial_singles
        self.duration += time.time() - time_start
        print("-INFO: Time for relabeling step {}: {}. Success rate: {}".format(r, time.time() - time_start, success_rate))
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
        :note: currently not in use, as it doesn't provide any advantages in compare to the reduced LSH clustering step.
        """

        def cmn_substr(x, a, w, t):
            ind = x.find(a)
            if ind == -1:
                ind = 0
            return x[ind:min(len(x), ind + w + t)]

        time_start = time.time()
        relevant_centers = [center for center, clstr in self.C_til.items() if 1 <= len(clstr) <= 13]
        if len(relevant_centers) == 0: return self.C_til
        for _ in range(repeats):
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
        print("-INFO: Common sub-string step took: {}".format(time.time() - time_start))
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
        print("Reduced clustering step:")
        self.reduced_clustering()
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

def handle_args():
    parser = argparse.ArgumentParser(description='DNA strands clustering tool using LSH-based approach.')
    parser.add_argument('-v', '--evyat', type=str, help="evyat file (as generated by the DNA simulator). "
                                                  "useful for testing the tool's accuracy")
    parser.add_argument('-s', '--errors_shuffled', type=str, help="errors_shuffled file (as generated by the DNA simulator)")
    args = parser.parse_args()
    oracle_mode = False if args.evyat is None else True
    if oracle_mode:
        if args.errors_shuffled:
            print("-INFO: only evyat file will be used")
        path = args.evyat
    else:
        if args.errors_shuffled:
            path = args.errors_shuffled
        else:
            print("-ERR: cannot proceed without a path to the input file. exiting")
            exit(1)
    return oracle_mode, path


if __name__ == '__main__':
    reads_cl = []
    oracle, dataset = handle_args()
    with open(dataset) as file:
        print("-INFO: using dataset: {}".format(dataset))
        for line in file:
            reads_cl.append(line.strip())
    reads = []
    for i in range(len(reads_cl)):
        if reads_cl[i] != "":
            if reads_cl[i][0] == "*":
                rep = reads_cl[i - 1]
                reads.append(rep)

    # construct the setup for a run
    # C_reps = [(Read, Cluster rep of the cluster to which the read belongs to)]
    # C_dict = {Cluster rep: All the Reads that belong to that cluster}
    C_reps = []
    C_dict = {}
    if oracle:
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
        for i in range(len(C_reps)):
            reads_err[i] = C_reps[i][0]
    else:
        reads_err = reads_cl
    random.shuffle(reads_err)

    # test the clustering algorithm
    size = len([center for center, clstr in C_dict.items() if len(clstr) > 0])
    singles_num = len([1 for _, clstr in C_dict.items() if len(clstr) == 1])
    print("-INFO: input has: {} clusters. True size (neglecting empty clusters): {}".format(len(C_dict), size))
    print("-INFO: out of them: {} are singles.".format(singles_num))
    lsh = LSHCluster(reads_err, q=6, k=3, m=40, L=26)
    lsh.run(accrcy=oracle)
