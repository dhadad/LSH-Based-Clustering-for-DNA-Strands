# **********************************
#   Imports
# **********************************
import argparse
import time
import random
import multiprocessing as mp
from statistics import mean
import queue
import math
import traceback
from Levenshtein import distance
import platform
import logging
import sys

from simulator import *

# **********************************
#   Globals
# **********************************
BASE_VALS = {"A": 0, "C": 1, "G": 2, "T": 3}
# multiprocessing consts:
CPUS = 2 / 5
QSIZE = 2000000
RESULTS_DELIEVERY = 2500
SLEEP_BEFORE_TRY = 0.03


# **********************************
#   Main class
# **********************************

class LSHBasedCluster:
    def __init__(self,
                 log_file='log.txt',
                 chosen_technology='minion_idt',
                 q=6, k=3, m=40, L=32, distance_threshold=12,
                 report_func=None,
                 reps_per_cluster=5, reps_per_chunk=3, allowed_bad_rounds=4,
                 accrcy=True):
        """
        Initiate an object dedicated for clustering the DNA sequences
        :param chosen_technology: string, synthesizing technology used in generating the errors
        :param m: size of the LSH signature
        :param q: length of the divided sub-sequences (Q-grams)
        :param k: number of MH signatures in a LSH signature
        :param L: number of iterations of the algorithm
        :param reps_per_cluster: representatives to keep for each cluster. uses the score mechanism:
            the sequences with the highest score in the cluster are kept.
        :param rep_per_chunk: similar to 'reps_per_cluster', used for the chunk partitioning part
        :param allowed_bad_rounds: number of bad rounds (rounds with a relatively little amount of work) before
            quitting the procedure (in the final clustering step).
        :param distance_threshold: maximal edit distance between sequences for merging their clusters
            (in case the sorensen dice similarity was sufficient)
        :param accrcy: True for printing accuracy results during the run
        """
        self.L = L
        self.q = q
        self.k = k
        self.m = m
        self.top = 4 ** q  # upper boundary for items in numsets
        self.accrcy = accrcy
        self.distance_threshold = distance_threshold

        # handle logging
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()
        logger.addHandler(logging.FileHandler(log_file, 'w'))
        sys.stdout.write = logger.info

        self.technology = chosen_technology
        if platform.system() == "Linux":
            self.shuffled_file = '/home_nfs/sgamer8/DNAindex' + str(
                self.index) + '/files/' + self.technology + '/' + 'errors_shuffled.txt'
        elif platform.system() == "Windows":
            self.shuffled_file = 'files/' + self.technology + '/' + 'errors_shuffled.txt'
        if platform.system() == "Linux":
            self.evyat_path = '/home_nfs/sgamer8/DNAindex' + str(self.index) + '/files/' + self.technology + '/' + 'evyat.txt'
        elif platform.system() == "Windows":
            self.evyat_path = 'files/' + self.technology + '/' + 'evyat.txt'
        self.temp_evyat_path = self.evyat_path.removesuffix('evyat.txt')
        self.temp_evyat_path += 'temp_evyat.txt'
        
        self.all_reads = []
        self.original_strand_dict = {}  # map from orig strand id to the actual strand
        self.reads_err_original_strand_dict = {}  # map from read_err to it's orig strand id
        self.C_reps = []  # C_reps = [(Read, Cluster rep of the cluster to which the read belongs to)]
        self.C_dict = {}  # C_dict = {Cluster rep: All the Reads that belong to that cluster}
        self.process_input()

        self.tmp_bar = 0
        self.total_bar_size = 0
        # if the input is extremely small, more bad rounds are allowed as they are cheap
        self.allowed_bad_rounds = min(max(math.ceil(float(1) / len(self.all_reads) * (10 ** 7)), allowed_bad_rounds), 1000)
        print("-INFO: amount of allowed bad rounds: {}".format(self.allowed_bad_rounds))
        self.work_in_bad_round = math.ceil(len(self.all_reads) ** (1 / 5))
        print("-INFO: singles handeled in a bad round: {}".format(self.work_in_bad_round))
        self.reps_per_cluster = reps_per_cluster
        self.reps_per_chunk = reps_per_chunk
        self.duration = 0  # sum of the all the calculations
        self.avg_chunk = None
        self.base_qgram = [(4 ** pos) for pos in range(self.q)]

        # array of clusters: C_til[rep] = [reads assigned to the cluster]
        self.C_til = {idx: [idx] for idx in range(len(self.all_reads))}

        # array for tracking the sequences with the highest score in the cluster
        self.max_score = [[(idx, 0)] for idx in range(len(self.all_reads))]

        # array for tracking the scores
        self.score = [0 for _ in range(len(self.all_reads))]

        # mapping between a sequence's index to it's parent's index
        self.parent = [idx for idx in range(len(self.all_reads))]

        # handle first step of the algorithm, partition into chunks
        self.chunks = [[idx] for idx in range(len(self.all_reads))]
        self.chunk_parent = [idx for idx in range(len(self.all_reads))]

        # calculate singatures upon initializing the object
        if platform.system() == "Linux":
            self.numsets, self.lsh_sigs = dict(), dict()
            self._numsets, self._lsh_sigs = self._numsets_mul, self._lsh_sigs_mul
            self.jobs = max(int(mp.cpu_count() * CPUS), 1)
            print("-INFO: CPU's to be used: {}".format(self.jobs + 1))  # 1 for main thread
        else:
            self.numsets, self.lsh_sigs = list(), list()
            self._numsets, self._lsh_sigs = self._numsets_ser, self._lsh_sigs_ser
        self.perms = list()
        self._numsets()
        self._lsh_sigs()

    def process_input(self):
        """
        Set the data structures having the nosiy DNA copies. self.all_reads is the one we use thorugh the
        algorithm. The rest are used for accuracy computation.
        """
        C_rev = {}  # C_rev = {<nosiy_copy>: index in self.all_reads}
        C_org_rev = {}  # C_org_rev = {<line_from_data>: index in self.original_strand_dict}
        strand_id = 0
        rep = None
        with open(self.evyat_path, 'r') as evyat_f:
            print("-INFO: using dataset: {}".format(self.evyat_path))
            prev_line = ''
            for line in evyat_f:
                line = line.strip()
                if line != "":
                    if line[0] == '*':
                        if len(self.C_reps) > 0:
                            self.C_dict[rep].pop()
                            self.C_reps.pop()
                        rep = prev_line
                        self.original_strand_dict.update({strand_id: rep})
                        C_org_rev[rep] = strand_id
                        strand_id += 1
                        self.C_dict[rep] = []
                    elif rep is not None:
                        self.C_dict[rep].append(line)
                        self.C_reps.append((line, rep))
                    prev_line = line
        self.C_reps.sort(key=lambda x: x[0])  # useful in printing accuracy results, using binary search
        for i in range(len(self.C_reps)):
            self.all_reads.append(self.C_reps[i][0])

        random.shuffle(self.all_reads)  # otherwise, the nosiy copies are in the order of the true clusters
        for i in range(len(self.all_reads)):
            C_rev[self.all_reads[i]] = i

        for i in range(len(self.C_reps)):
            self.reads_err_original_strand_dict.update({C_rev[self.C_reps[i][0]]: C_org_rev[self.C_reps[i][1]]})
        del C_org_rev
        del C_rev
        self.read_err_dict = {}
        self.reads_err_ind = [0] * (len(self.all_reads))
        for i in range(0, len(self.all_reads)):
            self.read_err_dict.update({i: self.all_reads[i]})
            self.reads_err_ind[i] = (i, self.all_reads[i])
        print("-INFO: size of the input: {}".format(len(self.all_reads)))

    def rep_find(self, inp, chunks=False):
        """
        Obtain the representative of the cluster a given sequence is related to.
        In the beginning, for each sequence the "parent" is itself.
        In addition, updates the items in the path from 'inp' to its topmost ancestor, in order to achieve
        amortized complexity of log*(n) when getting the parent of an element.
        :param inp: the unique index of the sequence
        :param chunks: boolean. which 'parent' array are we referring to.
        :return: the parent's index.
        """
        parent = self.chunk_parent if chunks else self.parent
        temp = inp
        path = [inp]
        while parent[temp] != temp:
            temp = parent[temp]
            path.append(temp)
        # update parent for items in path:
        for idx in path:
            parent[idx] = temp
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
                tot += self.base_qgram[pos] * BASE_VALS[sub[pos]]
            numset.append(tot)
        return numset

    def _numsets_ser(self):
        """
        Generate the numbers sets for all the sequences. Creates a dictionary, mapping a number set for
        each sequence in the input, while the key is the index of the sequence in all_reads
        Serial Implementation
        """
        time_start = time.time()
        for seq_idx in range(0, len(self.all_reads)):
            self.numsets.append(self._single_numset(seq_idx, self.q))
        self.duration += time.time() - time_start
        print("-INFO: time to create number set for each sequence: {}".format(time.time() - time_start))

    def _lsh_sigs_ser(self):
        """
        Calculate the LSH signatures of all the sequences in the input
        Serial implementation
        """
        time_start = time.time()
        # generate m permutations.
        vals = [num for num in range(self.top)]
        for _ in range(self.m):
            random.shuffle(vals)
            self.perms.append(vals.copy())
        for seq_idx in range(len(self.all_reads)):
            sig = [0] * self.m
            for perm_idx in range(self.m):
                sig[perm_idx] = min(
                    [self.perms[perm_idx][int(num)] for num in self.numsets[seq_idx]])  # append a MH signature
            self.lsh_sigs.append(sig)
        del self.perms
        self.duration += time.time() - time_start
        print("-INFO: time to create LSH signatures for each sequence: {}".format(time.time() - time_start))

    def _numsets_mul(self):
        """
        Generate the numbers sets for all the sequences. Creates a dictionary, mapping a number set for
        each sequence in the input, while the key is the index of the sequence in all_reads
        Multiprocessed implementation
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
                while True:
                    idx, numset = results.get_nowait()
                    self.numsets[idx] = numset
            except queue.Empty:
                pass
            time.sleep(SLEEP_BEFORE_TRY)
            if not results.empty():
                continue
            liveprocs = [p for p in liveprocs if p.is_alive()]  # implicit join
        self.duration += time.time() - time_start
        print("-INFO: time to create number set for each sequence: {}".format(time.time() - time_start))

    def _lsh_sigs_mul(self):
        """
        Calculate the LSH signatures of all the sequences in the input
        Multiprocessed implementation
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
                seq_idx = tasks.get()
                if seq_idx is None:
                    tasks.task_done()
                    break
                res = list()
                for perm in self.perms:
                    res.append(min([perm[num] for num in self.numsets[seq_idx]]))  # append a MH signature
                tasks.task_done()
                results.put((seq_idx, res))
            return

        time_start = time.time()
        # generate m permutations.
        vals = [num for num in range(self.top)]
        for _ in range(self.m):
            random.shuffle(vals)
            self.perms.append(vals.copy())
        # LSH signature tuple (size m, instead of k, as the original paper suggests) for each sequence
        tasks, results, processes = mp.JoinableQueue(), mp.Queue(maxsize=QSIZE), list()
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
                while True:
                    idx, sig = results.get_nowait()
                    self.lsh_sigs[idx] = sig
            except queue.Empty:
                pass
            time.sleep(SLEEP_BEFORE_TRY)
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
            if update_maxscore:
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

                both_max_score = sorted(both_max_score, reverse=True, key=lambda t: t[1])[:self.reps_per_cluster]
                self.max_score[merged] = [tuple()]
                self.max_score[center] = both_max_score

    def chunk_partitioning(self,
                           min_work=0.002,
                           allowed_bad_rounds_first=3,
                           allowed_bad_rounds_second=4):
        """
        Divides the nosiy copies into chunks, hopefully resembling the clusters. uses common sub-strings as a way to
        approximate similiarity between sequences.
        :param min_work: float, work (merges out of the total size of the input) is required to meet this threshold
        :param allowed_bad_rounds_first: integer, number of allowed rounds with low amount of work before moving towards
            different signatures. percentages can drop, so we  give it a another change before giving up
        :param allowed_bad_rounds_second: integer, number allowed rounds with little work before quiting the procedure
        """

        def cmn_substr(x, a, w, t):
            """
            Create the signature, which is a sub-string of the original sequence.
            """
            ind = x.find(a)
            if ind == -1:
                ind = 0
            return x[ind:min(len(x), ind + w + t)]

        time_start = time.time()
        repeats = 64  # upper bound
        cnt_merges = 0
        # params when multi-sig
        w = max(math.ceil(math.log(len(self.all_reads[0]), 4)) - 1, 1)
        t = max(math.ceil(math.log(len(self.all_reads), 4)) - 1, w)
        print("-INFO: (common sub-string) params: w = {}, t = {}, max repeats = {}".format(w, t, repeats))
        cnt_bad_rounds = 0
        multi_sigs = 2
        allowed_bad = allowed_bad_rounds_first
        for itr in range(repeats):
            time_itr = time.time()
            if itr > 0 and float(cnt_merges) / len(self.all_reads) < min_work and cnt_bad_rounds < allowed_bad:
                cnt_bad_rounds += 1
            if itr > 0 and float(cnt_merges) / len(self.all_reads) < min_work and cnt_bad_rounds >= allowed_bad:
                cnt_bad_rounds = 0
                if multi_sigs == 2:
                    multi_sigs = 1
                    allowed_bad = allowed_bad_rounds_second
                    # take longer substrings if we are using only one sig
                    w = math.ceil(math.log(len(self.all_reads[0]), 4))
                    t = max(math.ceil(math.log(len(self.all_reads), 4)), w)
                    print("-INFO: (common sub-string) params: w = {}, t = {}, max repeats = {}".format(w, t, repeats))
                else:
                    print("-INFO: (common sub-string) enoguh bad rounds. stops.")
                    break

            sigs = [''.join(random.choice('ACGT') for _ in range(w)) for _ in range(multi_sigs)]
            cnt_merges = 0
            common_substr_hash = []
            for chunk in self.chunks:
                if len(chunk) == 0:
                    continue
                elif len(chunk) <= self.reps_per_chunk:
                    for idx in chunk:
                        full_sig = sorted(cmn_substr(self.all_reads[idx], a, w, t) for a in sigs)
                        common_substr_hash.append((idx, full_sig))
                else:
                    idxs = random.choices(chunk, k=self.reps_per_chunk)
                    for idx in idxs:
                        full_sig = sorted(cmn_substr(self.all_reads[idx], a, w, t) for a in sigs)
                        common_substr_hash.append((idx, full_sig))

            common_substr_hash.sort(key=lambda x: x[1])
            for idx in range(0, len(common_substr_hash) - 1, 1):
                if common_substr_hash[idx][1] == common_substr_hash[idx + 1][1]:
                    p1 = self.rep_find(common_substr_hash[idx][0], chunks=True)
                    p2 = self.rep_find(common_substr_hash[idx + 1][0], chunks=True)
                    if p1 != p2:
                        cnt_merges += 1
                        center, merged = min(p1, p2), max(p1, p2)
                        self.chunks[center].extend(self.chunks[merged])
                        self.chunks[merged] = []
                        self.chunk_parent[merged] = center
            print("-INFO: (common sub-string) iteration {} took: {} for {} merges - {} with {} sigs"
                  .format(itr, time.time() - time_itr, cnt_merges, float(cnt_merges) / len(self.all_reads), multi_sigs))
        self.duration += time.time() - time_start
        print("-INFO: End Stage: Common sub-string step took: {}".format(time.time() - time_start))

    def clustering_pre_chunk(self, chunk_rep, sd_high=0.32, sd_low=0.28):
        """
        LSH based clustering done in respect to a single chunks.
        :param chunk_rep: integer, the index of the representative of the chunk
        :param sd_high: threshold for sorensen dice similarity from which we merge the clusters.
        :param sd_low: threshold for sorensen dice similarity from which we merge the clusters
            only if the edit distance is low enough
        """
        base = [(self.top ** i) for i in range(self.k)]
        for itr in range(self.L):
            time_start = time.time()
            sigs = list()
            # choose random k elements of the LSH signature
            indexes = random.sample(range(self.m), self.k)
            for idx in self.chunks[chunk_rep]:
                # represent the sig as a single integer
                sig = sum(int(self.lsh_sigs[idx][indexes[i]]) * base[i] for i in range(self.k))
                sigs.append((idx, sig))
            sigs.sort(key=lambda x: x[1])
            for a in range(0, len(sigs) - 1):
                if sigs[a][1] == sigs[a + 1][1]:
                    sd = LSHBasedCluster.sorensen_dice(self.numsets[sigs[a][0]], self.numsets[sigs[a + 1][0]])
                    if sd >= sd_high or \
                            (sd >= sd_low and distance(self.all_reads[sigs[a][0]], self.all_reads[sigs[a + 1][0]]) <= self.distance_threshold):
                        self.score[sigs[a][0]] += 1
                        self.score[sigs[a + 1][0]] += 1
                        self._add_pair(sigs[a][0], sigs[a + 1][0])
            self.duration += time.time() - time_start
            print("-INFO: chunk {}, time for iteration {} in the algorithm: {}"
                  .format(chunk_rep, itr + 1, time.time() - time_start))

    def final_clustering(self,
                         sd_high=0.25,
                         sd_low=0.22,
                         low_work_rate=0.005,
                         high_work_rate=0.03,
                         rounds_before_refresh=8):
        """
        Clustering via LSH signatures. Uses a limited amount of representatives from each cluster. All the clusters
        take part of this stage togather (not chunk dedicated)
        :param sd_high: threshold for sorensen dice similarity from which we merge the clusters.
        :param sd_low: threshold for sorensen dice similarity from which we merge the clusters
            only if the edit distance is low enough
        :param low_work_rate: float, work done in a round (number of singles that were handles, divided by the number of
            singles in the beginning) should be higher than this constant. otherwise, little work was done consider
            replace the representatives (refresh the 'focus' array)
        :param high_work_rate: float, if work done in a round (number of singles that were handles, divided by the
            number of singles in the beginning) is higher than this, then we should refresh the 'focus' array as it's
            not relevant anymore. but in this case we don't relace the 'kind' of the representative (variable r). that
            is, if we used the one with the best-score,  we continue using the representatives with the second-best score
        :param rounds_before_refresh: integer, number of rounds with working rate lower than 'low_work_rate' before
            refreshing
        """
        tot = time.time()
        debug_time = 0
        initial_singles = sum([1 for clstr in self.C_til.values() if len(clstr) == 1])
        singles_round_end = initial_singles
        r = -1
        bad_rounds = 0
        working_rate = 0
        cnt_before_refresh = 0
        focus = list()

        iters_num = max(math.ceil(len(self.all_reads) ** (1 / 2.2)), 300)
        print("-INFO: maximum iterations of the reduced LSH clustring step: {}".format(iters_num))
        for itr in range(iters_num):
            if itr > 0 and itr % 200 == 0 and self.accrcy:  # DEUBG PRINTS
                time_measure = time.time()
                print(self.string_accrcy('old'))
                debug_time += (time.time() - time_measure)
            time_start = time.time()
            sigs = list()
            singles_round_start = sum([1 for clstr in self.C_til.values() if len(clstr) == 1])
            if singles_round_start == 0:
                break

            # reset data structures
            cnt_before_refresh = cnt_before_refresh + 1 if working_rate <= low_work_rate else 0
            refresh = itr % 5 == 0 and working_rate >= high_work_rate
            change_r = itr % 20 == 0 or cnt_before_refresh >= rounds_before_refresh
            if change_r or refresh:
                print("-INFO: refreshing 'focus' array")
                if change_r:
                    r = (r + 1) % self.reps_per_cluster
                focus = [center for center, clstr in self.C_til.items() if len(clstr) == 1]
                clstr_reps = [center for center, clstr in self.C_til.items() if len(clstr) > 1]
                for rep in clstr_reps:
                    if len(self.max_score[rep]) > r and len(self.max_score[rep][r]) > 0:
                        focus.append(self.max_score[rep][r][0])
                cnt_before_refresh = 0
            random.shuffle(focus)

            # choose random k elements of the LSH signature. random.sample faster than np.random.choice for small sizes.
            indexes = random.sample(range(self.m), self.k)
            for idx in focus:
                # represent the sig as a single integer
                sig = sum(int(self.lsh_sigs[idx][indexes[i]]) * (self.top ** i) for i in range(self.k))
                sigs.append((idx, sig))

            sigs.sort(key=lambda x: x[1])
            for a in range(len(sigs) - 1):
                if sigs[a][1] == sigs[a + 1][1]:
                    sd = LSHBasedCluster.sorensen_dice(self.numsets[sigs[a][0]], self.numsets[sigs[a + 1][0]])
                    if sd >= sd_high or \
                            (sd >= sd_low and distance(self.all_reads[sigs[a][0]], self.all_reads[sigs[a + 1][0]]) <= self.distance_threshold):
                        self.score[sigs[a][0]] += 1
                        self.score[sigs[a + 1][0]] += 1
                        self._add_pair(sigs[a][0], sigs[a + 1][0])

            singles_round_end = sum([1 for clstr in self.C_til.values() if len(clstr) == 1])
            working_rate = float(singles_round_start - singles_round_end) / singles_round_start
            print("-INFO: {} | {} s | rate: {} | r={} | first={} | end={} | diff={}"
                  .format(itr + 1, time.time() - time_start, working_rate, r, singles_round_start,
                          singles_round_end, singles_round_start - singles_round_end))
            bad_rounds = bad_rounds + 1 if (singles_round_start - singles_round_end) <= self.work_in_bad_round else 0
            if bad_rounds >= self.allowed_bad_rounds:
                print("-INFO: enough bad rounds in a row, finish secondary LSH step.")
                break

        success_rate = float(initial_singles - singles_round_end) / initial_singles if initial_singles != 0 else 0
        self.duration += time.time() - tot - debug_time
        print(
            "-INFO: time for a 'reduced clustering' stage: {}. Success rate: {}"
                .format(time.time() - tot - debug_time, success_rate))
        return success_rate

    def string_accrcy(self, metric):
        """
        Returns a string describing the accuracy of the clustring at the current state.
        """
        if metric not in ['old', 'gamma', 'absolute']:
            return "wrong metric"
        clstrs = dict(filter(lambda elem: len(elem[1]) > 1, self.C_til.items()))
        singles = [center for center, clstr in self.C_til.items() if len(clstr) == 1]
        size = len(clstrs) + len(singles)
        if size == 0: return
        res = ["Time: {}".format(self.duration), "Accuracy1 ({}):".format(metric),
               "Clusters > 1: {}, Singles: {}".format(len(clstrs), len(singles))]
        ac = Accrcy(metric)
        try:
            accrcy = {gamma: ac.calc(self.C_til, self.C_dict, self.C_reps, gamma, self.all_reads) / size
                      for gamma in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]}
            res.extend([("{}: {}".format(key, value)) for key, value in accrcy.items()])
        except Exception:
            res.append(traceback.format_exc())
        return '\n'.join(res)

    def run(self, resonable_chunk=0.5):
        """
        Preforms the whole algorithm flow.
        :param resonable_chunk: float, fraction of a chunk's size out of the average size for a chunk. minimal size
            in order to preform the clustering algorithm for this chunk.
        """
        self.chunk_partitioning()
        lsh_cls_time = time.time()
        self.avg_chunk = mean(
            [len(self.chunks[idx]) for idx in range(len(self.all_reads)) if len(self.chunks[idx]) >= 1])
        print("-INFO: average chunk size (threshold for basic clustering step): {}".format(self.avg_chunk))
        print("-INFO: chunks to be analyzed: {}".format(sum((1 for chunk_rep in range(len(self.chunks)) if len(
            self.chunks[chunk_rep]) >= resonable_chunk * self.avg_chunk))))
        print("-INFO: chunks to be ignored: {}"
              .format(sum((1 for chunk_rep in range(len(self.chunks)) if
                           0 < len(self.chunks[chunk_rep]) < resonable_chunk * self.avg_chunk))))
        for chunk_rep in range(len(self.chunks)):
            if len(self.chunks[chunk_rep]) >= resonable_chunk * self.avg_chunk:
                time_itr = time.time()
                self.clustering_pre_chunk(chunk_rep)
                print("-INFO: time for chunk with rep {}: {}".format(chunk_rep, time.time() - time_itr))
                print("++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("-INFO: time for work done in chunks: {}".format(time.time() - lsh_cls_time))
        if self.accrcy:
            print(self.string_accrcy('old'))
        print("-INFO: Final Clustering Step:")
        self.final_clustering()
        if self.accrcy:
            print(self.string_accrcy('old'))
        print("-INFO: Total time: {}".format(self.duration))
        clusters = [sorted(x) for x in list(self.C_til.values()) if x != []]
        with open(self.temp_evyat_path, 'w', newline='\n') as temp_f:
            for cluster in clusters:
                orig_strand_candidates = []
                for cluster_element in cluster:
                    orig_strand_candidates.append(self.reads_err_original_strand_dict.get(cluster_element))
                orig_strand_id = max(orig_strand_candidates, key= orig_strand_candidates.count)
                temp_f.write(str(self.original_strand_dict.get(orig_strand_id)) + '\n')
                temp_f.write('*****************************\n')
                for cluster_element in cluster:
                    temp_f.write(self.read_err_dict.get(cluster_element) + '\n')
                temp_f.write('\n\n')

        os.remove(self.evyat_path)
        os.rename(self.temp_evyat_path, self.evyat_path)
        return self.string_accrcy('old')

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


class Accrcy:
    def __init__(self, metric='old'):
        self.metric = metric
        self.cnt_notsufficientlybig_mistake = 0
        self.cnt_falsepos_mistake = 0
        self.cnt_falsepos = 0
        self.cnt_toobig = 0

    def calc(self, clustering, C_dict, C_reps, gamma, reads_err):
        acrcy = 0
        self.cnt_toobig = 0
        self.cnt_falsepos = 0
        self.cnt_falsepos_mistake = 0
        self.cnt_notsufficientlybig_mistake = 0
        for i in clustering.keys():
            if len(clustering[i]) >= 1:
                acrcy += self.comp_clstrs(clustering[i],
                                          C_dict[rep_in_C(reads_err[clustering[i][0]], C_reps)], gamma, reads_err)
        print("-INFO: ACCRCY g={}: {} bad clusters due to false positives (there're {} false positives)."
              " {} due to not being sufficiently big enough. {} because they were too big"
              .format(gamma, self.cnt_falsepos_mistake, self.cnt_falsepos, self.cnt_notsufficientlybig_mistake,
                      self.cnt_toobig))
        return acrcy

    def comp_clstrs(self, alg_clstr, org_clstr, gamma, reads_err):
        true_positives = 0
        min_true = gamma * len(org_clstr)
        if self.metric.lower() == 'new':
            if len(org_clstr) >= 30:
                max_false = 3
            elif 30 >= len(org_clstr) >= 20:
                max_false = 2
            elif len(org_clstr) >= 10:
                max_false = 1
            else:
                max_false = 0
        elif self.metric.lower() == 'gamma':
            max_false = math.ceil((float(1) / gamma - 1) * len(org_clstr))
        else:
            max_false = 0
        if self.metric.lower() == 'old':
            if len(alg_clstr) > len(org_clstr):
                self.cnt_toobig += 1
                return 0
        else:  # gamma, new
            if max_false + len(org_clstr) < len(alg_clstr):
                self.cnt_toobig += 1
                return 0

        for i in range(0, len(alg_clstr)):
            flg_exist = 0
            for j in range(0, len(org_clstr)):
                if reads_err[alg_clstr[i]] == org_clstr[j]:
                    flg_exist = 1
                    true_positives += 1
                    break
            if flg_exist == 0:
                self.cnt_falsepos += 1
                if self.cnt_falsepos > max_false:  # allow limited number of false positives
                    self.cnt_falsepos_mistake += 1
                    return 0
        if true_positives < min_true:
            self.cnt_notsufficientlybig_mistake += 1
            return 0
        return 1
