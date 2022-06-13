# LSH Based Clustering for DNA Strands

## Background
An implementation for a DNA strands clustering algorithm, using LSH signatures. Ideas from previous works by [1], [2] are used throughout the algorithm's flow, altogether with several adjustments and new ideas. The algorithm aims to obtain a partition of a pool of DNA sequences into sets of similar strings, which are highly likely to orignate from the
same line in the dataset, when the synthesis was preformed.

## Input
The tool receives an 'evyat.txt' text file, in the following format: 
```
<original data string #1> 
*****************************
<erroneous copies of the above string>

(two blank line)
<original data string #2> 
etc...
```
For example:
```
CGAGGTTGTGGGCTTCTGTATATGCTCCAATGAAAGTGCCAACTTTTCATAGTCTTCCCAGTGGATTCAATGACGACATCGCACACATACCGCAGTGCGGAAGGCCTAG
*****************************
CGAGGGTTATGGGCTTCTGTATATGCCTCCAATGAACAGTGCCAACTTTTCATAGTCTTCCCAGTGGATATGACGACATCGCACACATACCGCAGTGCGGAAGGCCTAG
CGAGGTTGTGGGGCTTCTGTATATGCTCCAATGAAACCAACTTTTACATAGTCTTCCCAGTGGATTAAATGACGACATCGCACACATACCGCAGTGGCGGATAGGCCTAG
CGAGGGTTGTGTGGCCTTCTGTATATGCCAATGAAAGTGCCAACTTTTCATAGTCTTCCCAGTGGATTCAATGACGACATCGCACACATACCGCAGTGCGGAAGGCCTAG


TCATCAGTGTTAAAATCTTGTGTAGGCAGACGCTTCCTGGAAAACCCGTCCTGGGTATACACAACGGTATGTACACTCTAAGAATTGGTTGCCACTGCGCACTTCTAGG
*****************************
TCATCAGTAGCTAAATCTTGTGTAGGCAGACGCTTCCTGGAAAAACCCGTCCTGGGTATACATCAACGGTATGTACACTTTACGAATTAGTTGCCACTGCGCACTTCTAGG
TCATCAGGTGTTAAAATCTTGTGGCAGGCAGACGCTTCCTGGAAAACCCGTCCTGGGTATACACAACCGGTATGTACACTCTAAGATATTGGTTGCCACTGCGCACTTCTAGG
TCATCATTGTTAAAATCTTGTAGTAGGCAGACGCTTCCTGGAAAACCCGTCCTGGGTATACACAACGGTTATGTACACTCTAAGAATATGGTTGCCACATGCGCACTTCTAGG
```
The file consists of noisy copies generated from the original data. In other words, the sequences *after* the synthesis.
For our purposes, as we aim to test the clustering result, the file is expeted to include the parition to clusters beforehand, so it can be later compared to our result, using different metrics.

## Output
After the clustering successfully finishes, the following output is printed to the standard output in the following structure:
```
Total time: 91.63022971153259
Total Clusters: 5049, Singles: 317
Metric Accrcy:
0.6: 0.9683105565458506
0.7: 0.9651416122004357
0.8: 0.9629629629629629
0.9: 0.9582095464448406
0.95: 0.9510794216676569
0.99: 0.9425628837393544
1.0: 0.9425628837393544
Metric FalsePos:
Total num. of strands: 75009
(FP) False Positives: 0
(TN) True Negatives: 75009
(FN) False Negatives: 0
(TP) True Positives: 75009
(TS) Threat Score / (CSI) Critical Success Index: 1.0
Metric NMI:
0.9303673018393045
Metric RandIndex:
0.5803458414377837
```
Few metrics are used in order to evaulate the result.
* Accuracy: defined in [1]
* FalsePos: refer to [Sensitivity and specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
* NMI: refer to [Normalized Mutual Info](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score)
* RandIndex: refer to [Adjusted Rand index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)

## How-To
```
usage: lsh_based_clustering.py [-h] -e EVYAT
lsh_based_clustering.py: error: the following arguments are required: -e/--evyat
```
The log is printed to the standard output as a default, can be turned off manually. Pay attention in case you redirect the output to a file, since longer runs may result with a heavier log file.

## References
[1] C. Rashtchian et al. “Clustering billions of reads for DNA data storage,” Advances in Neural Information Processing Systems, vol. 30, 2017.\
[2] P. L. Antkowiak, J. Lietard, M. Z. Darestani et al. ”Low cost DNA data storage using photolithographic synthesis and advanced information reconstruction and error correction,” Nature Communications, vol. 11, 2020.

## Note - DNAsimulator
(Refers to [DNASimulator](https://github.com/gadihh/DNASimulator))\
For possible future usages, if the tool is to be assimilated in the DNAsimulator, the needed files are attached in 'dna_simulator.zip' file. \
Changes to the original DNAsimulator code:
* Modified:
	- \DNASimulator\dnaSimulator\app.py
	- \DNASimulator\dnaSimulator\dnaSimulator_ui2.py
* Added:
	- \DNASimulator\dnaSimulator\lsh_based_clustering.py

Notes:
- Multiprocessing not supported in the Windows version
- Progress bar is currently unreponsive.
