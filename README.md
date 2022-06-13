# LSH Based Clustering for DNA Strands

## Background
An implementation for a DNA strands clustering algorithm, using LSH signatures. Ideas from previous works by [1], [2] are used throughout the algorithm's flow, altogether with several adjustments and new ideas. The algorithm aims to obtain a partition of a pool of DNA sequences into sets of similar strings, which are highly likely to orignate from the
same line in the dataset, when the synthesis was preformed.

## Input:
The tool receives an 'evyat.txt' text file, in the following format: 
```
<original data string #1> 

<erroneous copies of the above string>
//two blank line
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

## How-To
```
usage: lsh_based_clustering.py [-h] -e EVYAT
lsh_based_clustering.py: error: the following arguments are required: -e/--evyat
```
The log is printed to the standard output as a default.

## References
[1] C. Rashtchian et al. “Clustering billions of reads for DNA data storage,” Advances in Neural Information Processing Systems, vol. 30, 2017.
[2] P. L. Antkowiak, J. Lietard, M. Z. Darestani et al. ”Low cost DNA data storage using photolithographic synthesis and advanced information
reconstruction and error correction,” Nature Communications, vol. 11, 2020.

## Note - DNAsimulator
For possible future usages, if the tool is to be assimilated in the DNAsimulator, the needed files are attached in 'dna_simulator.zip' file.
Changes to the original DNAsimulator code:
Modified:
\DNASimulator\dnaSimulator\app.py
\DNASimulator\dnaSimulator\dnaSimulator_ui2.py
Added:
\DNASimulator\dnaSimulator\lsh_based_clustering.py
Notes:
Multiprocessing not supported in the Windows version
Progress bar currently unreponsive.
