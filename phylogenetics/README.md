# Phylogenetic trees and ancestor sequences

`phylogenetics.py` takes aligned DNA sequences and:

* estimates the evolutionary time between every two sequences with the Jukes Cantor model
* builds trees with UPGMA and with neighbor joining, and plots them
* uses Felsenstein's pruning algorithm to compute the likelihood of a tree and rebuild the ancestor sequence

`report.pdf` is our report (in Hebrew).

## How to run

```
python phylogenetics.py --fasta aligned.fa --part PART --tree_type TYPE
python phylogenetics.py --help
```
