# HMM, EM and CpG islands

* `part1.py` a two state HMM (fair coin and loaded coin) trained with Baum Welch (EM) on coin toss sequences. It compares models and plots the log likelihood curves.
* `part2.py` a two state HMM that finds CpG islands in DNA. It learns its parameters from labeled sequences and then marks the CpG islands in new sequences.
* `presentation.pdf` our presentation (in Hebrew)

## How to run

part1 reads `data/q_coins.seq.fa`. part2 trains on `data/CpG-islands.2K.seq.fa` and `data/CpG-islands.2K.lbl.fa`. These data files are not included.

```
python part1.py
python part2.py --fasta_path input.fa --output_file output.fa.gz
```
