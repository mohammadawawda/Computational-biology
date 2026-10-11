# Fetal trisomy 21 from shallow cfDNA

Hackathon project, team work. A pipeline that takes shallow whole genome sequencing of cell free DNA (cfDNA) and checks for fetal trisomy 21.

1. read the raw reads and clean them
2. align them to the human genome (hg38) with bwa
3. sort and index the BAM files with samtools
4. count the reads in 100 kb bins
5. compute the GC content of every bin and correct the counts for GC bias
6. compute the share of reads on chromosome 21 for every sample
7. build a baseline (mean and std) from normal samples and give a new sample a Z score

The code was written in Google Colab, so `cfdna_pipeline.py` is the notebook saved as a script. Lines that start with `!` are shell commands, so it's easiest to run it in Colab.
