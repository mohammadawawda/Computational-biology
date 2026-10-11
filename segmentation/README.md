# Signal segmentation

Splits a noisy signal into segments with a constant average, with dynamic programming. Every new segment costs a penalty `p`, and a segment is at most `q` long.

* `segment.py` the segmentation
* `analysis.py` measures the runtime as n, q and the number of segments grow, and saves `runtime_analysis.png`
* `report.pdf` the proofs, handwritten (in Hebrew)

## How to run

```
python segment.py --filepath signal.txt --penalty P --maxlen Q
python analysis.py
```
