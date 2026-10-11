import numpy as np
import argparse


def read_signal_file(filepath: str) -> np.ndarray:
    """
    Read signal data from a tab-separated text file.
    Each line contains one or more columns of values separated by tabs.

    Args:
        filepath: Path to the input text file

    Returns:
        np.ndarray: Array of signal values from all columns
    """
    return np.loadtxt(filepath, delimiter='\t', dtype=np.float32)


def print_segments(s: np.ndarray, c: float) -> None:
    """
    Print segments array with 3 decimal places.
    Each row contains [start, end, average].

    Args:
        s: numpy array of shape (n_segments, 3)
        c: Total cost value of optimal segmentation
    """
    for row in s:
        print(f"{int(row[0])} {int(row[1])} {np.round(row[2], 3)}")
    print(np.round(c, 3))
    return

def segment(x: np.ndarray, p: float, q: int):
    """
    Segment a signal using dynamic programming.

    Args:
        x: Input signal (numpy 1d array)
        p: Penalty parameter
        q: Maximum segment length

    Returns:
        s: numpy Array of segments. Shaped (n, 3), each row in format [start, end, average]
        c: Total cost value of optimal segmentation
    """
    n = x.shape[0]

    if n == 0:
        # Empty sequence: no segments, zero cost
        return np.zeros((0, 3)), 0.0

    # we preprocess and compute segment error as a prefix sum algorithms
    prefix_sum = np.zeros(n + 1)
    prefix_sq = np.zeros(n + 1)

    prefix_sum[1:] = np.cumsum(x)
    prefix_sq[1:] = np.cumsum(x * x)

    def segment_error(start, end):
        """
        Sum over t=start... end of the following expression (xt - mean)^2, using the previously defined
        prefix sums.

        note that  0 <= start <= end < n.
        """
        length = end - start + 1
        s = prefix_sum[end + 1] - prefix_sum[start]
        sq = prefix_sq[end + 1] - prefix_sq[start]
        # SSE = sum(x^2) - (sum(x))^2 / length
        return sq - (s * s) / length

    # the dynamic programming:
    #  we assign
    #  C[i]   = optimal cost for x[0 ... i-1] (prefix of length i)
    #  prev[i] = starting index of the last segment in that optimal solution

    C = np.full(n + 1, np.inf)
    prev = np.full(n + 1, -1, dtype=int)
    C[0] = 0.0  # cost of empty prefix

    for i in range(1, n + 1):
        max_len = min(q, i) # sequences cannot be longer than q

        best_cost = np.inf
        best_start = -1

        # Consider all legal segment of length l ending at position i-1
        for l in range(1, max_len + 1):
            start = i - l  # segment is x[start : i-1]
            end = i - 1

            err = segment_error(start, end)

            # we assign penalty equals to 0 if we're dealing with one segment
            # the penalty p is added if we add additional segment( as explained in the forum)
            if start ==0 :
                penalty =0.0
            else :
                penalty=p


            cand_cost = C[start] + err + penalty

            if cand_cost < best_cost:
                best_cost = cand_cost
                best_start = start

        C[i] = best_cost
        prev[i] = best_start

    total_cost = float(C[n])

    # now that we have the optimal solution, we need to traceback to reconstruct the segments from "prev"
    segments = []
    i = n
    while i > 0:
        start = prev[i]
        end = i - 1

        # Compute the mean of this segment
        length = end - start + 1
        s = prefix_sum[end + 1] - prefix_sum[start]
        mean = s / length

        segments.append([start, end, mean])
        i = start  # jump to the start of the previous segment

    # now that we built the segments backwards, we need to change the
    # pointers' directions (reverse them) to get them in order
    segments.reverse()
    segs = np.asarray(segments, dtype=float)

    return segs, total_cost


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process signal data from a text file.')
    parser.add_argument('--filepath', type=str, required=True, help='Path to the input text file')
    parser.add_argument('--penalty', type=float, required=True, help='Penalty parameter')
    parser.add_argument('--maxlen', type=int, required=True, help='Maximum segment length')
    args = parser.parse_args()
    seq = read_signal_file(args.filepath)

    segs, cost = segment(seq, args.penalty, args.maxlen)
    score_sum =0
    for row in segs:
        start_idx = int(row[0]) +1
        end_idx = int(row[1]) +1
        mean_val = float(row[2])
        print(f"{start_idx} {end_idx} {mean_val:.3f}")
    print(f"{cost:.3f}")