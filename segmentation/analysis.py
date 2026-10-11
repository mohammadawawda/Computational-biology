import time
import random
import matplotlib.pyplot as plt
import statistics
import numpy as np


try:
    from segment import segment

    print("Successfully imported 'segment' from segment.py")
except ImportError:
    print("Could not find the file 'segment.py'. Using a dummy function for demonstration.")



    def segment(x, p, q):
        # Simulate O(n * q) complexity
        time.sleep((len(x) * q) / 10000000)
        # Return dummy segments (start, end, mean) to calculate k
        num_segments = max(1, int(len(x) / 10))  # Fake segment count
        return np.zeros((num_segments, 3)), 0


# generate the sequences

def generate_sequence(n):
    """Generates a sequence of length n with random floats."""
    return np.random.uniform(0, 100, size=n)



# 3. EXPERIMENT FUNCTIONS

def test_runtime_vs_n():
    """
    test 1: time as a function of n which is the sequence length
    """
    print("Running Experiment 1: Time vs N...")
    n_values = [100, 500, 1000, 2000, 3000, 4000, 5000]  # Adjust based on your speed
    times = []

    fixed_q = 10
    fixed_p = 1.0

    for n in n_values:
        data = generate_sequence(n)

        start_time = time.time()
        segment(data, fixed_p, fixed_q)
        end_time = time.time()

        times.append(end_time - start_time)

    return n_values, times


def test_runtime_vs_q():
    """test 2:
     time as a function of q (max segment length)"""
    print("Running Experiment 2: Time vs q...")
    q_values = [5, 10, 20, 50, 100, 200]
    times = []

    fixed_n = 2000  # Keep N constant
    fixed_p = 1.0

    for q in q_values:
        data = generate_sequence(fixed_n)

        start_time = time.time()
        segment(data, fixed_p, q)
        end_time = time.time()

        times.append(end_time - start_time)

    return q_values, times


def test_runtime_vs_k():
    """
    test 3: time as a function of k (num of segments).
    since we cannot set k directly. We vary the value of the penalty p to influence k.
    """
    print("Running Experiment 3: time vs k (via varying penalty)...")
    # low penalty = many segments (high k), where high penalty = few segments (low k)
    p_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    times = []
    k_values = []

    fixed_n = 2000
    fixed_q = 50

    data = generate_sequence(fixed_n)

    for p in p_values:
        start_time = time.time()
        segments, cost = segment(data, p, fixed_q)
        end_time = time.time()

        # k is the number of segments returned
        k = len(segments)

        times.append(end_time - start_time)
        k_values.append(k)

    # Sort by k for cleaner plotting (since p and k are related)
    if k_values:
        sorted_pairs = sorted(zip(k_values, times))
        k_values, times = zip(*sorted_pairs)

    return k_values, times




# plotting the results
def plot_results():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # --- Graph 1: N vs Time ---
    n_x, n_y = test_runtime_vs_n()
    ax1.plot(n_x, n_y, marker='o', color='b')
    ax1.set_title("Runtime vs Sequence Length (n)")
    ax1.set_xlabel("n (Length)")
    ax1.set_ylabel("Time (seconds)")
    ax1.grid(True)

    # --- Graph 2: Q vs Time ---
    q_x, q_y = test_runtime_vs_q()
    ax2.plot(q_x, q_y, marker='o', color='r')
    ax2.set_title("Runtime vs Max Segment Length (q)")
    ax2.set_xlabel("q (MaxLen)")
    ax2.set_ylabel("Time (seconds)")
    ax2.grid(True)

    # --- Graph 3: K vs Time ---
    k_x, k_y = test_runtime_vs_k()
    ax3.plot(k_x, k_y, marker='o', color='g')
    ax3.set_title("Runtime vs Number of Segments (k)")
    ax3.set_xlabel("k (Count of Segments)")
    ax3.set_ylabel("Time (seconds)")
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig("runtime_analysis.png")
    plt.show()


if __name__ == "__main__":
    plot_results()