import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# =====================================================
#                 FASTA SEQUENCE READER
# =====================================================
def read_fasta_sequences(path):
    """
    Reads FASTA sequences into a list of strings.
    Each sequence consists of 'H' and 'T' characters.
    """
    seqs = []
    with open(path) as f:
        current = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    seqs.append("".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            seqs.append("".join(current))
    return seqs


# =====================================================
#                   HMM_EM CLASS
# =====================================================
class HMM_EM:
    """
    Two-state HMM with Baum-Welch (EM) training.
    States must be provided, e.g. ["Fair","Loaded"].
    Observations must be provided, e.g. ["H","T"].
    """

    def __init__(self, states, observations):
        self.states = states
        self.observations = observations

        N = len(states)
        M = len(observations)

        # Initial state distribution
        self.initial = np.array([0.5, 0.5])

        # Random transition matrix
        self.transition = np.random.rand(N, N)
        self.transition /= self.transition.sum(axis=1, keepdims=True)

        # Random emission matrix
        self.emission = np.random.rand(N, M)
        self.emission /= self.emission.sum(axis=1, keepdims=True)

        self.obs_index = {o: i for i, o in enumerate(observations)}
        self.log_likelihoods = []

        print("\n> NEW MODEL INITIALIZATION")
        print("Initial:\n", self.initial)
        print("Transition:\n", self.transition)
        print("Emission:\n", self.emission)

    # =====================================================
    #                   FORWARD ALGORITHM
    # =====================================================
    def forward(self, obs_seq):
        T = len(obs_seq)
        N = len(self.states)

        alpha = np.zeros((T, N))
        alpha[0, :] = self.initial * self.emission[:, self.obs_index[obs_seq[0]]]

        for t in range(1, T):
            ot = self.obs_index[obs_seq[t]]
            alpha[t, :] = self.emission[:, ot] * (alpha[t-1, :] @ self.transition)

        return alpha

    # =====================================================
    #                   BACKWARD ALGORITHM
    # =====================================================
    def backward(self, obs_seq):
        T = len(obs_seq)
        N = len(self.states)

        beta = np.zeros((T, N))
        beta[-1, :] = 1.0

        for t in range(T-2, -1, -1):
            ot1 = self.obs_index[obs_seq[t+1]]
            temp = self.emission[:, ot1] * beta[t+1, :]
            beta[t, :] = self.transition @ temp

        return beta

    # =====================================================
    #     SEQUENCE LOG-LIKELIHOOD USING FORWARD
    # =====================================================
    def sequence_log_likelihood(self, seq):
        alpha = self.forward(seq)
        prob = max(alpha[-1, :].sum(), 1e-300)
        return np.log(prob)

    # =====================================================
    #              BAUM-WELCH (EM TRAINING)
    # =====================================================
    def baum_welch(self, seqs, max_iter=150, tol=1e-6):
        N = len(self.states)
        M = len(self.observations)

        prev_ll = -np.inf

        for it in range(max_iter):

            init_counts = np.zeros(N)
            trans_counts = np.zeros((N, N))
            trans_total_out = np.zeros(N)
            emit_counts = np.zeros((N, M))
            emit_totals = np.zeros(N)

            total_ll = 0.0

            # ------------------------- E-STEP -------------------------
            for seq in seqs:
                alpha = self.forward(seq)
                beta = self.backward(seq)

                prob = max(alpha[-1, :].sum(), 1e-300)
                total_ll += np.log(prob)
                gamma = (alpha * beta) / prob

                # initial
                init_counts += gamma[0, :]

                # emissions
                for t, symbol in enumerate(seq):
                    emit_counts[:, self.obs_index[symbol]] += gamma[t, :]
                emit_totals += gamma.sum(axis=0)

                # transitions
                T = len(seq)
                for t in range(T - 1):
                    ot1 = self.obs_index[seq[t+1]]
                    temp_j = self.emission[:, ot1] * beta[t+1, :]
                    xi_t = self.transition * (
                        alpha[t, :].reshape(N, 1) * temp_j.reshape(1, N)
                    )
                    xi_t /= prob
                    trans_counts += xi_t

                trans_total_out += gamma[:-1, :].sum(axis=0)

            self.log_likelihoods.append(total_ll)
            print(f"Iteration {it+1} | LL = {total_ll:.6f}")

            # check convergence
            if total_ll - prev_ll < tol:
                print("Converged.")
                break
            prev_ll = total_ll

            # ------------------------- M-STEP -------------------------
            self.initial = init_counts / init_counts.sum()

            for i in range(N):
                if trans_total_out[i] > 0:
                    self.transition[i, :] = trans_counts[i, :] / trans_total_out[i]

            for i in range(N):
                if emit_totals[i] > 0:
                    self.emission[i, :] = emit_counts[i, :] / emit_totals[i]

        return self.transition, self.emission


# =============================================================
#          DATA SPLIT: TRAIN / VALIDATION / TEST
# =============================================================
def split_data(seqs):
    train, temp = train_test_split(seqs, test_size=0.30, shuffle=True, random_state=42)
    val, test = train_test_split(temp, test_size=0.50, shuffle=True, random_state=42)
    return train, val, test


# =============================================================
#            PLOT LOG-LIKELIHOOD CURVES FOR ALL RUNS
# =============================================================
def plot_ll_curves(all_ll_curves):
    plt.figure(figsize=(10, 6))
    for i, ll_curve in enumerate(all_ll_curves):
        plt.plot(ll_curve, label=f"Run {i+1}")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.title("EM Convergence Curves Across Random Initializations")
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================================================
#                  AUTOMATIC MODEL COMPARISON
# =============================================================
def compare_models(results):
    """
    results is a list of dicts:
    {
      'model': hmm_object,
      'train_ll': float,
      'val_ll': float,
      'transition': ...,
      'emission': ...,
      'll_curve': [...]
    }
    """

    print("\n===== MODEL COMPARISON =====\n")
    print("Run | Train LL | Validation LL")
    print("----------------------------------")
    for i, r in enumerate(results):
        print(f"{i+1:3d} | {r['train_ll']:.2f} | {r['val_ll']:.2f}")
    print("----------------------------------")

    # select best model based on VAL LL
    best = max(results, key=lambda d: d["val_ll"])
    print("\nBest model = RUN", results.index(best) + 1)
    return best


# =============================================================
#                         MAIN
# =============================================================
if __name__ == "__main__":
    sequences = read_fasta_sequences("data/q_coins.seq.fa")
    print(f"Loaded {len(sequences)} sequences.")

    train_seqs, val_seqs, test_seqs = split_data(sequences)
    print(f"Train: {len(train_seqs)}, Validation: {len(val_seqs)}, Test: {len(test_seqs)}")

    NUM_RUNS = 10
    results = []
    all_ll_curves = []

    for run in range(NUM_RUNS):
        print(f"\n======== EM RUN {run+1}/{NUM_RUNS} ========")

        hmm = HMM_EM(states=["Fair", "Loaded"], observations=["H", "T"])
        hmm.baum_welch(train_seqs, max_iter=150)

        train_ll = sum(hmm.sequence_log_likelihood(s) for s in train_seqs)
        val_ll = sum(hmm.sequence_log_likelihood(s) for s in val_seqs)

        results.append({
            "model": hmm,
            "train_ll": train_ll,
            "val_ll": val_ll,
            "transition": hmm.transition.copy(),
            "emission": hmm.emission.copy(),
            "ll_curve": hmm.log_likelihoods.copy()
        })

        all_ll_curves.append(hmm.log_likelihoods.copy())

    # ---- plot convergence curves across runs ----
    plot_ll_curves(all_ll_curves)

    # ---- compare models and select best ----
    best = compare_models(results)

    print("\n===== BEST MODEL PARAMETERS =====")
    print("\nInitial:", best["model"].initial)
    print("\nTransition:\n", best["transition"])
    print("\nEmission:\n", best["emission"])

    # ---- final test set evaluation ----
    test_ll = sum(best["model"].sequence_log_likelihood(s) for s in test_seqs)
    print("\nTest Log-Likelihood:", test_ll)
