import argparse
import math
import os
from io import StringIO
import numpy as np
from Bio import SeqIO, Phylo
from matplotlib import pyplot as plt

NUCLEOTIDES = ['A', 'C', 'G', 'T']


# ------------------ I/O ------------------ #

def parse_fasta_file(file_path: str):
    """
    Parse a FASTA file into a dictionary {id: sequence}.
    All sequences are assumed to be aligned.
    """
    sequences = {}
    with open(file_path, 'r') as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequences[record.id] = str(record.seq)
    return sequences

def plot_tree(newick_string: str, file_name: str):
    """
    Visualizes a phylogenetic tree given in Newick format and saves it as an image.
    """
    tree = Phylo.read(StringIO(newick_string), "newick")

    def branch_length_formatter(clade):
        if clade.branch_length is not None:
            return f"{clade.branch_length:.3f}"
        return ""

    Phylo.draw(tree, do_show=False, branch_labels=branch_length_formatter)
    plt.title("Phylogenetic Tree")
    plt.savefig(file_name)
    plt.close()


# ------------------ Part 1: Evolutionary Time ------------------ #

def calculate_time_between_two_sequences(seq1, seq2):
    """
    Calculate evolutionary distance between two aligned sequences
    using the Jukes–Cantor (JC69) model.

    Returns:
        float: evolutionary distance
    """
    if len(seq1) != len(seq2):
        raise ValueError("Aligned sequences must have the same length")

    n = len(seq1)
    if n == 0:
        raise ValueError("Sequences must not be empty")

    q = 0
    for nt1, nt2 in zip(seq1, seq2):
        if nt1 != nt2:
            q += 1

    d = q / n
    
    if d >= 0.75:
        return float("inf")

    t = -(3 / 4) * math.log(1 - (4 / 3) * d)
    return t


# ------------------ Part 2: Distance Matrix ------------------ #

def calculate_distance_matrix(sequences):
    """
    Compute a pairwise distance matrix for all sequences
    using the Jukes–Cantor model.

    Returns:
        ids (list)
        distance_matrix (np.ndarray)
    """
    ids = list(sequences.keys())
    n = len(ids)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = calculate_time_between_two_sequences(
                sequences[ids[i]],
                sequences[ids[j]]
            )
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return ids, distance_matrix


# ------------------ Part 3: Tree Construction ------------------ #

def neighbor_joining(distance_matrix, ids):
    """
    Build a phylogenetic tree using the Neighbor Joining algorithm.

    Steps:
    1. Compute total divergence r_i
    2. Compute adjusted matrix M_ij
    3. Join the closest pair
    4. Update distances
    5. Repeat until two nodes remain

    Returns:
        str: Newick formatted tree
    """
    D = distance_matrix.astype(float).copy()
    current_ids = ids.copy()
    node_strings = {name: name for name in current_ids}

    while len(current_ids) > 2:
        n = len(current_ids)
        r = np.sum(D, axis=1)
        M = np.full((n, n), np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = (n - 2) * D[i, j] - r[i] - r[j]

        i, j = np.unravel_index(np.argmin(M), M.shape)

        if i > j:
            i, j = j, i

        id_i = current_ids[i]
        id_j = current_ids[j]
        dij = D[i, j]

        delta_i = 0.5 * dij + (r[i] - r[j]) / (2 * (n - 2))
        delta_j = dij - delta_i

        new_node = f"({node_strings[id_i]}:{delta_i:.6f},{node_strings[id_j]}:{delta_j:.6f})"

        new_distances = []
        remaining_indices = [k for k in range(n) if k not in (i, j)]
        remaining_ids = [current_ids[k] for k in remaining_indices]

        for k in remaining_indices:
            dik = D[i, k]
            djk = D[j, k]
            duk = 0.5 * (dik + djk - dij)
            new_distances.append(duk)

        new_size = n - 1
        new_D = np.zeros((new_size, new_size))

        for a, old_a in enumerate(remaining_indices):
            for b, old_b in enumerate(remaining_indices):
                new_D[a, b] = D[old_a, old_b]

        for a, dist in enumerate(new_distances):
            new_D[a, new_size - 1] = dist
            new_D[new_size - 1, a] = dist

        new_id = f"{id_i}_{id_j}"
        current_ids = remaining_ids + [new_id]
        node_strings[new_id] = new_node
        D = new_D

    id_i = current_ids[0]
    id_j = current_ids[1]
    final_dist = D[0, 1]

    tree = f"({node_strings[id_i]}:{final_dist/2:.6f},{node_strings[id_j]}:{final_dist/2:.6f});"
    return tree

def UPGMA(distance_matrix, ids):
    """
    Build a phylogenetic tree using the UPGMA algorithm.

    Notes:
    - Assumes a molecular clock
    - Distances must be ultrametric

    Returns:
        str: Newick formatted tree
    """
    D = distance_matrix.astype(float).copy()
    current_ids = ids.copy()

    cluster_strings = {name: name for name in current_ids}
    cluster_sizes = {name: 1 for name in current_ids}
    cluster_heights = {name: 0.0 for name in current_ids}

    while len(current_ids) > 1:
        n = len(current_ids)

        min_dist = float("inf")
        min_i, min_j = -1, -1

        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] < min_dist:
                    min_dist = D[i, j]
                    min_i, min_j = i, j

        i, j = min_i, min_j
        if i > j:
            i, j = j, i

        id_i = current_ids[i]
        id_j = current_ids[j]

        new_height = min_dist / 2.0

        branch_i = new_height - cluster_heights[id_i]
        branch_j = new_height - cluster_heights[id_j]

        new_cluster_string = (
            f"({cluster_strings[id_i]}:{branch_i:.6f},{cluster_strings[id_j]}:{branch_j:.6f})"
        )
        new_cluster_id = f"{id_i}_{id_j}"
        new_cluster_size = cluster_sizes[id_i] + cluster_sizes[id_j]

        remaining_indices = [k for k in range(n) if k not in (i, j)]
        remaining_ids = [current_ids[k] for k in remaining_indices]

        new_distances = []
        for k in remaining_indices:
            id_k = current_ids[k]
            dist_to_new = (
                cluster_sizes[id_i] * D[i, k] + cluster_sizes[id_j] * D[j, k]
            ) / new_cluster_size
            new_distances.append(dist_to_new)

        new_size = n - 1
        new_D = np.zeros((new_size, new_size))

        for a, old_a in enumerate(remaining_indices):
            for b, old_b in enumerate(remaining_indices):
                new_D[a, b] = D[old_a, old_b]

        for a, dist in enumerate(new_distances):
            new_D[a, new_size - 1] = dist
            new_D[new_size - 1, a] = dist

        current_ids = remaining_ids + [new_cluster_id]
        cluster_strings[new_cluster_id] = new_cluster_string
        cluster_sizes[new_cluster_id] = new_cluster_size
        cluster_heights[new_cluster_id] = new_height

        D = new_D

    return cluster_strings[current_ids[0]] + ";"


# ------------------ Part 4: Felsenstein Pruning ------------------ #

def jukes_cantor_transition_prob(t):
    """
    Compute JC69 transition probabilities for branch length t.

    P_ii(t) = 1/4 + 3/4 * exp(-4t/3)
    P_ij(t) = 1/4 - 1/4 * exp(-4t/3),  i != j

    Returns:
        dict mapping (parent_nt, child_nt) -> probability
    """
    exp_term = math.exp(-4.0 / 3.0 * t)
    p_same = 0.25 + 0.75 * exp_term
    p_diff = 0.25 - 0.25 * exp_term

    probs = {}
    for i in NUCLEOTIDES:
        for j in NUCLEOTIDES:
            probs[(i, j)] = p_same if i == j else p_diff
    return probs


def felsenstein_pruning(newick_string: str, sequences: dict):
    """
    Compute the log-likelihood of a tree using Felsenstein's pruning algorithm
    and reconstruct the most likely root sequence.

    Algorithm (per site s):
      1. Leaf v with observed base x_v:  L_v(i) = 1 if i == x_v, else 0
      2. Internal node v with children c1, c2, ...:
         L_v(i) = prod_c [ sum_j P_ij(t_vc) * L_c(j) ]
      3. Site likelihood = (1/4) * sum_i L_root(i)
      4. Total log-likelihood = sum_s log(site_likelihood_s)
      5. Reconstructed base at site s = argmax_i L_root(i)
         (since pi_i = 1/4 is uniform, argmax pi_i*L_r(i) = argmax L_r(i))

    Returns:
        tuple:
            reconstructed_root_sequence (str)
            log_likelihood (float)
    """
    tree = Phylo.read(StringIO(newick_string), "newick")

    # Determine sequence length
    seq_length = len(next(iter(sequences.values())))
    num_bases = len(NUCLEOTIDES)
    base_to_idx = {b: i for i, b in enumerate(NUCLEOTIDES)}

    # Store likelihood vectors: clade -> np.ndarray of shape (seq_length, 4)
    likelihood = {}

    # Post-order traversal (leaves first, root last)
    for clade in tree.find_clades(order='postorder'):
        if clade.is_terminal():
            # ----- Leaf initialization -----
            seq = sequences.get(clade.name)
            if seq is None:
                raise ValueError(
                    f"Sequence not found for leaf '{clade.name}'. "
                    f"Available: {list(sequences.keys())}"
                )
            L = np.zeros((seq_length, num_bases))
            for s in range(seq_length):
                base = seq[s].upper()
                if base in base_to_idx:
                    L[s, base_to_idx[base]] = 1.0
                else:
                    # Unknown base (N, etc.) — all equally likely
                    L[s, :] = 1.0
            likelihood[id(clade)] = L
        else:
            # ----- Internal node recursion -----
            L = np.ones((seq_length, num_bases))

            for child in clade.clades:
                t = child.branch_length if child.branch_length is not None else 0.0
                t = max(t, 0.0)  # guard against negative branch lengths

                # Build 4x4 JC transition matrix for this branch
                exp_term = math.exp(-4.0 / 3.0 * t)
                p_same = 0.25 + 0.75 * exp_term
                p_diff = 0.25 - 0.25 * exp_term
                P = np.full((num_bases, num_bases), p_diff)
                np.fill_diagonal(P, p_same)

                # contribution[s, i] = sum_j P[i,j] * L_child[s, j]
                # = (L_child @ P^T)[s, i]
                child_L = likelihood[id(child)]
                contribution = child_L @ P.T
                L *= contribution

            likelihood[id(clade)] = L

    # ----- Compute log-likelihood at the root -----
    root = tree.root
    root_L = likelihood[id(root)]  # shape: (seq_length, 4)

    # Site likelihood = pi * sum_i L_root(i) = (1/4) * sum_i L_root(i)
    site_likelihoods = 0.25 * np.sum(root_L, axis=1)  # shape: (seq_length,)
    site_likelihoods = np.maximum(site_likelihoods, 1e-300)  # avoid log(0)
    log_likelihood = np.sum(np.log(site_likelihoods))

    # ----- Reconstruct ancestral sequence -----
    # argmax_i [ pi_i * L_root(i) ] = argmax_i L_root(i) (uniform prior)
    best_indices = np.argmax(root_L, axis=1)
    reconstructed_root_sequence = ''.join(NUCLEOTIDES[idx] for idx in best_indices)

    return reconstructed_root_sequence, log_likelihood


# ------------------ Command Line Interface ------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phylogenetic trees and ancestor sequences")

    parser.add_argument(
        "--fasta",
        type=str,
        required=True,
        help="Path to aligned FASTA file"
    )

    parser.add_argument(
        "--part",
        type=str,
        required=True,
        choices=["build_tree", "likelihood", "reconstruction"],
        help="Which analysis step to run"
    )

    parser.add_argument(
        "--tree_type",
        type=str,
        choices=["NJ", "UPGMA"],
        help="Tree construction method (required for build_tree, likelihood, reconstruction)"
    )

    args = parser.parse_args()
    sequences = parse_fasta_file(args.fasta)

    # Create graphs directory and build a descriptive filename from the fasta path
    os.makedirs("graphs", exist_ok=True)
    fasta_basename = os.path.splitext(os.path.basename(args.fasta))[0]  # e.g. "descendants_50"

    if args.part == "build_tree":
        if args.tree_type is None:
            raise ValueError("--tree_type is required for build_tree")

        ids, dist_matrix = calculate_distance_matrix(sequences)

        if args.tree_type == "NJ":
            tree = neighbor_joining(dist_matrix, ids)
        else:
            tree = UPGMA(dist_matrix, ids)

        print(tree)
        tree_img = os.path.join("graphs", f"tree_{fasta_basename}_{args.tree_type}.png")
        plot_tree(tree, tree_img)
        print(f"Tree saved to: {tree_img}")

    elif args.part in ["likelihood", "reconstruction"]:
        if args.tree_type is None:
            raise ValueError("--tree_type is required")

        ids, dist_matrix = calculate_distance_matrix(sequences)

        if args.tree_type == "NJ":
            tree = neighbor_joining(dist_matrix, ids)
        else:
            tree = UPGMA(dist_matrix, ids)

        print(tree)
        tree_img = os.path.join("graphs", f"tree_{fasta_basename}_{args.tree_type}.png")
        plot_tree(tree, tree_img)
        print(f"Tree saved to: {tree_img}")

        root_seq, log_likelihood = felsenstein_pruning(tree, sequences)

        if args.part == "likelihood":
            print("Log-Likelihood:", log_likelihood)
        else:
            print("Reconstructed Root Sequence:", root_seq)
