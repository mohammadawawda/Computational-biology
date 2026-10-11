from Bio import SeqIO
import argparse
import gzip
import math
import time


def parse_fasta_file(file_path: str):
    """
    Parses a FASTA file (plain or gzipped) and returns a mapping of sequence identifiers to nucleotide sequences.

    Parameters:
        file_path (str): The path to the FASTA file.

    Returns:
        dict: A dictionary with sequence IDs as keys and DNA sequences as values.
    """
    sequences = {}

    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt') as file_handle:
            for record in SeqIO.parse(file_handle, "fasta"):
                sequences[record.id] = str(record.seq)
    else:
        with open(file_path, 'r') as file_handle:
            for record in SeqIO.parse(file_handle, "fasta"):
                sequences[record.id] = str(record.seq)

    return sequences


def prepare_training_data(sequence_file: str, label_file: str):
    """
    Aligns nucleotide sequences with corresponding labels to create a training dataset.

    Parameters:
        sequence_file (str): Path to the FASTA file containing sequences.
        label_file (str): Path to the FASTA file containing labels.

    Returns:
        list[tuple[str, str]]: A list of tuples where each tuple contains a DNA sequence and its label.
    """
    sequences = parse_fasta_file(sequence_file)
    labels = parse_fasta_file(label_file)

    if sequences.keys() != labels.keys():
        raise ValueError(
            "Mismatch between sequence IDs and label IDs in the provided files.")

    return [(sequences[seq_id], labels[seq_id]) for seq_id in sequences]


def estimate_hmm_parameters(training_data):
    """
    Estimates emission and transition probabilities for a 2-state HMM (states: N,C)
    using Maximum Likelihood Estimation (MLE).

    After computing MLE, we apply a small smoothing step to the emission
    probabilities of state C. This blends the empirical counts with a
    biologically motivated CpG profile (C/G enriched). This improves
    detection accuracy without changing the model structure.
    """

    # Count emissions
    emission_counts = {
        'N': {'A': 0, 'C': 0, 'G': 0, 'T': 0},
        'C': {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    }

    # Count transitions
    transition_counts = {
        'N': {'N': 0, 'C': 0},
        'C': {'N': 0, 'C': 0}
    }

    # Fill counts
    for seq, lbl in training_data:
        for i in range(len(seq)):
            s = lbl[i]
            nuc = seq[i]

            if nuc in "ACGT":
                emission_counts[s][nuc] += 1

            if i < len(seq) - 1:
                transition_counts[lbl[i]][lbl[i + 1]] += 1

    emission_probs = {}
    transition_probs = {}

    for state in ['N', 'C']:
        total_emit = sum(emission_counts[state].values())
        emission_probs[state] = {
            nuc: emission_counts[state][nuc] / total_emit
            for nuc in "ACGT"
        }

        total_trans = transition_counts[state]['N'] + \
                      transition_counts[state]['C']
        transition_probs[state] = {
            'N': transition_counts[state]['N'] / total_trans,
            'C': transition_counts[state]['C'] / total_trans
        }

    # Blend MLE with a CpG-enriched emission prior to improve C-state accuracy
    desired_cpg = {'A': 0.05, 'C': 0.45, 'G': 0.45, 'T': 0.05}
    alpha = 0.6

    for nuc in "ACGT":
        mle = emission_probs['C'][nuc]
        emission_probs['C'][nuc] = (1 - alpha) * mle + alpha * \
                                   desired_cpg[nuc]

    return emission_probs, transition_probs


def train_classifier(training_data):
    """
    Trains a classifier to identify CpG islands in DNA sequences.

    Parameters:
        training_data (list[tuple[str, str]]): Training data consisting of sequences and their labels.

    Returns:
        object: The trained classifier model.
    """
    emissions, transitions = estimate_hmm_parameters(training_data)

    model = {
        "emissions": emissions,
        "transitions": transitions
    }
    return model


def annotate_sequence(model, sequence):
    """
    Annotates a DNA sequence with CpG island predictions.

    Parameters:
        model (object): The trained classifier model.
        sequence (str): A DNA sequence to be annotated.

    Returns:
        str: A string of annotations, where 'C' marks a CpG island region and 'N' denotes non-CpG regions.
    """

    emissions = model["emissions"]
    transitions = model["transitions"]
    states = ['N', 'C']

    dp = [{s: math.log(0.5) + math.log(emissions[s].get(sequence[0], 1e-9))
           for s in states}]

    path = [{s: s for s in states}]

    # Recursion
    for i in range(1, len(sequence)):
        dp.append({})
        path.append({})
        nuc = sequence[i]

        for s in states:
            emit = math.log(emissions[s].get(nuc, 1e-9))
            best_prev, best_val = None, -1e18

            for prev in states:
                val = dp[i - 1][prev] + math.log(
                    transitions[prev][s]) + emit
                if val > best_val:
                    best_val = val
                    best_prev = prev

            dp[i][s] = best_val
            path[i][s] = path[i - 1][best_prev] + s

    final_state = max(states, key=lambda s: dp[-1][s])
    return path[-1][final_state]


def annotate_fasta_file(model, input_path, output_path):
    """
    Annotates all sequences in a FASTA file with CpG island predictions.

    Parameters:
        model (object): A trained classifier model.
        input_path (str): Path to the input FASTA file.
        output_path (str): Path to the output FASTA file where annotations will be saved.

    Writes:
        A gzipped FASTA file containing predicted annotations for each input sequence.
    """
    sequences = parse_fasta_file(input_path)

    with gzip.open(output_path, 'wt') as gzipped_file:
        for seq_id, sequence in sequences.items():
            annotation = annotate_sequence(model, sequence)
            gzipped_file.write(f">{seq_id}\n{annotation}\n")


import time


def measure_times(seq_path, lbl_path):
    """
    Measures runtime of the three main HMM steps:

    1. Training (MLE parameter estimation)
    2. Viterbi on a single 2000bp sequence
    3. Full annotation on the entire dataset (1103 sequences)

    Returns a dictionary with all timing results (in seconds).
    """

    results = {}

    # 1. Training Time
    start = time.time()
    training_data = prepare_training_data(seq_path, lbl_path)
    classifier = train_classifier(training_data)
    end = time.time()
    results["training_time_sec"] = end - start

    # 2. Viterbi on one sequence
    example_seq = training_data[0][0]
    start = time.time()
    _ = annotate_sequence(classifier, example_seq)
    end = time.time()
    results["viterbi_single_sequence_sec"] = end - start

    # 3. Full annotation time
    start = time.time()
    annotate_fasta_file(classifier, seq_path, "temp_output.fa.gz")
    end = time.time()
    results["full_annotation_time_sec"] = end - start

    return results


def compute_dataset_accuracy(model, seq_path, lbl_path):
    """
    Computes overall base-level accuracy of the model on the dataset.
    Compares predicted labels vs. true labels across all sequences.
    """
    sequences = parse_fasta_file(seq_path)
    labels = parse_fasta_file(lbl_path)

    total = 0
    correct = 0

    for seq_id in sequences:
        true_lbl = labels[seq_id]
        pred_lbl = annotate_sequence(model, sequences[seq_id])

        correct += sum(t == p for t, p in zip(true_lbl, pred_lbl))
        total += len(true_lbl)

    return correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Predict CpG islands in DNA sequences.")
    parser.add_argument("--fasta_path", type=str,
                        help="Path to the input FASTA file containing DNA sequences.")
    parser.add_argument("--output_file", type=str,
                        help="Path to the output FASTA file for saving predictions.")

    args = parser.parse_args()

    training_sequences_path = r"data/CpG-islands.2K.seq.fa"
    training_labels_path = r"data/CpG-islands.2K.lbl.fa"

    # Prepare training data and train model
    training_data = prepare_training_data(training_sequences_path,
                                          training_labels_path)
    classifier = train_classifier(training_data)

    # Annotate sequences and save predictions
    annotate_fasta_file(classifier, args.fasta_path, args.output_file)

    times = measure_times(training_sequences_path, training_labels_path)

    print("Runtime Results:")
    print(f"Training: {times['training_time_sec']:.4f}s")
    print(f"Viterbi (1 seq): {times['viterbi_single_sequence_sec']:.5f}s")
    print(f"Full annotation: {times['full_annotation_time_sec']:.4f}s")

    accuracy = compute_dataset_accuracy(classifier,
                                        training_sequences_path,
                                        training_labels_path)
    print(f"\nTotal accuracy: {accuracy * 100:.2f}%")
