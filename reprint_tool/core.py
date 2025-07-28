import numpy as np
import pandas as pd


def normalize(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))


def calculate_rmse(x, y):
    x_normalized = normalize(x)
    y_normalized = normalize(y)
    return np.sqrt(np.nanmean((x_normalized - y_normalized) ** 2))


def calculate_cosine(x, y):
    return 1 - np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def reprint(data, epsilon=1e-4):
    mutation_types = data.index
    signatures = data.columns[0:]
    reprint_probabilities = {signature: {} for signature in signatures}

    for signature in signatures:
        signature_probs = data[signature].values + epsilon
        for idx, mutation in enumerate(mutation_types):
            NL = mutation[0]
            NR = mutation[6]
            X, Y = mutation[2], mutation[4]
            denominator = np.sum([
                signature_probs[j]
                for j in range(len(mutation_types))
                if mutation_types[j].startswith(f"{NL}[{X}>") and
                   mutation_types[j].endswith(f"]{NR}") and
                   mutation_types[j][4] != X
            ])
            reprint_prob = signature_probs[idx] / denominator if denominator != 0 else 0
            reprint_probabilities[signature][mutation] = reprint_prob

    return pd.DataFrame(reprint_probabilities)