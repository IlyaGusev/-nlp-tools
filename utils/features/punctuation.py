import numpy as np


def punctuation_features(sentences, punctuation_items):
    features = []
    for punct in punctuation_items:
        features.append([int(punct in sentence) for sentence in sentences])
    return np.transpose(features)
