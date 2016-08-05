import numpy as np

PUNC_LIST = ["!", "?", "!!", "!!!", "??", "???", "?!?", "!?!", "?!?!", "!?!?"]


def punctuation_features(sentences):
    features = []
    for punct in PUNC_LIST:
        features.append([int(punct in sentence) for sentence in sentences])
    return np.transpose(features)
