import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

from utils.classifiers.w2v_model import load


def average_vector(words, model, num_features):
    result_vector = np.zeros((num_features,), dtype="float32")
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            result_vector = np.add(result_vector, model[word])
    result_vector = np.divide(result_vector, nwords)
    return result_vector


def w2v_average(train_texts, test_texts, model_name, output):
    model = load(model_name)
    train_data = []
    for text in train_texts:
        features = average_vector(text, model, model.syn0.shape[1])
        train_data.append(features)
    train_data = pd.DataFrame(train_data)

    test_data = []
    for review in test_texts:
        features = average_vector(review, model, model.syn0.shape[1])
        test_data.append(features)
    test_data = pd.DataFrame(test_data)

    return train_data, test_data


def create_bag_of_centroids(words, word_centroid_map):
    num_centroids = max(word_centroid_map.values()) + 1
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    for word in words:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids


def w2v_clustering(train_texts, test_texts, model_name, output):
    model = Word2Vec.load(model_name)
    word_vectors = model.syn0
    num_clusters = int(word_vectors.shape[0]/5)
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)
    word_centroid_map = dict(zip(model.index2word, idx))

    train_centroids = np.zeros((len(train_texts), num_clusters), dtype="float32")
    counter = 0
    for text in train_texts:
        train_centroids[counter] = create_bag_of_centroids(text, word_centroid_map)
        counter += 1

    test_centroids = np.zeros((len(train_texts), num_clusters), dtype="float32")
    counter = 0
    for text in test_texts:
        test_centroids[counter] = create_bag_of_centroids(text, word_centroid_map)
        counter += 1

    return train_centroids, test_centroids