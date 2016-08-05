import common
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from gensim.models import Word2Vec


def average_vector(words, model, num_features):
    result_vector = np.zeros((num_features,), dtype="float32")
    nwords = 0
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords += 1
            result_vector = np.add(result_vector, model[word])
    if nwords == 0:
        print(words)
    result_vector = np.divide(result_vector, nwords)
    return result_vector


def w2v_average(train, test, model_name, output):
    train_reviews, train_answer, train_additional_features = common.preprocess_data(
        common.semeval_get_data(train))
    test_reviews, test_answer, test_additional_features = common.preprocess_data(
        common.semeval_get_data(test))

    # model = Word2Vec.load(model_name)
    model = Word2Vec.load_word2vec_format(model_name, binary=True)

    train_data = []
    for review in train_reviews:
        features = average_vector(review, model, model.syn0.shape[1])
        train_data.append(features)
    train_data = pd.DataFrame(train_data)
    train_data['Category'] = train_additional_features[0]

    test_data = []
    for review in test_reviews:
        features = average_vector(review, model, model.syn0.shape[1])
        test_data.append(features)
    test_data = pd.DataFrame(test_data)
    test_data['Category'] = test_additional_features[0]

    answer = common.rf_fit_predict(train_data, train_answer, test_data)
    result = common.evaluate(test_answer, answer)
    with open(output, 'w') as f:
        f.write(result)


def create_bag_of_centroids(words, word_centroid_map):
    num_centroids = max(word_centroid_map.values()) + 1
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    for word in words:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids


def w2v_clustering(train, test, model_name, output):
    model = Word2Vec.load(model_name)
    word_vectors = model.syn0
    num_clusters = int(word_vectors.shape[0]/5)
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)
    word_centroid_map = dict(zip(model.index2word, idx))

    train_reviews, train_answer, train_additional_features = common.preprocess_data(
        common.semeval_get_data(train))
    test_reviews, test_answer, test_additional_features = common.preprocess_data(
        common.semeval_get_data(test))

    train_centroids = np.zeros((len(train_reviews), num_clusters), dtype="float32")
    counter = 0
    for review in train_reviews:
        train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    test_centroids = np.zeros((len(test_reviews), num_clusters), dtype="float32")
    counter = 0
    for review in test_reviews:
        test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    answer = common.rf_fit_predict(train_centroids, train_answer, test_centroids)
    result = common.evaluate(test_answer, answer)
    with open(output, 'w') as f:
        f.write(result)