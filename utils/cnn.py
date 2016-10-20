import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.constraints import unitnorm
from keras.regularizers import l2

from utils.preprocess import text_to_wordlist
from utils.w2v_model import load


def clean_data(data_train, data_test):
    vocab = defaultdict(float)
    # Pre-process train data set
    train_texts = []
    for i in range(len(data_train)):
        words = text_to_wordlist(data_train[i])
        for word in set(words):
            vocab[word] += 1
        train_texts.append(" ".join(words))

    # Pre-process test data set
    test_texts = []
    for i in range(len(data_test)):
        words = text_to_wordlist(data_test[i])
        for word in set(words):
            vocab[word] += 1
        test_texts.append(" ".join(words))

    return train_texts, test_texts, vocab


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    word_idx_map = dict()
    W = np.zeros(shape=(len(word_vecs) + 1, k), dtype=np.float32)
    W[0] = np.zeros(k, dtype=np.float32)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(w2v_model_filename, vocab):
    word_vecs = {}
    model = load(w2v_model_filename)
    for word in vocab.keys():
        if word in model.vocab:
            word_vecs[word] = model[word]
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def get_idx_from_sent(sent, word_idx_map, max_l=51, kernel_size=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = kernel_size - 1
    for i in range(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x


def make_idx_data(train_texts, test_texts, train_answer, word_idx_map, max_l=51, kernel_size=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for i in range(len(train_texts)):
        sent = get_idx_from_sent(train_texts[i], word_idx_map, max_l, kernel_size)
        sent.append(train_answer[i])
        train.append(sent)
    for i in range(len(test_texts)):
        sent = get_idx_from_sent(test_texts[i], word_idx_map, max_l, kernel_size)
        test.append(sent)
    train = np.array(train, dtype=np.int)
    test = np.array(test, dtype=np.int)
    return train, test


def assemble_model(input_dim, output_dim, conv_input_height, conv_input_width, weigths, number_of_classes=2):
    # Number of feature maps (outputs of convolutional layer)
    N_fm = 300
    # kernel size of convolutional layer
    kernel_size = 8

    model = Sequential()
    # Embedding layer (lookup table of trainable word vectors)
    model.add(Embedding(input_dim=input_dim,
                        output_dim=output_dim,
                        input_length=conv_input_height,
                        weights=weigths,
                        W_constraint=unitnorm()))
    # Reshape word vectors from Embedding to tensor format suitable for Convolutional layer
    model.add(Reshape((1, conv_input_height, conv_input_width)))

    # first convolutional layer
    model.add(Convolution2D(N_fm,
                            kernel_size,
                            conv_input_width,
                            border_mode='valid',
                            W_regularizer=l2(0.0001)))
    # ReLU activation
    model.add(Activation('relu'))

    # aggregate data in every feature map to scalar using MAX operation
    model.add(MaxPooling2D(pool_size=(conv_input_height - kernel_size + 1, 1)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    # Inner Product layer (as in regular neural network, but without non-linear activation function)
    model.add(Dense(number_of_classes))
    # SoftMax activation; actually, Dense+SoftMax works as Multinomial Logistic Regression
    model.add(Activation('softmax'))

    # Custom optimizers could be used, though right now standard adadelta is employed
    opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def prepare_data(train, test, train_answer, filename, w2v_model="models/GoogleNews-vectors-negative300.bin.gz"):
    train_texts, test_texts, vocab = clean_data(train, test)

    vectors = load_bin_vec(w2v_model, vocab)
    print('word2vec loaded!')

    add_unknown_words(vectors, vocab)
    W, word_idx_map = get_W(vectors)
    pickle.dump([train_texts, test_texts, train_answer.tolist(), W, word_idx_map, vocab],
                open(filename, 'wb'))
    print('dataset created!')


def train_model(data_filename, model_filename, n_epoch=3, number_of_classes=2):
    print("loading data...")
    x = pickle.load(open(data_filename, "rb"))
    train_texts, test_texts, train_answer, W, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4], x[5]
    print("data loaded!")

    train_data, test_data = make_idx_data(train_texts, test_texts, train_answer, word_idx_map,
                                          max_l=2514, kernel_size=5)

    # Train data preparation
    N = train_data.shape[0]
    conv_input_width = W.shape[1]
    conv_input_height = int(train_data.shape[1] - 1)

    # For each word write a word index (not vector) to X tensor
    train_X = np.zeros((N, conv_input_height), dtype=np.int)
    train_Y = np.zeros((N, 3), dtype=np.int)
    for i in range(N):
        for j in range(conv_input_height):
            train_X[i, j] = train_data[i, j]
        train_Y[i, train_data[i, -1]] = 1

    model = assemble_model(input_dim=W.shape[0], output_dim=W.shape[1], weigths=[W],
                           conv_input_height=conv_input_height, conv_input_width=conv_input_width,
                           number_of_classes=number_of_classes)
    epoch = 0
    for i in range(n_epoch):
        model.fit(train_X, train_Y, batch_size=50, nb_epoch=1, verbose=1)
        print("Epoch " + str(epoch) + " done.")
        epoch += 1

    model.save_weights(model_filename)


def predict_answer(data_filename, model_filename, test_answer, number_of_classes=2):
    print("loading data...")
    x = pickle.load(open(data_filename, "rb"))
    train_texts, test_texts, train_answer, W, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4], x[5]
    print("data loaded!")

    train_data, test_data = make_idx_data(train_texts, test_texts, train_answer,
                                          word_idx_map, max_l=2514, kernel_size=5)
    # Test data preparation
    Nt = test_data.shape[0]
    conv_input_width = W.shape[1]
    conv_input_height = int(test_data.shape[1] - 1)

    # For each word write a word index (not vector) to X tensor
    test_X = np.zeros((Nt, conv_input_height), dtype=np.int)
    for i in range(Nt):
        for j in range(conv_input_height):
            test_X[i, j] = test_data[i, j]

    print('test_X.shape = {}'.format(test_X.shape))

    model = assemble_model(input_dim=W.shape[0], output_dim=W.shape[1], weigths=[W],
                           conv_input_height=conv_input_height, conv_input_width=conv_input_width,
                           number_of_classes=number_of_classes)
    model.load_weights(model_filename)
    p = model.predict_classes(test_X, batch_size=10)
    # model.evaluate(test_X, np.array(test_answer), batch_size=10, show_accuracy=True)

    return p
    # data = pd.read_csv(test_filename, sep='\t')
    # d = pd.DataFrame({'id': data['id'], 'sentiment': p[:, 0]})
    # d.to_csv('results/cnn_3epochs.csv', index=False)
