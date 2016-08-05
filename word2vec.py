import re
from nltk.tokenize import sent_tokenize
from gensim.models import word2vec, Word2Vec
import logging
import common


def sentence_to_wordlist(review):
    review_text = re.sub("[^а-яА-Я]"," ", review)
    words = review_text.lower().split()
    words = [x for x in words if x not in common.get_stopwords()]
    return words


def line_to_sentences(line):
    raw_sentences = sent_tokenize(line.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(sentence_to_wordlist(raw_sentence))
    return sentences


def append_reviews(sentences, filename, border):
    counter = 1
    with open(filename, 'r') as f:
        lines = []
        line = f.readline()
        while counter < border and line != '':
            lines.append(line)
            line = f.readline()
            counter += 1
        for line in lines:
            sentences += line_to_sentences(line)
    return sentences


def fetch(filenames):
    print("Datasets fetching...")
    all_sentences = []
    for filename in filenames:
        all_sentences = append_reviews(all_sentences, filename, 1000000)
    return all_sentences


def train(filenames, model_name):
    all_sentences = fetch(filenames)
    print("Training model...")
    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    window = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    model = word2vec.Word2Vec(all_sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                              window=window, sample=downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    print("Saving to: " + model_name)
    model.save(model_name)


# train(["datasets/wiki_ru.txt", "datasets/reviews_restoran.txt", "datasets/train.txt", "datasets/test.txt"], "models/300-40-10-1e3-wiki_ru-restoran-train16-test16-1kk")

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# model = Word2Vec.load_word2vec_format(("models/GoogleNews-vectors-negative300.bin.gz"), binary=True)
# print(model.similarity('woman', 'man'))
