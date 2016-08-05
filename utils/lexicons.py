import numpy as np

from utils.preprocess import text_to_wordlist

NEGATE = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
 "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
 "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
 "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
 "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
 "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
 "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
 "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

blp = open("datasets/BL_positive.txt")
positive_words = dict([(word, 1) for word in blp.read().splitlines()])
blp.close()

bln = open("datasets/BL_negative.txt")
negative_words = dict([(word, 1) for word in bln.read().splitlines()])
bln.close()

afinn = dict(map(lambda row: (row[0], int(row[1])), [line.split('\t') for line in open("datasets/AFINN.txt")]))


def bl_get_word_score(word):
    return positive_words.get(word, 0) - negative_words.get(word, 0)


def bl_score_with_negation(reviews):
    feature = []
    for review in reviews:
        score = 0
        inverse = False
        for word in text_to_wordlist(review):
            if word in NEGATE:
                inverse = True
                continue
            word_score = bl_get_word_score(word)
            if inverse and word_score != 0:
                word_score = -word_score
                inverse = False
            score += word_score
        feature.append(score)
    return feature


def bl_positive_counter(x):
    return sum([positive_words.get(word, 0) for word in text_to_wordlist(x)])


def bl_negative_counter(x):
    return sum([negative_words.get(word, 0) for word in text_to_wordlist(x)])


def afinn_counter_positive(x):
    return sum(map(lambda word: afinn.get(word, 0) if afinn.get(word, 0) > 0 else 0, text_to_wordlist(x)))


def afinn_counter_negative(x):
    return abs(sum(map(lambda word: afinn.get(word, 0) if afinn.get(word, 0) < 0 else 0, text_to_wordlist(x))))


def bl_lexicon_features(reviews):
    print("BL")
    data = np.column_stack((list(map(bl_positive_counter, reviews)),
                            list(map(bl_negative_counter, reviews))))
    data = np.column_stack((data, bl_score_with_negation(reviews)))
    return data


def afinn_lexicon_features(reviews):
    print("AFINN")
    data = np.column_stack((list(map(afinn_counter_positive, reviews)),
                            list(map(afinn_counter_negative, reviews))))
    return data
