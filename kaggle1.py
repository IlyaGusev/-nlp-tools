import pandas as pd
from bs4 import BeautifulSoup
from bow import bow_core
from common import rf_fit_predict, sentence_to_wordlist
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import gc
import numpy as np
import csv


def preprocessor(x):
    return " ".join(sentence_to_wordlist(BeautifulSoup(x, "lxml").get_text()))


def bl_positive_words_counter(x):
    return sum([1 if word in positive_words else 0 for word in sentence_to_wordlist(x)])


def bl_negative_words_counter(x):
    return sum([1 if word in negative_words else 0 for word in sentence_to_wordlist(x)])

afinn = dict(map(lambda row: (row[0], int(row[1])), [line.split('\t') for line in open("datasets/AFINN.txt")]))


def afinn_counter(x):
    return sum(map(lambda word: afinn.get(word, 0), sentence_to_wordlist(x)))

blp = open("datasets/BL_positive.txt")
positive_words = blp.read().splitlines()
blp.close()

bln = open("datasets/BL_negative.txt")
negative_words = bln.read().splitlines()
bln.close()

train_file = open('datasets/KaggleSA1_LabeledTrainData.tsv', 'r')
reader = csv.reader(train_file, delimiter='\t', quotechar='"')
train_raw_sentences = []
train_answer = []
for row in reader:
    train_raw_sentences.append(row[2])
    train_answer.append(row[1])
train_raw_sentences.pop(0)
train_answer.pop(0)
train_file.close()

test_file = open('datasets/KaggleSA1_TestData.tsv', 'r')
reader = csv.reader(test_file, delimiter='\t', quotechar='"')
test_raw_sentences = []
test_ids = []
for row in reader:
    test_raw_sentences.append(row[1])
    test_ids.append(row[0])
test_raw_sentences.pop(0)
test_ids.pop(0)
test_file.close()

print("Preprocessing...")

train_sentences = list(map(preprocessor, train_raw_sentences))
test_sentences = list(map(preprocessor, test_raw_sentences))
train_data, test_data = bow_core(train_sentences, test_sentences, True, False, 'en')

print("BL")
train_data = np.column_stack((train_data, list(map(bl_positive_words_counter, train_raw_sentences))))
train_data = np.column_stack((train_data, list(map(bl_negative_words_counter, train_raw_sentences))))
test_data = np.column_stack((test_data, list(map(bl_positive_words_counter, test_raw_sentences))))
test_data = np.column_stack((test_data, list(map(bl_negative_words_counter, test_raw_sentences))))

print("AFINN")
train_data = np.column_stack((train_data, list(map(afinn_counter, train_raw_sentences))))
test_data = np.column_stack((test_data, list(map(afinn_counter, test_raw_sentences))))

answer = rf_fit_predict(train_data, train_answer, test_data)
output = pd.DataFrame(data={"id": test_ids, "sentiment": answer})
output.to_csv("results/KaggleSA1_bow_stem_lexicon.csv", index=False, quoting=3)

forest = RandomForestClassifier(n_estimators=100)
cv = cross_validation.ShuffleSplit(len(train_raw_sentences), n_iter=10, test_size=0.2, random_state=10)
scores = cross_validation.cross_val_score(forest, train_data, train_answer, cv=cv)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
