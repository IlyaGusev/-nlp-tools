import pandas as pd
from bs4 import BeautifulSoup
from common import sentence_to_wordlist
from bow import bow_core
from lexicons import bl_lexicon_features, bl_score_with_negation
from nltk.classify import MaxentClassifier
from rules import punctuation_features
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from nltk.tokenize import sent_tokenize
import numpy as np
import csv


def preprocessor(x):
    return " ".join(sentence_to_wordlist(BeautifulSoup(x, "html.parser").get_text()))

train_file = open('datasets/KaggleSA1_LabeledTrainData.tsv', 'r', encoding="utf8")
reader = csv.reader(train_file, delimiter='\t', quotechar='"')
train_raw_reviews = []
train_answer = []
for row in reader:
    train_raw_reviews.append(row[2])
    train_answer.append(row[1])
train_raw_reviews.pop(0)
train_answer.pop(0)
train_file.close()

test_file = open('datasets/KaggleSA1_TestData.tsv', 'r', encoding="utf8")
reader = csv.reader(test_file, delimiter='\t', quotechar='"')
test_raw_reviews = []
test_ids = []
for row in reader:
    test_raw_reviews.append(row[1])
    test_ids.append(row[0])
test_raw_reviews.pop(0)
test_ids.pop(0)
test_file.close()

print("Preprocessing...")
train_reviews = list(map(preprocessor, train_raw_reviews))
test_reviews = list(map(preprocessor, test_raw_reviews))

train_data, test_data = bow_core(train_reviews, test_reviews, False, False, 'en')
# train_data = np.column_stack((train_data, bl_score_with_negation(train_reviews)))
# test_data = np.column_stack((test_data, bl_test_data))

print("Classifying...")
# forest = RandomForestClassifier(n_estimators=100)
# forest.fit(train_data, train_answer)
# answer = forest.predict(test_data)
# log_reg = LogisticRegression()
# log_reg.fit(train_data, train_answer)
# answer = log_reg.predict(test_data)
nb = MultinomialNB(alpha=0.1)
nb.fit(train_data, train_answer)
answer = nb.predict(test_data)
output = pd.DataFrame(data={"id": test_ids, "sentiment": answer})
output.to_csv("results/KaggleSA1_tfidf_nb.csv", index=False, quoting=3)


print("CV")
forest = RandomForestClassifier(n_estimators=100)
maxent = LogisticRegression()
nb = MultinomialNB(alpha=0.1)
cv = cross_validation.ShuffleSplit(len(train_raw_reviews), n_iter=20, test_size=0.2, random_state=10)
scores = cross_validation.cross_val_score(nb, train_data, train_answer, cv=cv)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
