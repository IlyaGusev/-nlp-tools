import pandas as pd
from bs4 import BeautifulSoup
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from utils.bow import bow
from utils.preprocess import text_to_wordlist


def preprocessor(x):
    return " ".join(text_to_wordlist(BeautifulSoup(x, "html.parser").get_text()))

train = pd.read_csv("datasets/KaggleSA1_LabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("datasets/KaggleSA1_TestData.tsv", header=0, delimiter="\t", quoting=3)

train_data, test_data = bow(train["review"].tolist(), test["review"].tolist(), use_tfidf=True, language='en',
                            preprocessor=preprocessor)
# train_data = np.column_stack((train_data, bl_score_with_negation(train_reviews)))
# test_data = np.column_stack((test_data, bl_test_data))

print("Classifying...")
nb = MultinomialNB(alpha=0.1)
nb.fit(train_data, train['sentiment'])
answer = nb.predict(test_data)
output = pd.DataFrame(data={"id": test['id'], "sentiment": answer})
output.to_csv("results/KaggleSA1/tfidf_nb.csv", index=False, quoting=3)


print("CV")
forest = RandomForestClassifier(n_estimators=100)
maxent = LogisticRegression()
nb = MultinomialNB(alpha=0.1)
cv = cross_validation.ShuffleSplit(train.shape[0], n_iter=20, test_size=0.2, random_state=10)
scores = cross_validation.cross_val_score(nb, train_data, train['sentiment'], cv=cv)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))