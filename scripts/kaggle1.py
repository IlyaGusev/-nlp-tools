import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from bs4 import BeautifulSoup
# from utils.cnn import prepare_data, train_model, predict_answer

from utils.clf_pipeline import Pipeline, BowFeaturesStep, POSFeaturesStep, PunctuationStep, CVStep, OutputStep
from utils.preprocess import text_to_wordlist


def preprocessor(x):
    return " ".join(text_to_wordlist(BeautifulSoup(x, "html.parser").get_text()))


def main():
    train = pd.read_csv("datasets/KaggleSA1_LabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("datasets/KaggleSA1_TestData.tsv", header=0, delimiter="\t", quoting=3)

    pipeline = Pipeline(train["review"].tolist(), test["review"].tolist())
    pipeline.add_step(BowFeaturesStep(language='en', stem=True, tokenizer=text_to_wordlist, preprocessor=preprocessor,
                                      use_tfidf=True, max_features=None, bow_ngrams=(1, 2)))
    pipeline.add_step(POSFeaturesStep(language='en', stem=False, tokenizer=text_to_wordlist, preprocessor=preprocessor,
                                      use_tfidf=False, max_features=None, pos_ngrams=(1, 2)))
    # pipeline.add_step(PunctuationStep(['?!?', '!?!', '))', ')))', '((', '(((', ':)', ':(']))
    clf = MultinomialNB(alpha=0.1)
    # forest = RandomForestClassifier(n_estimators=100)
    # maxent = LogisticRegression()
    pipeline.add_step(CVStep(train['sentiment'], clf, n=20, test_size=0.2, scoring=None, random_state=2016))
    pipeline.add_step(OutputStep(clf, "results/KaggleSA1/tfidf_nb.csv", train['sentiment'], test['id']))
    pipeline.run()


# print("Lexicons...")
# train_data = np.column_stack((train_data, bl_score_with_negation(train_reviews)))
# test_data = np.column_stack((test_data, bl_test_data))
#
# print("CNN...")
# prepare_data(train, test, "models/imdb-train-val-test.pickle")
# train_model("models/imdb-train-val-test.pickle", 'models/cnn_3epochs.model')
# predict_answer("models/imdb-train-val-test.pickle", 'models/cnn_3epochs.model', 'datasets/KaggleSA1_TestData.tsv')

if __name__ == '__main__':
    main()