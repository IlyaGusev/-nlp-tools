import json
import re
from scipy.sparse import hstack, csr_matrix
from utils.preprocess import text_to_wordlist
from utils.bow import bow
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import ShuffleSplit, cross_val_score


class BowStep(object):
    def __init__(self, language='en', stem=True, tokenizer=None,
                 preprocessor=None, use_tfidf=True, max_features=None,
                 bow_ngrams=(1, 2), analyzer='word'):
        self.language = language
        self.stem = stem
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.use_tfidf = use_tfidf
        self.max_features = max_features
        self.ngrams = bow_ngrams
        self.analyzer = analyzer

    def run(self, init_train_data, init_test_data, train_data, test_data):
        bow_train_data, bow_test_data = bow(init_train_data, init_test_data, language=self.language, stem=self.stem,
                                            tokenizer=self.tokenizer, preprocessor=self.preprocessor,
                                            use_tfidf=self.use_tfidf, max_features=self.max_features,
                                            bow_ngrams=self.ngrams, analyzer=self.analyzer)
        if len(init_train_data) != 0:
            train_data = hstack([train_data, bow_train_data])
        if len(init_test_data) != 0:
            test_data = hstack([test_data, bow_test_data])
        return train_data, test_data


class CVStep(object):
    def __init__(self, train_answer, clf, n=20, test_size=0.2, scoring='neg_mean_squared_error', random_state=2016):
        self.clf = clf
        self.n = n
        self.test_size = test_size
        self.random_state = random_state
        self.scoring = scoring
        self.train_answer = train_answer

    def run(self, init_train_data, init_test_data, train_data, test_data):
        cv = ShuffleSplit(self.n, test_size=self.test_size, random_state=self.random_state)
        cv_scores = cross_val_score(self.clf, train_data, self.train_answer, cv=cv, scoring=self.scoring)
        print("Microblogs: Accuracy: %0.3f (+/- %0.3f)" % (cv_scores.mean(), cv_scores.std() * 2))
        return train_data, test_data


class Pipeline(object):
    def __init__(self, train_data, test_data):
        self.steps = []
        self.init_train_data = train_data
        self.init_test_data = test_data
        self.train_data = csr_matrix((len(train_data), 1))
        self.test_data = csr_matrix((len(test_data), 1))

    def run(self):
        for step in self.steps:
            self.train_data, self.test_data = step.run(self.init_train_data, self.init_test_data,
                                                       self.train_data, self.test_data)

    def add_step(self, step):
        self.steps.append(step)


