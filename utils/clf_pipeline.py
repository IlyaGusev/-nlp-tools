import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ShuffleSplit, cross_val_score

from utils.features.bow import bow
from utils.features.punctuation import punctuation_features
from utils.preprocess import text_to_wordlist, get_sentence_tags


def concat(len_train, len_test, train_data, test_data, train_new, test_new):
    if len_train != 0:
        train_data = hstack([train_data, train_new])
    if len_test != 0:
        test_data = hstack([test_data, test_new])
    return train_data, test_data


class BowFeaturesStep(object):
    def __init__(self, language='en', stem=True, tokenizer=text_to_wordlist,
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
        print("Bow step...")
        bow_train_data, bow_test_data = bow(init_train_data, init_test_data, language=self.language, stem=self.stem,
                                            tokenizer=self.tokenizer, preprocessor=self.preprocessor,
                                            use_tfidf=self.use_tfidf, max_features=self.max_features,
                                            bow_ngrams=self.ngrams, analyzer=self.analyzer)
        return concat(len(init_train_data), len(init_test_data), train_data, test_data, bow_train_data, bow_test_data)


class POSFeaturesStep(object):
    def __init__(self, language='en', stem=False, tokenizer=text_to_wordlist,
                 preprocessor=None, use_tfidf=True, max_features=None,
                 pos_ngrams=(1, 1), analyzer='word'):
        self.language = language
        self.stem = stem
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.use_tfidf = use_tfidf
        self.max_features = max_features
        self.ngrams = pos_ngrams
        self.analyzer = analyzer

    def run(self, init_train_data, init_test_data, train_data, test_data):
        print("POS step...")
        pos_train_data = []
        pos_test_data = []
        for text in init_train_data:
            pos_train_data.append(get_sentence_tags(text, self.language))
        for text in init_test_data:
            pos_test_data.append(get_sentence_tags(text, self.language))
        pos_train_data, pos_test_data = bow(pos_train_data, pos_test_data, language=self.language, stem=self.stem,
                                            tokenizer=self.tokenizer, preprocessor=self.preprocessor,
                                            use_tfidf=self.use_tfidf, max_features=self.max_features,
                                            bow_ngrams=self.ngrams, analyzer=self.analyzer)
        return concat(len(init_train_data), len(init_test_data), train_data, test_data, pos_train_data, pos_test_data)


class PunctuationStep(object):
    def __init__(self, punctuation_items):
        self.punctuation_items = punctuation_items

    def run(self, init_train_data, init_test_data, train_data, test_data):
        print("Punctuation step...")
        punctuation_train_data = punctuation_features(init_train_data, self.punctuation_items)
        punctuation_test_data = punctuation_features(init_test_data, self.punctuation_items)
        return concat(len(init_train_data), len(init_test_data), train_data, test_data,
                      punctuation_train_data, punctuation_test_data)


class RawFeaturesStep(object):
    def __init__(self, train_features, test_features):
        self.train_features = train_features
        self.test_features = test_features

    def run(self, init_train_data, init_test_data, train_data, test_data):
        return concat(len(init_train_data), len(init_test_data), train_data, test_data,
                      self.train_features, self.test_features)


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
        print("CV: %0.3f (+/- %0.3f)" % (cv_scores.mean(), cv_scores.std() * 2))
        return train_data, test_data


class EvaluateFMeasureStep(object):
    def __init__(self, clf, train_answer, test_answer):
        self.train_answer = train_answer
        self.test_answer = test_answer
        self.clf = clf

    def run(self, init_train_data, init_test_data, train_data, test_data):
        self.clf.fit(train_data, self.train_answer)
        answer = self.clf.predict(test_data)
        result = ""
        result += "Accuracy: " + str(accuracy_score(self.test_answer, answer)) + '\n'
        result += "F-macro: " + str(f1_score(self.test_answer, answer, average='macro')) + '\n'
        result += "F-classes: " + str(f1_score(self.test_answer, answer, average=None)) + '\n'
        print(result)
        return train_data, test_data


class OutputStep(object):
    def __init__(self, clf, output_filename, train_answer, test_id):
        self.clf = clf
        self.output_filename = output_filename
        self.train_answer = train_answer
        self.test_id = test_id

    def run(self, init_train_data, init_test_data, train_data, test_data):
        print("Classifying...")
        self.clf.fit(train_data, self.train_answer)
        answer = self.clf.predict(test_data)
        output = pd.DataFrame(data={"id": self.test_id, "sentiment": answer})
        output.to_csv(self.output_filename, index=False, quoting=3)
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
