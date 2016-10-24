import json
import re
from scipy.sparse import hstack, csr_matrix
from utils.preprocess import text_to_wordlist
from utils.bow import bow
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import ShuffleSplit, cross_val_score


def process_microblogs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        train = [json.loads(obj) for obj in re.findall("\{[^}]*\}", content)]
        scores = [float(obj['sentiment score']) for obj in train if obj['spans'][0] != ""]

        train_texts = [" ".join(text_to_wordlist(
            re.sub(r"\$[a-zA-Z0-9]+", "", (" ".join(obj['spans'])))
        )) for obj in train if obj['spans'][0] != ""]

        train_data = csr_matrix((len(train_texts), 1))
        test_data = csr_matrix((0, 1))

        # Word ngrams
        word_train_data, word_test_data = bow(train_texts, [], language='en', stem=True, tokenizer=None,
                                              preprocessor=None, use_tfidf=True, max_features=None, bow_ngrams=(1, 2))
        train_data = hstack([train_data, word_train_data])

        # Char ngrams
        char_train_data, char_test_data = bow(train_texts, [], language='en', stem=False, tokenizer=None,
                                              preprocessor=None, use_tfidf=True, max_features=None, bow_ngrams=(3, 4),
                                              analyzer='char')
        train_data = hstack([train_data, char_train_data])

        # Cross-validation
        clf = LinearSVR()
        cv = ShuffleSplit(20, test_size=0.2, random_state=10)
        cv_scores = cross_val_score(clf, train_data, scores, cv=cv, scoring='neg_mean_squared_error')
        print("Microblogs: Accuracy: %0.3f (+/- %0.3f)" % (cv_scores.mean(), cv_scores.std() * 2))


def process_headlines(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        train = [json.loads(obj) for obj in re.findall("\{[^}]*\}", content)]
        scores = [float(obj['sentiment']) for obj in train]
        train_texts = [obj['title'] for obj in train]
        companies = set([obj['company'] for obj in train])
        for company in list(companies):
            for variant in company.split():
                companies.add(variant)
            companies.add("".join([word[0] for word in company.split() if len(company.split()) >= 2]))
        for company in reversed(sorted(list(companies), key=len)):
            for i in range(len(train_texts)):
                train_texts[i] = train_texts[i].replace(company, "")
        train_texts = [" ".join(text_to_wordlist(re.sub(r"\$[a-zA-Z0-9]+", "", text))) for text in train_texts]

        train_data = csr_matrix((len(train_texts), 1))
        test_data = csr_matrix((0, 1))

        # Word ngrams
        word_train_data, word_test_data = bow(train_texts, [], language='en', stem=True, tokenizer=None,
                                              preprocessor=None, use_tfidf=True, max_features=None, bow_ngrams=(1, 2))
        train_data = hstack([train_data, word_train_data])

        # Cross-validation
        clf = LinearSVR()
        cv = ShuffleSplit(20, test_size=0.2, random_state=10)
        cv_scores = cross_val_score(clf, train_data, scores, cv=cv, scoring='neg_mean_squared_error')
        print("Headlines: Accuracy: %0.3f (+/- %0.3f)" % (cv_scores.mean(), cv_scores.std() * 2))


def main():
    process_microblogs("datasets/Microblog_Train.json.txt")
    process_headlines("datasets/Headline_Train.json.txt")
main()