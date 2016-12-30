import json
import re
from utils.preprocess import text_to_wordlist
from utils.pipeline import BowFeaturesStep, CVStep, Pipeline
from sklearn.svm import LinearSVR


def process_microblogs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        train = [json.loads(obj) for obj in re.findall("\{[^}]*\}", content)]
        scores = [float(obj['sentiment score']) for obj in train if obj['spans'][0] != ""]

        train_texts = [" ".join(text_to_wordlist(
            re.sub(r"\$[a-zA-Z0-9]+", "", (" ".join(obj['spans'])))
        )) for obj in train if obj['spans'][0] != ""]

        pipeline = Pipeline(train_texts, [])
        pipeline.add_step(BowFeaturesStep(language='en', stem=True, tokenizer=None, preprocessor=None,
                                          use_tfidf=True, max_features=None, bow_ngrams=(1, 2)))
        pipeline.add_step(BowFeaturesStep(language='en', stem=False, tokenizer=None,
                                          preprocessor=None, use_tfidf=True, max_features=None, bow_ngrams=(3, 4),
                                          analyzer='char'))
        pipeline.add_step(CVStep(scores, LinearSVR(), n=20, test_size=0.2,
                                 random_state=2016, scoring='neg_mean_squared_error'))
        pipeline.run()


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

        pipeline = Pipeline(train_texts, [])
        pipeline.add_step(BowFeaturesStep(language='en', stem=True, tokenizer=None, preprocessor=None,
                                          use_tfidf=True, max_features=None, bow_ngrams=(1, 2)))
        pipeline.add_step(CVStep(scores, LinearSVR(), n=20, test_size=0.2,
                                 random_state=2016, scoring='neg_mean_squared_error'))
        pipeline.run()


def main():
    process_microblogs("datasets/Microblog_Train.json.txt")
    process_headlines("datasets/Headline_Train.json.txt")

if __name__ == '__main__':
    main()