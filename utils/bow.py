import copy

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from utils.preprocess import text_to_wordlist, stem_sentence, get_sentence_tags


def bow(train_texts, test_texts, language='en', stem=False, tokenizer=text_to_wordlist, preprocessor=None,
        use_tfidf=False, max_features=None, bow_ngrams=(1,2), analyzer='word'):
    train = copy.deepcopy(train_texts)
    test = copy.deepcopy(test_texts)
    if stem:
        for i in range(len(train)):
            train[i] = stem_sentence(train[i], language)
        for i in range(len(test)):
            test[i] = stem_sentence(test[i], language)

    if use_tfidf:
        vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=bow_ngrams, tokenizer=tokenizer,
                                     preprocessor=preprocessor, max_features=max_features)
    else:
        vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=bow_ngrams, tokenizer=tokenizer,
                                     preprocessor=preprocessor, max_features=max_features)
    data = train+test
    data = vectorizer.fit_transform(data)
    train_data = data[:len(train)]
    test_data = data[len(train):]
    return train_data, test_data