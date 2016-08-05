import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pymorphy2
import common
from nltk.stem.snowball import SnowballStemmer


def stem(reviews, language):
    morph_ru = pymorphy2.MorphAnalyzer()
    morph_en = SnowballStemmer("english")
    for i in range(len(reviews)):
        words = reviews[i].split(" ")
        for j in range(len(words)):
            if language == 'ru':
                words[j] = morph_ru.parse(words[j])[0].normal_form
            if language == 'en':
                words[j] = morph_en.stem(words[j])
        reviews[i] = " ".join(words)
    return reviews


def bow_core(train_reviews, test_reviews, stemming=True, removing_stopwords=False, language='ru'):
    print("Stemming...")
    if stemming:
        train_reviews = stem(train_reviews, language)
        test_reviews = stem(test_reviews, language)
    print("Vectorizing...")
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None,
                                 stop_words=common.get_stopwords(language) if removing_stopwords else None,
                                 max_features=10000)
    vectorizer.fit(train_reviews+test_reviews)
    train_data = vectorizer.transform(train_reviews).toarray()
    test_data = vectorizer.transform(test_reviews).toarray()
    return train_data, test_data


def bow(train, test, output, stemming=True, removing_stopwords=False):
    train_reviews, train_answer, train_additional_features = common.preprocess_data(
        common.semeval_get_data(train))
    test_reviews, test_answer, test_additional_features = common.preprocess_data(
        common.semeval_get_data(test))

    train_data, test_data = bow_core(train_reviews, test_reviews, stemming, removing_stopwords)

    answer = common.rf_fit_predict(train_data, train_answer, test_data)
    result = common.evaluate(test_answer, answer)
    with open(output, 'w') as f:
        f.write(result)

# bow('datasets/ABSA16_Restaurants_Ru_Train.xml', 'datasets/ABSA16_Restaurants_Ru_Test.xml', "results/SemEval16RuRest/bow_baseline.log", False)
# bow('datasets/ABSA16_Restaurants_Ru_Train.xml', 'datasets/ABSA16_Restaurants_Ru_Test.xml', "results/SemEval16RuRest/bow_stemming.log", True)

