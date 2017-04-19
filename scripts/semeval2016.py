import random
import re
import os
import xml.etree.ElementTree as ET
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from utils.clf_pipeline import Pipeline, BowFeaturesStep, POSFeaturesStep, RawFeaturesStep, CVStep, \
    EvaluateFMeasureStep
from utils.preprocess import text_to_wordlist

random.seed(2016)


def semeval_get_data(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    data = []
    print("Num of reviews: " + str(len(root.findall(".//Review"))))
    count = 0
    for review in root.findall(".//Review"):
        rid = review.get('rid')
        for sentence_node in review.findall(".//sentence"):
            sentence = dict()
            sentence['rid'] = rid
            sentence['opinions'] = []
            sentence['text'] = sentence_node.find(".//text").text
            for opinion_node in sentence_node.findall(".//Opinion"):
                count += 1
                opinion = dict()
                opinion['from'] = int(opinion_node.get('from'))
                opinion['to'] = int(opinion_node.get('to'))
                opinion['target'] = opinion_node.get('target')
                opinion['polarity'] = opinion_node.get('polarity')
                opinion['category'] = opinion_node.get('category')
                sentence['opinions'].append(opinion)
            data.append(sentence)
    print("Num of opinions:" + str(count))
    return data


def get_separator_borders(text):
    separator_borders = []
    separators = ','
    prev = 0
    for i in range(len(text)):
        if text[i] in separators:
            separator_borders.append((prev, i))
            prev = i
    separator_borders.append((prev, len(text) - 1))
    return separator_borders


def get_word_borders(text):
    words_borders = []
    word_begin = -1
    word_end = -1
    for i in range(len(text)):
        if len(re.sub("[^а-яА-Яёa-zA-Z]", "", text[i])) != 0:
            if word_begin == -1:
                word_begin = i
            word_end = i
        else:
            if word_begin != -1:
                words_borders.append((word_begin, word_end))
            word_begin = -1
            word_end = -1
    if word_begin != -1:
        words_borders.append((word_begin, word_end))
    return words_borders


def get_opinion_borders(sentence):
    opinion_borders = []
    for opinion in sentence['opinions']:
        from_idx = int(opinion['from'])
        to_idx = int(opinion['to'])
        if from_idx != 0 or to_idx != 0:
            opinion_borders.append((from_idx, to_idx))
        else:
            opinion_borders.append((0, len(sentence['text'])))
    return opinion_borders


def get_opinions(sentence):
    opinions = []
    for opinion in sentence['opinions']:
        opinions.append(opinion)
    return opinions


def get_context(from_idx, to_idx, text, words_borders, opinion_borders, separator_borders, context_window):
    words = text_to_wordlist(text)

    begin = -1
    end = -1
    for i, border in enumerate(words_borders):
        if border[0] >= from_idx and begin == -1:
            begin = i
        if to_idx >= border[1] and begin != -1:
            end = i
    begin = 0 if begin - context_window < 0 else begin - context_window
    end = len(words) - 1 if end + context_window > len(words) - 1 else end + context_window

    b = words_borders[begin][0]
    e = words_borders[end][1]
    for border in separator_borders:
        if from_idx >= border[0] and to_idx <= border[1]:
            if border[0] > b:
                b = border[0]
            if border[1] < e:
                e = border[1]

    for border in opinion_borders:
        if border[0] < from_idx:
            if b < border[1] < from_idx:
                b = border[1]
        if border[1] > to_idx:
            if to_idx < border[0] < e:
                e = border[0] - 1
    return text[b:e+1]


def preprocess_data_for_sentiment(data, nlc_filename=None, nlc_meta_filename=None, context_window=5):
    reviews = []
    raw_reviews = []
    sentiments = []
    categories = []
    metas = []
    # Aspects
    current_rid = data[0]['rid']
    current_length = 1
    for sentence in data:
        if sentence['rid'] != current_rid:
            current_length = 1
            current_rid = sentence['rid']
        text = sentence['text'].lower()

        separator_borders = get_separator_borders(text)
        opinion_borders = get_opinion_borders(sentence)
        words_borders = get_word_borders(text)

        for opinion in sentence['opinions']:
            from_idx = int(opinion['from'])
            to_idx = int(opinion['to'])
            if opinion['target'] != 'NULL':
                context = get_context(from_idx, to_idx, text, words_borders, opinion_borders, separator_borders, context_window)
                result = text_to_wordlist(context)
                raw_reviews.append(context)
            else:
                result = text_to_wordlist(text)
                raw_reviews.append(text)
                to_idx += len(sentence['text']) + 1

            result = " ".join(result)
            # Polarity
            polarity_classes = ['negative', 'neutral', 'positive']
            if opinion['polarity'] in polarity_classes:
                reviews.append(result)
                metas.append({'rid': int(sentence['rid']), 'words': result, 'start': current_length+from_idx,
                             'end': current_length+to_idx, 'answer': polarity_classes.index(opinion['polarity']),
                              'isNull': opinion['target'] == 'NULL'})
                sentiments.append(opinion['polarity'])
                categories.append({"entity": opinion['category'].split("#")[0],
                                   "attr": opinion['category'].split("#")[1]})
        current_length += len(sentence['text']) + 1

    nlc_data = []
    if nlc_filename is not None:
        with open(nlc_filename) as f:
            nlc_data = [(" ".join(line.strip().split("; ")[1:])).replace(":", "0000") for line in f.readlines()]

        shuffle_order = []
        with open(nlc_meta_filename, encoding='utf-8') as meta_file:
            content = meta_file.readlines()
            for line in content:
                link = re.search(r'markupText\/[0-9\/]*""', line).group(0).split("/")
                nlc_end = int(link[-1][:-2])
                nlc_start = int(link[-2])
                nlc_rid = int(re.search(r'""[0-9]*#', line).group(0)[2:][:-1])
                flag = False
                for i in range(len(metas)):
                    meta = metas[i]
                    if meta['rid'] == nlc_rid:
                        if abs(nlc_start - meta['start']) <= 1 and abs(nlc_end - meta['end']) <= 1:
                            shuffle_order.append(i)
                            flag = True
                            break
                if not flag:
                    nlc_word = " ".join(text_to_wordlist(re.search(r'#.*""', line).group(0)[1:][:-2]))
                    print("ERRROR ON: " + str(nlc_rid) + " " + nlc_word)

        reviews = [reviews[i] for i in shuffle_order]
        raw_reviews = [raw_reviews[i] for i in shuffle_order]
        sentiments = [sentiments[i] for i in shuffle_order]
        categories = [categories[i] for i in shuffle_order]
    count = Counter()
    for sentiment in sentiments:
        count[sentiment] += 1
    print(count)

    # Sentiment encoding
    sent_le = LabelEncoder()
    sentiments = sent_le.fit_transform(sentiments)

    # Category
    category_dv = DictVectorizer()
    additional_features = category_dv.fit_transform(categories)

    return reviews, sentiments, additional_features, nlc_data, raw_reviews


def preprocess_data_for_aspect_category(data, context_window=5):
    reviews = []
    raw_reviews = []
    answers = []
    current_rid = data[0]['rid']
    current_length = 1
    for sentence in data:
        if sentence['rid'] != current_rid:
            current_length = 1
            current_rid = sentence['rid']
        text = sentence['text'].lower()

        separator_borders = get_separator_borders(text)
        opinion_borders = get_opinion_borders(sentence)
        opinions = get_opinions(sentence)
        words_borders = get_word_borders(text)

        for i, opinion in enumerate(opinions):
            from_idx = int(opinion['from'])
            to_idx = int(opinion['to'])
            context = text
            if opinion['target'] != 'NULL':
                context = get_context(from_idx, to_idx, text, words_borders, opinion_borders, separator_borders,
                                      context_window)
            result = " ".join(text_to_wordlist(context))
            category = opinion['category']
            raw_reviews.append(context)
            reviews.append(result)
            answers.append(category)

        # for word_begin, word_end in words_borders:
        #     context = get_context(word_begin, word_end, text, words_borders, opinion_borders,
        #                           separator_borders, context_window)
        #     result = text_to_wordlist(context)
        #     result = " ".join(result)
        #     for i, (opinion_begin, opinion_end) in enumerate(opinion_borders):
        #         if word_begin >= opinion_begin and word_end <= opinion_end:
        #             category = opinions[i]['category']
        #             raw_reviews.append(context)
        #             reviews.append(result)
        #             answers.append(category)
        current_length += len(sentence['text']) + 1
    answers_le = LabelEncoder()
    answers = answers_le.fit_transform(answers)
    return reviews, raw_reviews, answers


def sentiment_pipeline(train_filename, test_filename, stemming=True, context_window=5, bow_ngrams=(1, 2), pos_ngrams=(1, 1)):
    print("Preprocessing...")
    train_reviews, train_answer, train_additional_features, nlc_train_data, raw_train_reviews = \
        preprocess_data_for_sentiment(semeval_get_data(train_filename), context_window=context_window)
    test_reviews, test_answer, test_additional_features, nlc_test_data, raw_test_reviews = \
        preprocess_data_for_sentiment(semeval_get_data(test_filename), context_window=context_window)

    pipeline = Pipeline(train_reviews, test_reviews)
    pipeline.add_step(BowFeaturesStep(language='ru', stem=stemming, tokenizer=text_to_wordlist, preprocessor=None,
                                      use_tfidf=True, max_features=None, bow_ngrams=bow_ngrams))
    pipeline.add_step(POSFeaturesStep(language='ru', stem=False, tokenizer=text_to_wordlist, preprocessor=None,
                                      use_tfidf=False, max_features=None, pos_ngrams=pos_ngrams))
    pipeline.add_step(RawFeaturesStep(train_additional_features, test_additional_features))
    clf = LinearSVC(tol=0.1)
    pipeline.add_step(CVStep(train_answer, clf, n=20, test_size=0.25, scoring=None, random_state=2016))
    pipeline.add_step(EvaluateFMeasureStep(clf, train_answer, test_answer))
    pipeline.run()

    # # NLC
    # print("NLC...")
    # nlc_train_data, nlc_test_data = bow(nlc_train_data, nlc_test_data, stem=False, use_tfidf=False,
    #                                     tokenizer=None, bow_ngrams=(1, 1))
    # train_data = hstack([train_data, nlc_train_data])
    # test_data = hstack([test_data, nlc_test_data])

    # answer = run_single(train_data, test_data, train_answer, 201600)


def category_pipeline(train_filename, test_filename, stemming=True, context_window=5, bow_ngrams=(1, 2), pos_ngrams=(1, 1)):
    print("Preprocessing...")
    train_reviews, raw_train_reviews, train_answer = \
        preprocess_data_for_aspect_category(semeval_get_data(train_filename), context_window=context_window)
    test_reviews, raw_test_reviews, test_answer = \
        preprocess_data_for_aspect_category(semeval_get_data(test_filename), context_window=context_window)

    pipeline = Pipeline(train_reviews, test_reviews)
    pipeline.add_step(BowFeaturesStep(language='ru', stem=stemming, tokenizer=text_to_wordlist, preprocessor=None,
                                      use_tfidf=True, max_features=None, bow_ngrams=bow_ngrams))
    pipeline.add_step(POSFeaturesStep(language='ru', stem=False, tokenizer=text_to_wordlist, preprocessor=None,
                                      use_tfidf=False, max_features=None, pos_ngrams=pos_ngrams))
    clf = LinearSVC(tol=0.1)
    pipeline.add_step(CVStep(train_answer, clf, n=20, test_size=0.25, scoring=None, random_state=2016))
    pipeline.add_step(EvaluateFMeasureStep(clf, train_answer, test_answer))
    pipeline.run()

if __name__ == '__main__':
    # sentiment_pipeline(os.path.join('datasets', 'ABSA16_Restaurants_Ru_Train.xml'),
    #                    os.path.join('datasets', 'ABSA16_Restaurants_Ru_Test.xml'),
    #                    True, context_window=7)
    category_pipeline(os.path.join('datasets', 'ABSA16_Restaurants_Ru_Train.xml'),
                      os.path.join('datasets', 'ABSA16_Restaurants_Ru_Test.xml'),
                      True, context_window=0)
