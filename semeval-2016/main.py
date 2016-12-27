import xml.etree.ElementTree as ET
import re
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import ShuffleSplit, cross_val_score
from utils.preprocess import text_to_wordlist, get_sentence_tags
from utils.rules import punctuation_features
from utils.bow import bow
from utils.boosting import run_single

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


def preprocess_data(data, nlc_filename, nlc_meta_filename, context_window=5):
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
        words = text_to_wordlist(text)

        separator_borders = []
        separators = ','
        prev = 0
        for i in range(len(text)):
            if text[i] in separators:
                separator_borders.append((prev, i))
                prev = i
        separator_borders.append((prev, len(text)-1))

        opinion_borders = []
        for opinion in sentence['opinions']:
            from_idx = int(opinion['from'])
            to_idx = int(opinion['to'])
            if from_idx != 0 or to_idx != 0:
                opinion_borders.append((from_idx, to_idx))

        for opinion in sentence['opinions']:
            result = words
            from_idx = int(opinion['from'])
            to_idx = int(opinion['to'])

            if opinion['target'] != 'NULL':
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
                begin = -1
                end = -1
                for i in range(len(words_borders)):
                    if from_idx >= words_borders[i][0]-1 and begin == -1:
                        begin = i
                    if to_idx >= words_borders[i][1]-1 and begin != -1:
                        end = i

                begin = 0 if begin-context_window < 0 else begin-context_window
                end = len(words)-1 if end+context_window > len(words)-1 else end+context_window

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
                result = text_to_wordlist(text[b:e+1])
                raw_reviews.append(text[b:e+1])
            else:
                to_idx += len(sentence['text']) + 1
                raw_reviews.append(text)

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
    count = {"positive":0, "negative":0, "neutral":0}
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


def evaluate(test_answer, pred_answer):
    result = ""
    result += "Accuracy: " + str(accuracy_score(test_answer,  pred_answer)) + '\n'
    p_macro = precision_score(test_answer,  pred_answer, average='macro')
    r_macro = recall_score(test_answer,  pred_answer, average='macro')
    result += "F-macro: " + str(2*p_macro*r_macro/(p_macro+r_macro)) + '\n'
    result += "F-classes: " + str(f1_score(test_answer,  pred_answer, average=None)) + '\n'
    return result


def main(train, test, output, stemming=True, context_window=5, bow_ngrams=(1, 2), pos_ngrams=(1, 1), language='ru'):
    print("Preprocessing...")
    train_reviews, train_answer, train_additional_features, nlc_train_data, raw_train_reviews = \
        preprocess_data(semeval_get_data(train),
                        "datasets/ABSA16_Restaurants_Ru_Train_NLC.csv",
                        "datasets/ABSA16_Restaurants_Ru_Train_NLC_Meta.csv",
                        context_window=context_window)
    test_reviews, test_answer, test_additional_features, nlc_test_data, raw_test_reviews = \
        preprocess_data(semeval_get_data(test),
                        "datasets/ABSA16_Restaurants_Ru_Test_NLC.csv",
                        "datasets/ABSA16_Restaurants_Ru_Test_NLC_Meta.csv",
                        context_window=context_window)

    train_data = csr_matrix((len(train_reviews), 1))
    test_data = csr_matrix((len(test_reviews), 1))

    # TfIdf
    print("TfIdf...")
    bow_train_data, bow_test_data = bow(train_reviews, test_reviews, language='ru', stem=stemming, use_tfidf=True,
                                        bow_ngrams=bow_ngrams)
    train_data = hstack([train_data, bow_train_data])
    test_data = hstack([test_data, bow_test_data])

    # POS features
    print("POS...")
    pos_train_data = []
    pos_test_data = []
    for review in train_reviews:
        pos_train_data.append(get_sentence_tags(review, language))
    for review in test_reviews:
        pos_test_data.append(get_sentence_tags(review, language))
    pos_train_data, pos_test_data = bow(pos_train_data, pos_test_data, stem=False, use_tfidf=False,
                                        bow_ngrams=pos_ngrams)
    train_data = hstack([train_data, pos_train_data])
    test_data = hstack([test_data, pos_test_data])

    # # Punctuation
    # print("Punctuation...")
    # punct_train_data = punctuation_features(raw_train_reviews)
    # punct_test_data = punctuation_features(raw_test_reviews)
    # train_data = hstack([train_data,  punct_train_data])
    # test_data = hstack([test_data,  punct_test_data])

    # Category features
    print("Category...")
    train_data = hstack([train_data, train_additional_features])
    test_data = hstack([test_data, test_additional_features])

    # # NLC
    # print("NLC...")
    # nlc_train_data, nlc_test_data = bow(nlc_train_data, nlc_test_data, stem=False, use_tfidf=False,
    #                                     tokenizer=None, bow_ngrams=(1, 1))
    # train_data = hstack([train_data, nlc_train_data])
    # test_data = hstack([test_data, nlc_test_data])

    answer = run_single(train_data, test_data, train_answer, 201600)
    # clf = LinearSVC(tol=0.1)
    #
    # print("CV...")
    # cv = ShuffleSplit(train_data.shape[0], n_iter=20, test_size=0.25, random_state=10)
    # scores = cross_val_score(clf, train_data, train_answer, cv=cv)
    # print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    #
    # print("Predicting on test...")
    # clf.fit(train_data, train_answer)
    # answer = clf.predict(test_data)

    result = evaluate(test_answer, answer)
    with open(output, 'w') as f:
        f.write(result)

main('datasets/ABSA16_Restaurants_Ru_Train.xml', 'datasets/ABSA16_Restaurants_Ru_Test.xml',
     "results/SemEval16RuRest/boosting.log", True, context_window=7)