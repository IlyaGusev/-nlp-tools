import xml.etree.ElementTree as ET
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from scipy.sparse import hstack, csr_matrix
from sklearn import cross_validation

from utils.preprocess import text_to_wordlist
from utils.bow import bow, bot


def semeval_get_data(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    data = []
    for review in root.findall(".//Review"):
        for sentence_node in review.findall(".//sentence"):
            sentence = dict()
            sentence['opinions'] = []
            sentence['text'] = sentence_node.find(".//text").text
            for opinion_node in sentence_node.findall(".//Opinion"):
                opinion = dict()
                opinion['from'] = int(opinion_node.get('from'))
                opinion['to'] = int(opinion_node.get('to'))
                opinion['target'] = opinion_node.get('target')
                opinion['polarity'] = opinion_node.get('polarity')
                opinion['category'] = opinion_node.get('category')
                sentence['opinions'].append(opinion)
            data.append(sentence)
    return data


def preprocess_data(data, context_window=5):
    reviews = []
    sentiments = []
    categories = []
    # Aspects
    for sentence in data:
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
            if opinion['target'] != 'NULL':
                from_idx = int(opinion['from'])
                to_idx = int(opinion['to'])

                target_words = text_to_wordlist(text[from_idx:to_idx])
                begin = words.index(target_words[0])
                begin = 0 if begin-context_window < 0 else begin-context_window
                end = words.index(target_words[-1])
                end = len(words)-1 if end+context_window > len(words)-1 else end+context_window

                b = text.find(words[begin])
                e = text.rfind(words[end]) + len(words[end]) - 1
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

            if result:
                result = " ".join(result)
                # Polarity
                polarity_classes = ['positive', 'negative', 'neutral']
                if opinion['polarity'] in polarity_classes:
                    reviews.append(result)
                    sentiments.append(opinion['polarity'])
                    categories.append({"entity": opinion['category'].split("#")[0],
                                       "attr": opinion['category'].split("#")[1]})

    sent_le = LabelEncoder()
    sentiments = sent_le.fit_transform(sentiments)
    # Category
    category_dv = DictVectorizer()
    additional_features = category_dv.fit_transform(categories)
    return reviews, sentiments, csr_matrix(additional_features)


def evaluate(test_answer, pred_answer):
    result = ""
    result += "Accuracy: " + str(accuracy_score(test_answer,  pred_answer)) + '\n'
    p_macro = precision_score(test_answer,  pred_answer, average='macro')
    r_macro = recall_score(test_answer,  pred_answer, average='macro')
    result += "F-macro: " + str(2*p_macro*r_macro/(p_macro+r_macro)) + '\n'
    result += "F-classes: " + str(f1_score(test_answer,  pred_answer, average=None)) + '\n'
    return result


def main(train, test, output, stemming=True, context_window=5, bow_ngrams=(1, 2), pos_ngrams=(1, 1)):
    print("Preprocessing...")
    train_reviews, train_answer, train_additional_features = \
        preprocess_data(semeval_get_data(train), context_window=context_window)
    test_reviews, test_answer, test_additional_features = \
        preprocess_data(semeval_get_data(test), context_window=context_window)

    # BOW features
    train_data, test_data = bow(train_reviews, test_reviews, language='ru', stem=stemming, use_tfidf=True,
                                bow_ngrams=bow_ngrams)

    # Category features
    train_data = hstack([train_data, train_additional_features])
    test_data = hstack([test_data, test_additional_features])

    # POS features
    train_pos_data, test_pos_data = bot(train_reviews, test_reviews, language='ru', pos_ngrams=pos_ngrams)
    train_data = hstack([train_data, train_pos_data])
    test_data = hstack([test_data, test_pos_data])

    # nb = MultinomialNB(alpha=0.1)
    # nb.fit(train_data, train_answer)
    # answer = nb.predict(test_data)

    svm = LinearSVC(tol=0.1)

    print("CV...")
    cv = cross_validation.ShuffleSplit(train_data.shape[0], n_iter=20, test_size=0.125, random_state=10)
    scores = cross_validation.cross_val_score(svm, train_data, train_answer, cv=cv)
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

    print("Predicting on test...")
    svm.fit(train_data, train_answer)
    answer = svm.predict(test_data)

    # prepare_data(train_reviews, test_reviews, train_answer, "models/semeval_cnn_data.pickle",
    #              w2v_model="models/300-40-10-1e3-wiki_ru-restoran-train16-test16-1kk")
    # train_model("models/semeval_cnn_data.pickle", 'models/semeval_cnn_3epochs.model', number_of_classes=3)
    # answer = predict_answer("models/semeval_cnn_data.pickle", 'models/semeval_cnn_3epochs.model', test_answer, number_of_classes=3)

    result = evaluate(test_answer, answer)
    with open(output, 'w') as f:
        f.write(result)


# main('datasets/ABSA16_Restaurants_Ru_Train.xml', 'datasets/ABSA16_Restaurants_Ru_Test.xml',
#      "results/SemEval16RuRest/cnn.log", False)
main('datasets/ABSA16_Restaurants_Ru_Train.xml', 'datasets/ABSA16_Restaurants_Ru_Test.xml',
     "results/SemEval16RuRest/tfidf_stemming.log", True)