import re
import xml.etree.ElementTree as ET
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords


def sentence_to_wordlist(sentence):
    sentence = re.sub("[^а-яА-Яёa-zA-Z]"," ", sentence)
    result = sentence.lower().split()
    return result


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


def preprocess_data(data):
    reviews = []
    sentiments = []
    additional_features = [[]]
    # Aspects
    for sentence in data:
        text = sentence['text'].lower()
        words = sentence_to_wordlist(text)

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

                context = 5
                target_words = sentence_to_wordlist(text[from_idx:to_idx])
                begin = words.index(target_words[0])
                begin = 0 if begin-context < 0 else begin-context
                end = words.index(target_words[-1])
                end = len(words)-1 if end+context > len(words)-1 else end+context

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
                result = sentence_to_wordlist(text[b:e+1])

            if result:
                result = " ".join(result)
                # Polarity
                polarity_classes = ['positive', 'negative', 'neutral']
                if opinion['polarity'] in polarity_classes:
                    reviews.append(result)
                    sentiments.append(opinion['polarity'])
                    additional_features[0].append(opinion['category'])

    sent_le = LabelEncoder()
    sentiments = sent_le.fit_transform(sentiments)
    # Category
    category_le = LabelEncoder()
    additional_features[0] = category_le.fit_transform(additional_features[0])
    return reviews, sentiments, additional_features


def rf_fit_predict(train_data, train_answer, test_data, n_estimators=100):
    forest = RandomForestClassifier(n_estimators=n_estimators)
    forest.fit(train_data, train_answer)
    answer = forest.predict(test_data)
    return answer


def evaluate(test_answer, pred_answer):
    result = ""
    result += "Accuracy: " + str(accuracy_score(test_answer,  pred_answer)) + '\n'
    p_macro = precision_score(test_answer,  pred_answer, average='macro')
    r_macro = recall_score(test_answer,  pred_answer, average='macro')
    result += "F-macro: " + str(2*p_macro*r_macro/(p_macro+r_macro)) + '\n'
    result += "F-classes: " + str(f1_score(test_answer,  pred_answer, average=None)) + '\n'
    return result


def get_stopwords(lang):
    if lang == 'ru':
        stop_words = stopwords.words('russian')[:50]
        stop_words = [x for x in stop_words if x not in
                      ['не', 'но', 'нет', 'только', 'а', 'даже']]
        return stop_words
    if lang == 'en':
        stop_words = stopwords.words('english')[:50]
        print(stop_words)
        return stop_words
    return []
