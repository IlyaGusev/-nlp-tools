import re
from pymorphy2 import MorphAnalyzer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk import pos_tag

morph_ru = MorphAnalyzer()
morph_en = SnowballStemmer("english")


def text_to_wordlist(sentence):
    sentence = re.sub("[^а-яА-Яёa-zA-Z]"," ", sentence)
    result = sentence.lower().split()
    return result


def get_stopwords(lang):
    if lang == 'ru':
        stop_words = stopwords.words('russian')[:50]
        stop_words = [x for x in stop_words if x not in
                      ['не', 'но', 'нет', 'только', 'а', 'даже']]
        return stop_words
    if lang == 'en':
        stop_words = stopwords.words('english')[:50]
        return stop_words
    return []


def stem_sentence(sentence, language):
    words = text_to_wordlist(sentence)
    for j in range(len(words)):
        if language == 'ru':
            words[j] = morph_ru.parse(words[j])[0].normal_form
        if language == 'en':
            words[j] = morph_en.stem(words[j])
    return " ".join(words)


def get_sentence_tags(sentence, language):
    words = text_to_wordlist(sentence)
    tags = []
    if language == 'en':
        if len(words) != 0:
            tags = [i[1] for i in pos_tag(words)]
    if language == 'ru':
        for j in range(len(words)):
            pos = morph_ru.parse(words[j])[0].tag.POS
            if pos is not None:
                tags.append(pos)
    return " ".join(tags)
