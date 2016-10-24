from utils.preprocess import text_to_wordlist
from bs4 import BeautifulSoup
import os
import pickle
from os.path import isfile, join
from operator import itemgetter
from spellchecker.bor import Bor


def get_file_list(rel_dir="texts", ext=".fb2"):
    path = os.path.dirname(os.path.abspath(__file__)) + "\\" + rel_dir
    print(path)
    files = [path + '\\' + f for f in os.listdir(path) if isfile(join(path, f)) and ext in f]
    return files


def get_freq(files):
    all_words_freq = {}
    reversed_words_freq = {}
    count = 0
    unique_count = 0
    for filename in files:
        with open(filename, "r", encoding="utf-8", errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')
            words = text_to_wordlist(soup.get_text(), cyrillic=True)
            for word in words:
                if all_words_freq.get(word) is not None:
                    all_words_freq[word] += 1
                    reversed_words_freq[word[::-1]] += 1
                    count += 1
                else:
                    all_words_freq[word] = 1
                    reversed_words_freq[word[::-1]] = 1
                    count += 1
                    unique_count += 1
            print(filename + " done, " + str(count) + " words collected")
    return all_words_freq, reversed_words_freq, unique_count, count


def build_bor(files):
    b = Bor()
    count = 0
    for filename in files:
        with open(filename, "r", encoding="utf-8", errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')
            words = text_to_wordlist(soup.get_text(), cyrillic=True)
            for word in words:
                b.process(word)
                count += 1
        print(filename + " done, " + str(count) + " words collected")
    with open("bor_dump.pickle", "wb") as f:
        pickle.dump(b, f)


def dict_to_csv(d, filename):
    with open(filename, "w") as f:
        f.write("sep=,\n")
        for (k, v) in d:
            f.write(k+","+str(v)+"\n")


def build_ngrams():
    ngrams = set()
    with open("alphabet.csv", "r") as f:
        content = f.readlines()[1:]
        words = [line.split(',')[0] for line in content]
        for word in words:
            word = "#" + word + "#"
            for i in range(len(word)-2):
                ngrams.add(word[i:i+3])
    count = 0
    with open("texts/VK_test.txt", "r", encoding="utf-8") as f:
        with open("answer.txt", "w", encoding="utf-8") as o:
            words = text_to_wordlist(f.read(), cyrillic=True)
            for word in words:
                flag = False
                word = "#" + word + "#"
                for i in range(len(word)-2):
                    ngram = word[i:i+3]
                    if ngram not in ngrams:
                        flag = True
                        o.write(word + " "+ngram + "\n")
                if flag:
                    count += 1
    print(count)


def main():
    build_ngrams()
    # build_bor(get_file_list("texts", ".fb2"))
    # freqs, reversed_freqs, unique_count, count = get_freq(get_file_list("texts", ".fb2"))
    # print("Total unique words: " + str(unique_count))
    # print("Total words: " + str(count))
    #
    # dict_to_csv(sorted(freqs.items(), key=itemgetter(0)), "alphabet.csv")
    # dict_to_csv(reversed(sorted(freqs.items(), key=itemgetter(1))), "freq.csv")
    # dict_to_csv([(k[::-1], v) for (k,v) in sorted(reversed_freqs.items(), key=itemgetter(0))], "reversed.csv")

main()