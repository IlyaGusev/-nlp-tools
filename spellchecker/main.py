from utils.preprocess import text_to_wordlist
from bs4 import BeautifulSoup
import os
from os.path import isfile, join
from operator import itemgetter


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


def dict_to_csv(d, filename):
    with open(filename, "w") as f:
        f.write("sep=,\n")
        for (k, v) in d:
            f.write(k+","+str(v)+"\n")


def main():
    freqs, reversed_freqs, unique_count, count = get_freq(get_file_list("texts", ".fb2"))
    print("Total unique words: " + str(unique_count))
    print("Total words: " + str(count))

    dict_to_csv(sorted(freqs.items(), key=itemgetter(0)), "alphabet.csv")
    dict_to_csv(reversed(sorted(freqs.items(), key=itemgetter(1))), "freq.csv")
    dict_to_csv([(k[::-1], v) for (k,v) in sorted(reversed_freqs.items(), key=itemgetter(0))], "reversed.csv")

main()