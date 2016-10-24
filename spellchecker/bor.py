import pickle
import pandas as pd
import struct
from utils.preprocess import text_to_wordlist


class BorNode:
    def __init__(self):
        self.frequency = 0
        self.children = dict()

    def increment(self):
        self.frequency += 1


class Bor:
    def __init__(self):
        self.root = BorNode()

    def process(self, word):
        current = self.root
        for i in range(len(word)):
            if current.children.get(word[i]) is None:
                current.children[word[i]] = BorNode()
            current = current.children[word[i]]
            if i == len(word) - 1:
                current.increment()

    def check(self, word):
        current = self.root
        for i in range(len(word)):
            if current.children.get(word[i]) is None:
                return False
            current = current.children[word[i]]
        return True

    def serialize(self, file):
        current = self.root
        file.write(self._serialize(current)[1])

    def deserialize(self, file):
        info = file.read()
        print(info)

    def _serialize(self, current):
        info = b""
        info += current.frequency.to_bytes(4, byteorder='big')
        info += len(current.children.keys()).to_bytes(4, byteorder='big')
        size = 8
        child_infos = []
        for k, v in current.children.items():
            child_size, child_info = self._serialize(v)
            child_infos.append(child_info)
            info += k.encode('utf-16')
            info += size.to_bytes(4, byteorder='big')
            size += child_size
        for child_info in child_infos:
            info += child_info
        return size, info


def check_spelling():
    b = Bor()
    count = 0
    with open("bor_dump.pickle", "rb") as f:
         b = pickle.load(f)
    with open("texts/VK_test.txt", "r", encoding="utf-8") as f:
        with open("answer_bor.txt", "w", encoding="utf-8") as o:
            words = text_to_wordlist(f.read(), cyrillic=True)
            unique = set()
            for word in words:
                if not b.check(word):
                    unique.add(word)
            for word in unique:
                o.write(word + "\n")


def main():
    check_spelling()
    # b = Bor()
    # b.process("cat")
    # b.process("cahis")
    # b.process("dog")
    # b.process("cat")
    # with open("bor_dump.my", "wb") as f:
    #     b.serialize(f)
    # p = Bor()
    # with open("bor_dump.my", "rb") as f:
    #     p.deserialize(f)

    # with open("bor_dump.pickle", "wb") as f:
    #     pickle.dump(b, f)
    # print(b.root.children['c'].children['a'].children['t'].frequency)

main()