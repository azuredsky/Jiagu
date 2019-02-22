#!/usr/bin/env python
# encoding: utf-8
"""
 * Copyright (C) 2018 OwnThink.
 *
 * Name        : mmseg.py
 * Author      : Leo <1162441289@qq.com>
 * Version     : 0.01
 * Description : mmseg分词方法，目前算法比较耗时，仍在优化中
"""
import os
import pickle
from math import log
from collections import defaultdict


def add_curr_dir(name):
    return os.path.join(os.path.dirname(__file__), name)


class Trie(object):
    def __init__(self):
        self.root = {}
        self.value = "value"
        self.trie_file_path = os.path.join(os.path.dirname(__file__), "data/Trie.pkl")

    def get_matches(self, word):
        ret = []
        node = self.root
        for c in word:
            if c not in node:
                break
            node = node[c]
            if self.value in node:
                ret.append(node[self.value])
        return ret

    def load(self):
        with open(self.trie_file_path, "rb") as f:
            data = pickle.load(f)
        self.root = data


class Chunk:
    def __init__(self, words, chrs):
        # self.sentence_sep = ['?', '!', ';', '？', '！', '。', '；', '……', '…', "，", ",", "."]
        self.words = words
        self.lens_list = map(lambda x: len(x), words)
        self.length = sum(self.lens_list)
        self.mean = float(self.length) / len(words)
        self.var = sum(map(lambda x: (x - self.mean) ** 2, self.lens_list)) / len(self.words)
        self.entropy = sum([log(float(chrs[x])) for x in words if len(x) == 1 and x in chrs])

    def __lt__(self, other):
        return (self.length, self.mean, -self.var, self.entropy) < \
               (other.length, other.mean, -other.var, other.entropy)


class MMSeg:
    def __init__(self):
        # 加载词语字典
        trie = Trie()
        trie.load()
        self.words_dic = trie
        # 加载字频字典
        self.chrs_dic = self._load_word_freq()

    def _load_word_freq(self):
        chrs_dic = defaultdict()
        with open(add_curr_dir('data/chars.dic'), "r", encoding="utf-8") as f:
            for line in f:
                if line:
                    key, value = line.strip().split(" ")
                    chrs_dic.setdefault(key, int(value))
        return chrs_dic

    def __get_start_words(self, sentence):
        match_words = self.words_dic.get_matches(sentence)
        if sentence:
            if not match_words:
                return [sentence[0]]
            else:
                return match_words
        else:
            return False

    def __get_chunks(self, sentence):
        # 获取chunk，每个chunk中最多三个词
        ret = []
        first_match_words = self.__get_start_words(sentence)
        if not first_match_words:
            return ret
        else:
            for first_word in first_match_words:
                first_tmp = [first_word]
                second_match_words = self.__get_start_words(sentence[len(first_word):])
                if not second_match_words:
                    ret.append(Chunk(first_tmp, self.chrs_dic))
                else:
                    for second_word in second_match_words:
                        first_tmp += [second_word]
                        third_match_words = self.__get_start_words(sentence[len(second_word):])
                        if not third_match_words:
                            ret.append(Chunk(first_tmp, self.chrs_dic))
                        else:
                            for third_word in third_match_words:
                                first_tmp += [third_word]
                                ret.append(Chunk(first_tmp, self.chrs_dic))
        return ret

    def cws(self, sentence):
        """
        :param sentence: 输入的数据
        :return:         返回的分词生成器
        """
        while sentence:
            chunks = self.__get_chunks(sentence)
            best = max(chunks)
            word = best.words[0]
            sentence = sentence[len(word):]
            yield word


if __name__ == "__main__":
    mmseg = MMSeg()
    print(list(mmseg.cws("武汉市长江大桥上的日落非常好看，很喜欢看日出日落。")))
    print(list(mmseg.cws("人要是行干一行行一行")))
