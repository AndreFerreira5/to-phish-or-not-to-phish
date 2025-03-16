import string
from collections import defaultdict
import numpy as np

class BagOfWords:
    def __init__(self, texts=None):
        self.texts = texts if texts is not None else []
        self.sanitized_texts = []
        self.words = []
        self.vocabulary = None
        self.bow = []

        self.__sanitize_text()
        self.__build_vocabulary()
        self.__words_to_bag_of_words()

    def __sanitize_text(self):
        self.words = [
            word
            for text in self.texts
            for word in text.translate(str.maketrans('', '', string.punctuation)).lower().split()
        ]

        self.sanitized_texts = [
            text.translate(str.maketrans('', '', string.punctuation)).lower().split()
            for text in self.texts
        ]


    def __build_vocabulary(self):
        self.vocabulary = {word: i for i, word in enumerate(set(self.words))}


    def __words_to_bag_of_words(self):
        for text in self.sanitized_texts:
            bag = [0] * len(self.vocabulary)
            for word in text:
                bag[self.vocabulary[word]] += 1
            self.bow.append(bag)



        '''
        bag = [[0]*len(self.vocabulary) for _ in range(len(self.sanitized_texts))]
        [
            bag[i][self.vocabulary[word]]
            for i, text in enumerate(self.texts)
            for word in text
        ]
        '''