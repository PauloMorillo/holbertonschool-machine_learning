#!/usr/bin/env python3

from gensim import utils, corpora
import numpy as np


def bag_of_words(sentences, vocab=None):
    """This function creates a bag of words"""
    features = set()
    doc_tokenized = [utils.simple_preprocess(doc) for doc in sentences]
    for sentence in sentences:
        words = utils.simple_preprocess(sentence)
        [features.add(word) for word in words]
    # print(sorted(features), len(features))
    embedded = np.zeros((len(sentences), len(features)))
    dic_features = {}
    for i, feaute in enumerate(sorted(features)):
        # print(i, feaute)
        dic_features[feaute] = i

    for i, sentence in enumerate(doc_tokenized):
        for word in sentence:
            embedded[i, dic_features[word]] = int(
                embedded[i, dic_features[word]] + 1)
    return embedded, sorted(features)
