#!/usr/bin/env python3
"""This module has the function bag_of_words"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    This function creates a bag of words
    sentences is a list of sentences to analyze
    vocab is a list of the vocabulary words to use for the analysis
    If None, all words within sentences should be used
    Returns: embeddings, features
    embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
    s is the number of sentences in sentences
    f is the number of features analyzed
    features is a list of the features used for embeddings
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    embedded = vectorizer.get_feature_names()
    features = X.toarray()
    return embedded, list(features)
