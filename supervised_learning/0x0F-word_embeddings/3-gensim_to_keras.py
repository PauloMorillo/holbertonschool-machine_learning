#!/usr/bin/env python3
"""
This module has the gensim_to_keras method
"""


def gensim_to_keras(model):
    """
    this method get keras from gensim
    """
    return model.wv.get_keras_embedding(train_embeddings=False)
