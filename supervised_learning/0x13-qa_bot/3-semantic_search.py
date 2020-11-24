#!/usr/bin/env python3
"""
This module has the semantic_search(corpus_path, sentence) method
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
from os import listdir
import numpy as np


def semantic_search(corpus_path, sentence):
    """
    This method seacrh in path similarities to return the reference text
    """
    a = sentence[:-1].split(" ")
    clean = ["are", "is", "does", "do", "Where", "When", "What"]
    words_to_compare = []
    for word in a:
        if word not in clean:
            words_to_compare.append(word)
    files = listdir(corpus_path)
    lines = {}
    for file in files:
        if file[-3:] in [".md"]:
            with open('ZendeskArticles/' + file, 'r', encoding='UTF-8') as f:
                f_line = f.read().replace("\n", " ").replace("\'", " ")
            lines[file] = f_line
    print(lines)
    messages = [sentence]
    messages.extend(lines.values())
    print(messages)

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)

    def embed(input):
        return model(input)



    # print(words_to_compare)
    # print(listdir(corpus_path), a)
