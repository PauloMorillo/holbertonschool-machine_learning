#!/usr/bin/env python3
"""
This script has the method
uni_bleu(references, sentence):
"""
from nltk.translate.bleu_score import sentence_bleu


def uni_bleu(references, sentence):
    """ This method has the calculates the unigram BLEU score """
    score = sentence_bleu(reference, candidate)
    return score
