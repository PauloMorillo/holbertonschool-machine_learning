#!/usr/bin/env python3
"""
This script has the list_all method
"""


def list_all(mongo_collection):
    """
    Return an empty list if no document in the collection
    """

    documents = []

    list_all = mongo_collection.find()

    for elem in list_all:
        documents.append(elem)

    return documents
