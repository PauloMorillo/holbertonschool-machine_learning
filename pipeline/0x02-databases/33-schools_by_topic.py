#!/usr/bin/env python3
"""
This script has the schools_by_topic method
"""


def schools_by_topic(mongo_collection, topic):
    """
    this method returns the list of school having a specific topic
    """

    all_items = mongo_collection.find()
    documents = []
    doc_filter = []

    for elem in all_items:
        documents.append(elem)

    for elem in documents:
        if 'topics' in elem.keys():
            if topic in elem['topics']:
                doc_filter.append(elem)

    return doc_filter
