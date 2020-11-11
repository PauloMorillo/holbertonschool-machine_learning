#!/usr/bin/env python3
"""
This script has the update_topics method
"""


def update_topics(mongo_collection, name, topics):
    """
    This method changes all topics of a school document based on the name
    """

    aux = {'$set': {'topics': topics}}
    mongo_collection.update_many({'name': name}, aux)
