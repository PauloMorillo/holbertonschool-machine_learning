#!/usr/bin/env python3
"""
This script has the insert_school method
"""


def insert_school(mongo_collection, **kwargs):
    """
    This method inserts a new document in a collection based on kwargs
    """

    mongo_collection.insert(kwargs)
    new_elem = mongo_collection.find(kwargs)
    return new_elem.__dict__['_Cursor__spec']['_id']
