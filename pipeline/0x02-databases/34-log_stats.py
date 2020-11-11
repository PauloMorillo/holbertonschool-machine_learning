#!/usr/bin/env python3
"""
This script provides some stats about Nginx logs stored in MongoDB
"""

from pymongo import MongoClient

if __name__ == "__main__":

    client = MongoClient('mongodb://127.0.0.1:27017')
    collection_logs = client.logs.nginx
    num_docs = collection_logs.count_documents({})
    print("{} logs".format(num_docs))
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        num_met = collection_logs.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, num_met))

    my_dicti = {"method": "GET", "path": "/status"}

    num_dicti = collection_logs.count_documents(my_dicti)
    print("{} status check".format(num_dicti))
