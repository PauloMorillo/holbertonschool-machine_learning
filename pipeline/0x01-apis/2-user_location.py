#!/usr/bin/env python3
"""
This module has get_data method
"""

import requests
import sys
import time


def get_data(url):
    """
    This method return all data
    """
    header = {"Accept": "application/vnd.github.v3+json"}
    data = requests.get(url, params=header)
    return data


if __name__ == '__main__':
    if len(sys.argv) > 1:
        data = get_data(sys.argv[1])
        if data.status_code == 404:
            print("Not found")
        elif data.status_code == 403:
            print("Reset in {} min".format(
                int((int(data.headers["X-Ratelimit-Reset"]) - time.time())
                    / 60)
            ))
        else:
            data = data.json()
            print(data["location"])
