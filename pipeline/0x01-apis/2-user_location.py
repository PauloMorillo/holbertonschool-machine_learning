#!/usr/bin/env python3
"""
This module has the sentientPlanets(): method
"""

import requests
import sys


def get_data(url):
    """
    This method return all data
    """
    header = {"Accept": "application/vnd.github.v3+json"}
    data = requests.get(url, params=header)
    return data


if __name__ == '__main__':
    if sys.argv:
        data = get_data(sys.argv[1])
        if data.status_code == 404:
            print("Not found")
        elif data.status_code == 403:
            print("Reset in {} min".format(data.headers["X-Ratelimit-Reset"]))
        else:
            data = data.json()
            print(data["location"])
