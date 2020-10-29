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
    data = data.json()
    return data


if __name__ == '__main__':
    all_data = get_data("https://api.spacexdata.com/v4/launches/")
    rocket_c = dict()
    for data in all_data:
        if data["rocket"] in rocket_c.keys():
            rocket_c[data["rocket"]] += 1
        else:
            rocket_c[data["rocket"]] = 1
    rocket_c_names = list()
    for key, value in rocket_c.items():
        url = "https://api.spacexdata.com/v4/rockets/" + key
        data = get_data(url)
        rocket_c_names.append((data["name"], value))
    order_launch = sorted(
        rocket_c_names,
        key=lambda rocket_c_name: rocket_c_name[1],
        reverse=True
    )
    for data in order_launch:
        print("{}: {}".format(data[0], data[1]))
