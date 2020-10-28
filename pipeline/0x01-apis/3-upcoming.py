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
    all_data = get_data("https://api.spacexdata.com/v4/launches/upcoming")
    time_c = all_data[0]["date_unix"]
    pos = 0
    for i, data in enumerate(all_data):
        if time_c > data["date_unix"]:
            time_c = data["date_unix"]
            pos = i
    url = "https://api.spacexdata.com/v4/launchpads/" + \
          all_data[pos]["launchpad"]
    data_lp = get_data(url)
    url2 = "https://api.spacexdata.com/v4/rockets/" + all_data[pos]["rocket"]
    data_r = get_data(url)
    print("{} ({}) {} - {} ({})".format(all_data[pos]["name"],
                                        all_data[pos]["date_local"],
                                        data_r["name"],
                                        data_lp["name"],
                                        data_lp["locality"]
                                        ))
