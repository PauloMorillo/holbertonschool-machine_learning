#!/usr/bin/env python3
""" script that prints the location of a specific github user"""
import requests
import sys
import time


if __name__ == '__main__':
    argv = sys.argv
    argc = len(argv)
    if argc > 1:
        page_url = argv[1]

        res = requests.get(page_url)
        status_code = res.status_code
        if status_code == 200:
            print(res.json()['location'])
        elif status_code == 404:
            print('Not found')
        elif status_code == 403:
            reset = res.headers['X-Ratelimit-Reset']
            X = int((int(reset) - int(time.time())) / 60)
            print("Reset in {} min".format(X))