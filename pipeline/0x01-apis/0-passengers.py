#!/usr/bin/env python3
"""
This module has the availableShips(passengerCount): method
"""

import requests


def get_data(url):
    """
    This method return all data
    """
    data = requests.get(url)
    data = data.json()
    return data


def availableShips(passengerCount):
    """
    this method returns the list of ships that can hold a given
    number of passengers from Swapi API
    """
    data = get_data('https://swapi-api.hbtn.io/api/starships/')
    available_ships = []
    while data["next"]:
        for ship in data["results"]:
            passenger = ship["passengers"].replace(",", "")
            if passenger.isnumeric():
                n_passengers = int(passenger)
                if n_passengers >= passengerCount:
                    available_ships.append(ship["name"])
        data = get_data(data["next"])
    return available_ships
