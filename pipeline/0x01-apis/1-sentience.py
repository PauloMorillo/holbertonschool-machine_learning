#!/usr/bin/env python3
"""
This module has the sentientPlanets(): method
"""

import requests


def get_data(url):
    """
    This method return all data
    """
    data = requests.get(url)
    data = data.json()
    return data


def sentientPlanets():
    """
    this method returns the list of ships that can hold a given
    number of passengers from Swapi API
    """
    data = get_data("https://swapi-api.hbtn.io/api/species/")
    sentient_planets = []
    while data["next"]:
        for specie in data["results"]:
            if "sentient" in [specie["designation"], specie["classification"]]:
                if specie["homeworld"]:
                    planet = get_data(specie["homeworld"])
                    sentient_planets.append(planet["name"])
        data = get_data(data["next"])
    return sentient_planets
