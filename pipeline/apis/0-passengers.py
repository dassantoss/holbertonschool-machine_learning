#!/usr/bin/env python3
"""
Module to interact with SWAPI API to get starships based on passenger capacity
"""
import requests


def availableShips(passengerCount):
    """
    Returns list of ships that can hold a given number of passengers

    Args:
        passengerCount: minimum number of passengers the ship should hold

    Returns:
        list: names of ships that can hold the specified number of passengers
    """
    ships = []
    url = "https://swapi.dev/api/starships/"

    while url:
        # Make request to current page
        response = requests.get(url)
        data = response.json()

        # Check each ship on current page
        for ship in data['results']:
            # Get passenger capacity, handling non-numeric values
            passengers = ship['passengers'].replace(',', '')
            if passengers != 'n/a' and passengers != 'unknown':
                try:
                    if int(passengers) >= passengerCount:
                        ships.append(ship['name'])
                except ValueError:
                    continue

        # Get URL for next page
        url = data['next']

    return ships
