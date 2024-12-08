#!/usr/bin/env python3
"""
Module to interact with SWAPI API to get planets of sentient species
"""
import requests


def sentientPlanets():
    """
    Returns list of names of home planets of all sentient species

    Returns:
        list: names of planets that are home to sentient species
    """
    planets = set()  # Using set to avoid duplicates
    url = "https://swapi.dev/api/species/"

    while url:
        # Make request to current page
        response = requests.get(url)
        data = response.json()

        # Check each species on current page
        for species in data['results']:
            # Check if species is sentient
            is_sentient = False

            # Check classification
            if 'sentient' in species['classification'].lower():
                is_sentient = True

            # Check designation
            if 'sentient' in species['designation'].lower():
                is_sentient = True

            # If species is sentient, get their homeworld
            if is_sentient and species['homeworld']:
                # Get planet name from homeworld URL
                homeworld_response = requests.get(species['homeworld'])
                if homeworld_response.ok:
                    planet_data = homeworld_response.json()
                    planets.add(planet_data['name'])

        # Get URL for next page
        url = data['next']

    return sorted(list(planets))
