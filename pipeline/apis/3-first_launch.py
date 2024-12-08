#!/usr/bin/env python3
"""Script to display the first SpaceX launch details."""
import requests
from datetime import datetime


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)

    if response.status_code != 200:
        print("Error fetching data from SpaceX API")
        exit(1)

    launches = response.json()

    # Filtrar lanzamientos con fecha válida y ordenar por date_unix
    valid_launches = [launch for launch in launches if launch.get('date_unix')]
    valid_launches.sort(key=lambda x: x['date_unix'])

    # Seleccionar el primer lanzamiento
    first_launch = valid_launches[0]

    # Obtener detalles del primer lanzamiento
    launch_name = first_launch.get('name', 'Unknown')
    date_unix = first_launch['date_unix']
    date_local = datetime.fromtimestamp(date_unix).astimezone().isoformat()
    rocket_id = first_launch.get('rocket', '')
    launchpad_id = first_launch.get('launchpad', '')

    # Obtener detalles del cohete
    rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    rocket_response = requests.get(rocket_url)
    rocket_name = "Unknown Rocket"
    if rocket_response.status_code == 200:
        rocket_data = rocket_response.json()
        rocket_name = rocket_data.get('name', 'Unknown Rocket')

    # Obtener detalles de la plataforma de lanzamiento
    launchpad_url = f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    launchpad_response = requests.get(launchpad_url)
    launchpad_name = "Unknown Launchpad"
    launchpad_locality = "Unknown Locality"
    if launchpad_response.status_code == 200:
        launchpad_data = launchpad_response.json()
        launchpad_name = launchpad_data.get('name', 'Unknown Launchpad')
        launchpad_locality = launchpad_data.get('locality', 'Unknown Locality')

    # Formatear y mostrar los detalles
    print(f"{launch_name} ({date_local}) {rocket_name} - \
        {launchpad_name} ({launchpad_locality})")