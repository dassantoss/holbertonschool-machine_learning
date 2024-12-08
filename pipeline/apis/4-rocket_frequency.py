#!/usr/bin/env python3
"""Script to display the number of launches per rocket."""
import requests
from collections import defaultdict


if __name__ == '__main__':
    # Get all launches
    launches_url = "https://api.spacexdata.com/v4/launches"
    launches_response = requests.get(launches_url)

    if launches_response.status_code != 200:
        print("Error fetching launches data")
        exit(1)

    launches = launches_response.json()

    # Count launches per rocket
    rocket_counts = defaultdict(int)
    rocket_names = {}

    # Get all rocket IDs and count launches
    for launch in launches:
        rocket_id = launch.get('rocket')
        if rocket_id:
            rocket_counts[rocket_id] += 1

            # Get rocket name if we haven't already
            if rocket_id not in rocket_names:
                rocket_url = \
                    f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
                rocket_response = requests.get(rocket_url)
                if rocket_response.status_code == 200:
                    rocket_data = rocket_response.json()
                    rocket_names[rocket_id] = rocket_data.get('name',
                                                              'Unknown')

    # Create list of (name, count) tuples
    rocket_list = [(rocket_names.get(rid, 'Unknown'), count)
                   for rid, count in rocket_counts.items()]

    # Sort by count (descending) and name (ascending)
    rocket_list.sort(key=lambda x: (-x[1], x[0]))

    # Print results
    for name, count in rocket_list:
        print(f"{name}: {count}")
