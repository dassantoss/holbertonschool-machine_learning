#!/usr/bin/env python3
"""Script to fetch the location of a specific GitHub user."""
import requests
import sys
from datetime import datetime


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API URL>")
        sys.exit(1)

    url = sys.argv[1]

    response = requests.get(url)

    if response.status_code == 200:
        # User exists, print their location
        user_data = response.json()
        location = user_data.get('location', 'No location provided')
        print(location)
    elif response.status_code == 404:
        # User not found
        print("Not found")
    elif response.status_code == 403:
        # Rate limit reached, calculate reset time
        reset_time = int(response.headers.get('X-Ratelimit-Reset', 0))
        current_time = datetime.now()
        reset_datetime = datetime.fromtimestamp(reset_time)
        minutes_to_reset = \
            (reset_datetime - current_time).total_seconds() // 60
        print(f"Reset in {int(minutes_to_reset)} min")
    else:
        # Other errors
        print(f"Error: {response.status_code}")
