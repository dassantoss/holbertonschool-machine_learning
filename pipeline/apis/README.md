# API Data Collection Project

## Description
This project focuses on data collection through various APIs, which is fundamental for training Machine Learning models. The project involves working with different APIs (SWAPI, GitHub, SpaceX) to retrieve and transform data, serving as an introduction to data lake construction.

## Learning Objectives
At the end of this project, you should be able to explain:
- How to use the Python `requests` package
- How to make HTTP GET requests
- How to handle rate limiting
- How to handle pagination
- How to fetch JSON resources
- How to manipulate data from external services

## Requirements
### General
- Python 3.9 on Ubuntu 20.04 LTS
- Allowed editors: vi, vim, emacs
- All files must end with a newline
- First line of all files must be exactly `#!/usr/bin/env python3`
- Code must follow pycodestyle style (version 2.11.1)
- All modules, classes and functions must be documented
- All files must be executable

## Installation

pip install requests

## Tasks

### 0. Can I join?
**File:** `0-passengers.py`
- Create a method that returns list of Star Wars ships based on passenger capacity
- Use SWAPI API
- Handle pagination
- Return empty list if no ships available

### 1. Where I am?
**File:** `1-sentience.py`
- Create a method that returns list of home planets of all sentient species
- Use SWAPI API
- Handle pagination
- Check both classification and designation attributes

### 2. Rate me is you can!
**File:** `2-user_location.py`
- Script to print GitHub user location
- Handle cases:
  - User not found
  - Rate limit exceeded
- Display reset time when rate limited

### 3. First Launch
**File:** `3-first_launch.py`
- Display first SpaceX launch information including:
  - Launch name
  - Date (local time)
  - Rocket name
  - Launchpad details
- Format: `<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)`

### 4. How many by rocket?
**File:** `4-rocket_frequency.py`
- Display number of launches per rocket using SpaceX API
- Order by number of launches (descending)
- Use alphabetical order for rockets with same launch count
- Format: `<rocket name>: <number of launches>`

## Usage Examples

### Task 0: Available Ships
./0-passengers.py

Output shows list of ships that can hold specified number of passengers

### Task 1: Sentient Planets
./1-sentience.py

Output shows list of home planets of all sentient species

### Task 2: User Location
./2-user_location.py

Output shows GitHub user location

### Task 3: First Launch
./3-first_launch.py

Output shows first SpaceX launch information

### Task 4: Rocket Frequency
./4-rocket_frequency.py

Output shows number of launches per rocket

## Resources
- [Requests package documentation](https://docs.python-requests.org/en/latest/)
- SWAPI Documentation
- GitHub API Documentation
- SpaceX API Documentation

## Author
[Santiago Echeverri Londoño]

## License
This project is part of the Holberton School curriculum.
