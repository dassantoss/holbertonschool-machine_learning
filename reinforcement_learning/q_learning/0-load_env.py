#!/usr/bin/env python3
"""
Module to load the FrozenLake environment from Gymnasium.

This module provides a function to initialize the FrozenLake-v1 environment,
allowing customization of the map and whether or not the ice is slippery.
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the FrozenLake environment with specified options.

    Parameters:
    ----------
    desc : list of lists, optional
        Custom description of the map to load. Each sub-list represents a row
        in the map, where each character represents a type of tile:
        - 'S': Start position of the agent.
        - 'F': Frozen lake (safe to walk).
        - 'H': Hole (falling results in a game over).
        - 'G': Goal position the agent needs to reach.
        If `desc` is provided, `map_name` is ignored.

    map_name : str, optional
        The name of a pre-defined map to load, such as '4x4' or '8x8'.
        If both `desc` and `map_name` are None, the environment loads a
        randomly generated 8x8 map. Common options include:
        - '4x4': A smaller map, suitable for quick testing.
        - '8x8': A larger, more challenging map.

    is_slippery : bool, optional
        If True, the ice is slippery, making the agent's movements stochastic
        (i.e., there is a chance the agent will slide in an unintended
        direction). If False, movements are deterministic, making the agent's
        path easier to control.

    Returns:
    -------
    gym.Env
        The initialized FrozenLake environment, ready for interaction.

    Example:
    -------
    # Load a basic FrozenLake environment with no slippery ice
    env = load_frozen_lake()

    # Load a custom 3x3 map with slippery ice
    custom_map = [['S', 'F', 'F'],
                  ['F', 'H', 'H'],
                  ['F', 'F', 'G']]
    env = load_frozen_lake(desc=custom_map, is_slippery=True)

    # Load a pre-defined 4x4 map with non-slippery ice
    env = load_frozen_lake(map_name="4x4", is_slippery=False)
    """
    return gym.make('FrozenLake-v1', desc=desc, map_name=map_name,
                    is_slippery=is_slippery)
