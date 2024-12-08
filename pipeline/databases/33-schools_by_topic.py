#!/usr/bin/env python3
"""
Function that returns the list of schools having a specific topic
"""


def schools_by_topic(mongo_collection, topic):
    """
    Returns list of schools having a specific topic
    Args:
        mongo_collection: pymongo collection object
        topic: topic searched
    Returns:
        List of schools that have the specified topic
    """
    return list(mongo_collection.find({"topics": topic}))
