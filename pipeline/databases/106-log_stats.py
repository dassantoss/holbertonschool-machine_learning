#!/usr/bin/env python3
"""
Script that provides stats about Nginx logs stored in MongoDB
Including top 10 most present IPs
"""
from pymongo import MongoClient


def print_nginx_stats():
    """
    Provides stats about Nginx logs stored in MongoDB
    Including top 10 most present IPs
    """
    # Connect to MongoDB
    client = MongoClient('mongodb://127.0.0.1:27017')

    # Get the nginx collection
    nginx_collection = client.logs.nginx

    # Get total number of logs
    total_logs = nginx_collection.count_documents({})
    print(f"{total_logs} logs")

    # Print methods stats
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        count = nginx_collection.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")

    # Get number of status check
    status_check = nginx_collection.count_documents({
        "method": "GET",
        "path": "/status"
    })
    print(f"{status_check} status check")

    # Get top 10 IPs
    print("IPs:")
    pipeline = [
        {"$group": {"_id": "$ip", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    top_ips = nginx_collection.aggregate(pipeline)

    for ip_info in top_ips:
        print(f"\t{ip_info['_id']}: {ip_info['count']}")


if __name__ == "__main__":
    print_nginx_stats()
