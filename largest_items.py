import boto3
import json
import sys
from botocore.exceptions import ClientError

def find_largest_items(table_name: str, limit: int = 10):
    dynamodb = boto3.client("dynamodb")
    paginator = dynamodb.get_paginator("scan")

    max_items = []
    total_count = 0

    try:
        for page in paginator.paginate(TableName=table_name):
            for item in page.get("Items", []):
                total_count += 1
                # Calculate size of the item in bytes
                size_bytes = len(json.dumps(item).encode("utf-8"))

                max_items.append((size_bytes, item))
                # Keep only top N
                max_items = sorted(max_items, key=lambda x: x[0], reverse=True)[:limit]

        print(f"Scanned {total_count} items from table '{table_name}'")
        print(f"Top {limit} largest items:")

        for rank, (size, item) in enumerate(max_items, start=1):
            print(f"\n#{rank} — {size} bytes")
            print(json.dumps(item, indent=2))

    except ClientError as e:
        print(f"Error scanning table: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python largest_items.py <TableName> [limit]")
        sys.exit(1)

    table_name = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    find_largest_items(table_name, limit)
    