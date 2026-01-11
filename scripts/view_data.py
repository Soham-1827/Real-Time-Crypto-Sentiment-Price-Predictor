"""
View collected price data from Redis

This script shows you what data you've collected and provides
basic statistics.

Usage:
    python scripts/view_data.py
    OR
    python -m scripts.view_data
"""

import sys
from pathlib import Path

# Add project root to Python path
# This allows us to import from 'src' module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage.redis_client import get_redis_client
from datetime import datetime

def main():
    print("\n" + "="*60)
    print("VIEWING COLLECTED PRICE DATA")
    print("="*60 + "\n")
    
    # Connect to Redis
    redis = get_redis_client()
    
    # Get total count
    total = redis.get_list_length("raw:prices")
    print(f"📊 Total messages in Redis: {total}\n")
    
    if total == 0:
        print("No data yet! Run the ingestion script first:")
        print("  python -m src.ingestion.binance_ws")
        return
    
    # Get first 5 messages
    print("📥 First 5 messages:")
    print("-" * 60)
    first_msgs = redis.get_from_list("raw:prices", 0, 4)
    for i, msg in enumerate(first_msgs, 1):
        ts = datetime.fromtimestamp(msg['timestamp'] / 1000)
        print(f"{i}. ${msg['price']} at {ts.strftime('%H:%M:%S')}")
    
    # Get last 5 messages
    print(f"\n📤 Last 5 messages:")
    print("-" * 60)
    last_msgs = redis.get_from_list("raw:prices", -5, -1)
    for i, msg in enumerate(last_msgs, 1):
        ts = datetime.fromtimestamp(msg['timestamp'] / 1000)
        print(f"{i}. ${msg['price']} at {ts.strftime('%H:%M:%S')}")
    
    # Calculate some basic stats
    if len(last_msgs) > 1:
        prices = [float(msg['price']) for msg in last_msgs]
        print(f"\n📈 Recent price statistics:")
        print("-" * 60)
        print(f"Highest:  ${max(prices):.2f}")
        print(f"Lowest:   ${min(prices):.2f}")
        print(f"Range:    ${max(prices) - min(prices):.2f}")
        print(f"Latest:   ${prices[-1]:.2f}")
    
    print("\n" + "="*60)
    print("✅ Data looks good!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
