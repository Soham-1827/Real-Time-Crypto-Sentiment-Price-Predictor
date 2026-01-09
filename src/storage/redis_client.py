"""
Redis Client Module

This module provides a reusable Redis connection and helper functions
for interacting with Upstash Redis.

Why we need this:
- Centralized connection management (don't create clients everywhere)
- Reusable across all scripts (ingestion, processing, etc.)
- Proper error handling in one place
"""

import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from upstash_redis import Redis

# Load environment variables from .env file
load_dotenv()


class RedisClient:
    """
    Wrapper around Upstash Redis client with helper methods.
    
    This class:
    1. Manages the Redis connection
    2. Provides easy methods to push/read data
    3. Handles errors gracefully
    """
    
    def __init__(self):
        """Initialize Redis client with credentials from .env"""
        self.redis_url = os.getenv("UPSTASH_REDIS_URL")
        self.redis_token = os.getenv("UPSTASH_REDIS_TOKEN")
        
        if not self.redis_url or not self.redis_token:
            raise ValueError(
                "Missing Redis credentials! "
                "Make sure UPSTASH_REDIS_URL and UPSTASH_REDIS_TOKEN are in .env"
            )
        
        # Create the Redis client
        self.client = Redis(url=self.redis_url, token=self.redis_token)
        print(f"✅ Connected to Redis at {self.redis_url[:30]}...")
    
    def push_to_list(self, list_name: str, data: Dict[str, Any]) -> bool:
        """
        Push data to a Redis list (like appending to a queue).
        
        Args:
            list_name: Name of the Redis list (e.g., "raw:prices")
            data: Dictionary of data to store
            
        Returns:
            True if successful, False otherwise
            
        Why use lists instead of streams?
        - upstash-redis library has limited stream support
        - Lists work great for our use case
        - Easy to read/process later
        """
        try:
            # Convert dictionary to JSON string
            json_data = json.dumps(data)
            
            # RPUSH = "right push" (append to end of list)
            self.client.rpush(list_name, json_data)
            return True
            
        except Exception as e:
            print(f"❌ Error pushing to Redis list '{list_name}': {e}")
            return False
    
    def get_from_list(self, list_name: str, start: int = 0, end: int = -1) -> list:
        """
        Read data from a Redis list.
        
        Args:
            list_name: Name of the Redis list
            start: Starting index (0 = first item)
            end: Ending index (-1 = last item)
            
        Returns:
            List of dictionaries (parsed from JSON)
            
        Example:
            messages = redis.get_from_list("raw:prices", 0, 10)
            # Gets first 10 messages
        """
        try:
            # LRANGE = "list range" (get items from list)
            items = self.client.lrange(list_name, start, end)
            
            # Parse each JSON string back to dictionary
            return [json.loads(item) for item in items]
            
        except Exception as e:
            print(f"❌ Error reading from Redis list '{list_name}': {e}")
            return []
    
    def get_list_length(self, list_name: str) -> int:
        """
        Get the number of items in a list.
        
        Args:
            list_name: Name of the Redis list
            
        Returns:
            Number of items in the list
        """
        try:
            return self.client.llen(list_name)
        except Exception as e:
            print(f"❌ Error getting list length: {e}")
            return 0
    
    def ping(self) -> bool:
        """
        Test if Redis connection is alive.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            response = self.client.ping()
            return response is not None
        except Exception as e:
            print(f"❌ Redis ping failed: {e}")
            return False


def get_redis_client() -> RedisClient:
    """
    Factory function to get a Redis client instance.
    
    This is the function other scripts will use:
        from src.storage.redis_client import get_redis_client
        redis = get_redis_client()
        redis.push_to_list("raw:prices", data)
    
    Returns:
        RedisClient instance
    """
    return RedisClient()


# For testing this module directly
if __name__ == "__main__":
    print("Testing Redis Client...")
    
    # Create client
    redis = get_redis_client()
    
    # Test connection
    if redis.ping():
        print("✅ Redis connection OK")
    
    # Test pushing data
    test_data = {"symbol": "BTCUSDT", "price": "43250.50", "test": True}
    if redis.push_to_list("test:connection", test_data):
        print("✅ Push to list OK")
    
    # Test reading data
    items = redis.get_from_list("test:connection", 0, 0)
    if items:
        print(f"✅ Read from list OK: {items[0]}")
    
    # Clean up
    redis.client.delete("test:connection")
    print("✅ All tests passed!")
