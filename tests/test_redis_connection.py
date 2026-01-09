import os
from dotenv import load_dotenv
from upstash_redis import Redis

load_dotenv()

def test_connection():
    """Test basic redis connection and operations"""
    redis_url = os.getenv("UPSTASH_REDIS_REST_URL")
    redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    
    if not redis_url or not redis_token:
        raise EnvironmentError("UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN must be set in environment variables")
    
    print("Connecting to Redis...")
    print(f"URL: {redis_url[:30]}")
    
    