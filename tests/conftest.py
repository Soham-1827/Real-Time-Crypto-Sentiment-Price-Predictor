"""
Pytest Fixtures for Testing

This module provides reusable test fixtures including:
- Mock Redis client for testing without network
- Sample price data fixtures
- Sample sentiment data fixtures
"""

import json
import time
from collections import defaultdict
from typing import Dict, Any, List
from unittest.mock import MagicMock

import pytest


# ============================================================================
# MOCK REDIS CLIENT
# ============================================================================

class MockRedisClient:
    """
    Mock Redis client for testing.

    Implements the same interface as RedisClient but stores data in memory.
    This allows tests to run without a real Redis connection.
    """

    def __init__(self):
        """Initialize in-memory storage."""
        self._lists: Dict[str, List[str]] = defaultdict(list)
        self._keys: Dict[str, str] = {}

    def push_to_list(self, list_name: str, data: Dict[str, Any]) -> bool:
        """Push data to a mock list (mimics RPUSH)."""
        json_data = json.dumps(data)
        self._lists[list_name].append(json_data)
        return True

    def get_from_list(self, list_name: str, start: int = 0, end: int = -1) -> List[Dict]:
        """Get data from a mock list (mimics LRANGE)."""
        items = self._lists[list_name]

        # Handle negative indices like Redis
        if end == -1:
            end = len(items)
        else:
            end = end + 1  # LRANGE is inclusive

        if start < 0:
            start = max(0, len(items) + start)

        return [json.loads(item) for item in items[start:end]]

    def get_list_length(self, list_name: str) -> int:
        """Get length of a mock list (mimics LLEN)."""
        return len(self._lists[list_name])

    def ping(self) -> bool:
        """Always return True for mock."""
        return True

    def clear_list(self, list_name: str) -> None:
        """Clear a list (for test cleanup)."""
        self._lists[list_name] = []

    def clear_all(self) -> None:
        """Clear all data (for test cleanup)."""
        self._lists.clear()
        self._keys.clear()


@pytest.fixture
def mock_redis():
    """Provide a fresh mock Redis client for each test."""
    client = MockRedisClient()
    yield client
    client.clear_all()


# ============================================================================
# SAMPLE PRICE DATA
# ============================================================================

def generate_price_data(
    symbol: str = "BTCUSDT",
    base_price: float = 43000.0,
    count: int = 100,
    interval_ms: int = 1000,
    volatility: float = 0.001,
    start_timestamp: int = None
) -> List[Dict[str, Any]]:
    """
    Generate sample price data for testing.

    Args:
        symbol: Trading pair symbol
        base_price: Starting price
        count: Number of data points
        interval_ms: Milliseconds between data points
        volatility: Price change volatility (as percentage)
        start_timestamp: Starting timestamp (defaults to now - count * interval)

    Returns:
        List of price data dictionaries
    """
    import random

    if start_timestamp is None:
        start_timestamp = int(time.time() * 1000) - (count * interval_ms)

    prices = []
    current_price = base_price

    for i in range(count):
        # Add small random price change
        change = current_price * volatility * (random.random() * 2 - 1)
        current_price += change

        # Generate volume
        volume = random.uniform(0.001, 1.0)

        prices.append({
            "symbol": symbol,
            "price": f"{current_price:.2f}",
            "volume": f"{volume:.6f}",
            "timestamp": start_timestamp + (i * interval_ms),
            "event_type": "trade",
            "received_at": start_timestamp + (i * interval_ms) + 10
        })

    return prices


@pytest.fixture
def sample_price_data() -> List[Dict[str, Any]]:
    """Provide sample price data for tests."""
    return generate_price_data(count=100, volatility=0.002)


@pytest.fixture
def sample_price_data_small() -> List[Dict[str, Any]]:
    """Provide a small set of price data for quick tests."""
    return generate_price_data(count=20, volatility=0.001)


@pytest.fixture
def sample_price_data_trending_up() -> List[Dict[str, Any]]:
    """Provide price data with an upward trend for RSI testing."""
    prices = []
    base_price = 43000.0
    start_timestamp = int(time.time() * 1000) - 100000

    for i in range(30):
        # Consistently increasing price
        price = base_price + (i * 50)
        prices.append({
            "symbol": "BTCUSDT",
            "price": f"{price:.2f}",
            "volume": "0.5",
            "timestamp": start_timestamp + (i * 1000),
            "event_type": "trade",
            "received_at": start_timestamp + (i * 1000) + 10
        })

    return prices


@pytest.fixture
def sample_price_data_trending_down() -> List[Dict[str, Any]]:
    """Provide price data with a downward trend for RSI testing."""
    prices = []
    base_price = 43000.0
    start_timestamp = int(time.time() * 1000) - 100000

    for i in range(30):
        # Consistently decreasing price
        price = base_price - (i * 50)
        prices.append({
            "symbol": "BTCUSDT",
            "price": f"{price:.2f}",
            "volume": "0.5",
            "timestamp": start_timestamp + (i * 1000),
            "event_type": "trade",
            "received_at": start_timestamp + (i * 1000) + 10
        })

    return prices


# ============================================================================
# SAMPLE SENTIMENT DATA
# ============================================================================

def generate_sentiment_data(
    count: int = 50,
    interval_ms: int = 60000,  # 1 minute between posts
    start_timestamp: int = None,
    sentiment_bias: float = 0.0  # -1 to +1, affects average sentiment
) -> List[Dict[str, Any]]:
    """
    Generate sample Reddit sentiment data for testing.

    Args:
        count: Number of posts
        interval_ms: Milliseconds between posts
        start_timestamp: Starting timestamp
        sentiment_bias: Bias for sentiment (-1 to +1)

    Returns:
        List of sentiment data dictionaries
    """
    import random

    if start_timestamp is None:
        start_timestamp = int(time.time() * 1000) - (count * interval_ms)

    subreddits = ["cryptocurrency", "bitcoin", "ethereum", "CryptoMarkets"]

    # Positive and negative title templates
    positive_titles = [
        "Bitcoin hits new ATH!",
        "ETH upgrade successful",
        "Massive adoption incoming",
        "Bulls are back",
        "Great news for crypto",
    ]

    negative_titles = [
        "Market crashes hard",
        "BTC dumps 10%",
        "SEC crackdown incoming",
        "Major exchange hacked",
        "Bear market confirmed",
    ]

    neutral_titles = [
        "Daily discussion thread",
        "What are you buying?",
        "Price analysis for today",
        "New to crypto - questions",
        "Trading strategies",
    ]

    posts = []

    for i in range(count):
        # Determine sentiment based on bias
        rand = random.random() + sentiment_bias * 0.5

        if rand > 0.6:
            title = random.choice(positive_titles)
            sentiment_score = random.uniform(0.3, 1.0)
            sentiment_label = "positive"
        elif rand < 0.4:
            title = random.choice(negative_titles)
            sentiment_score = random.uniform(-1.0, -0.3)
            sentiment_label = "negative"
        else:
            title = random.choice(neutral_titles)
            sentiment_score = random.uniform(-0.2, 0.2)
            sentiment_label = "neutral"

        timestamp = start_timestamp + (i * interval_ms)

        posts.append({
            "post_id": f"test_{i:04d}",
            "subreddit": random.choice(subreddits),
            "title": title,
            "author": f"user_{i % 10}",
            "url": f"https://reddit.com/r/test/comments/test_{i}",
            "timestamp": timestamp,
            "sentiment_score": round(sentiment_score, 4),
            "sentiment_confidence": round(random.uniform(0.6, 0.95), 4),
            "sentiment_label": sentiment_label,
            "fetched_at": timestamp + 1000
        })

    return posts


@pytest.fixture
def sample_sentiment_data() -> List[Dict[str, Any]]:
    """Provide sample sentiment data for tests."""
    return generate_sentiment_data(count=50)


@pytest.fixture
def sample_sentiment_data_positive() -> List[Dict[str, Any]]:
    """Provide sentiment data with positive bias."""
    return generate_sentiment_data(count=30, sentiment_bias=0.8)


@pytest.fixture
def sample_sentiment_data_negative() -> List[Dict[str, Any]]:
    """Provide sentiment data with negative bias."""
    return generate_sentiment_data(count=30, sentiment_bias=-0.8)


# ============================================================================
# COMBINED FIXTURES
# ============================================================================

@pytest.fixture
def mock_redis_with_data(mock_redis, sample_price_data, sample_sentiment_data):
    """
    Provide a mock Redis client pre-populated with sample data.

    Returns:
        Tuple of (redis_client, price_data, sentiment_data)
    """
    # Populate price data
    for price in sample_price_data:
        mock_redis.push_to_list("raw:prices", price)

    # Populate sentiment data
    for sentiment in sample_sentiment_data:
        mock_redis.push_to_list("raw:reddit", sentiment)

    return mock_redis, sample_price_data, sample_sentiment_data


# ============================================================================
# TIMESTAMP HELPERS
# ============================================================================

@pytest.fixture
def fixed_timestamp() -> int:
    """Provide a fixed timestamp for deterministic tests."""
    return 1704672000000  # 2024-01-08 00:00:00 UTC


@pytest.fixture
def timestamp_generator(fixed_timestamp):
    """
    Factory fixture that generates sequential timestamps.

    Usage:
        def test_something(timestamp_generator):
            gen = timestamp_generator(interval_ms=1000)
            ts1 = next(gen)
            ts2 = next(gen)
    """
    def _generator(interval_ms: int = 1000):
        current = fixed_timestamp
        while True:
            yield current
            current += interval_ms

    return _generator
