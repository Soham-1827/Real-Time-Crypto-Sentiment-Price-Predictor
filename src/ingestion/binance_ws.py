import asyncio
import json
import signal
import sys
import time
from datetime import datetime
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
from src.storage.redis_client import RedisClient

# ============================================================================
# CONFIGURATION
# ============================================================================

# Binance WebSocket endpoint
# Format: wss://stream.binance.com:9443/ws/{symbol}@{stream_type}
# We're using @trade which gives us every trade that happens
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"

# Redis list name where we'll store the data
REDIS_LIST_NAME = "raw:prices"

# How many messages to show before summarizing (to avoid spam)
LOG_EVERY_N_MESSAGES = 10

# Reconnection settings
RECONNECT_DELAY_SECONDS = 5
MAX_RECONNECT_ATTEMPTS = 10

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Flag to gracefully shutdown
shutdown_flag = False

# Statistics for monitoring
stats = {
    "messages_received": 0,
    "messages_pushed": 0,
    "start_time": time.time(),
    "errors": 0
}

def parse_binance_trade_message(message: str) -> Optional[dict]:
    """
    Parse a trade message from Binance WebSocket.

    Binance sends JSON like this for trades:
    {
        "e": "trade",     // Event type
        "E": 123456789,   // Event time
        "s": "BTCUSDT",   // Symbol
        "t": 12345,       // Trade ID
        "p": "0.001",     // Price
        "q": "100",       // Quantity
        "b": 88,          // Buyer order ID
        "a": 50,          // Seller order ID
        "T": 123456785,   // Trade time
        "m": true,       // Is the buyer the market maker?
        "M": true        // Ignore
    }
    Args:
        message: Raw JSON message string from WebSocket
    """
    
    try:
        data = json.loads(message)
        cleaned_data = {
            "symbol": data.get("s"),
            "price": float(data.get("p")),
            "volume": float(data.get("q")),
            "timestamp": data.get("T"),
            "event_type": data.get("e"),
            "received_at": int(time.time() * 1000)  # Current time in ms
        }
        
        return cleaned_data
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"❌ Error parsing message: {e}")
        stats["errors"] += 1
        return None
    
    
def format_timestamp(ms: int) -> str:
    """Convert milliseconds timestamp to human-readable format."""
    return datetime.fromtimestamp(ms / 1000).strftime('%Y-%m-%d %H:%M:%S')


def print_statistics():
    """Print current statistics."""
    elapsed = time.time() - stats["start_time"]
    rate = stats["messages_received"] / elapsed if elapsed > 0 else 0
    print("\n" + "=" * 40)
    print("CURRENT STATISTICS")
    print("=" * 40)
    print(f"Messages Received: {stats['messages_received']}")
    print(f"Messages Pushed to Redis: {stats['messages_pushed']}")
    print(f"Errors: {stats['errors']}")
    print(f"Elapsed Time: {elapsed:.2f} seconds")
    print(f"Receive Rate: {rate:.2f} messages/second")
    print("=" * 40 + "\n")
    
    