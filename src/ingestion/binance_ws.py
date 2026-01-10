"""
Binance WebSocket Price Ingestion

This script connects to Binance's public WebSocket API and streams
live cryptocurrency price data into Redis.

Architecture:
    Binance WebSocket → This Script → Redis List (raw:prices)
    
No API key needed! Binance provides public market data for free.

Usage:
    python -m src.ingestion.binance_ws
"""

import asyncio
import json
import signal
import sys
import time
from datetime import datetime
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from src.storage.redis_client import get_redis_client


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

# Flag to gracefully shutdown when user presses Ctrl+C
shutdown_flag = False

# Statistics for monitoring
stats = {
    "messages_received": 0,
    "messages_pushed": 0,
    "errors": 0,
    "start_time": time.time()
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_binance_trade_message(message: str) -> Optional[dict]:
    """
    Parse a Binance trade message and extract relevant fields.
    
    Binance sends JSON like:
    {
        "e": "trade",           # Event type
        "s": "BTCUSDT",         # Symbol
        "p": "43250.50",        # Price
        "q": "0.001",           # Quantity
        "T": 1704672000000      # Trade time (milliseconds)
    }
    
    We want to extract:
    - symbol
    - price
    - volume (quantity)
    - timestamp
    
    Args:
        message: Raw JSON string from WebSocket
        
    Returns:
        Dictionary with cleaned data, or None if parsing fails
    """
    try:
        data = json.loads(message)
        
        # Extract the fields we care about
        cleaned_data = {
            "symbol": data.get("s"),                    # BTCUSDT
            "price": data.get("p"),                     # "43250.50"
            "volume": data.get("q"),                    # "0.001"
            "timestamp": data.get("T"),                 # 1704672000000
            "event_type": data.get("e"),                # "trade"
            "received_at": int(time.time() * 1000)      # When we received it
        }
        
        return cleaned_data
        
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON: {e}")
        stats["errors"] += 1
        return None
    except Exception as e:
        print(f"❌ Unexpected error parsing message: {e}")
        stats["errors"] += 1
        return None


def format_timestamp(timestamp_ms: int) -> str:
    """
    Convert millisecond timestamp to readable format.
    
    Args:
        timestamp_ms: Unix timestamp in milliseconds
        
    Returns:
        Formatted string like "2024-01-10 15:30:45"
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def print_statistics():
    """Print current statistics."""
    elapsed = time.time() - stats["start_time"]
    rate = stats["messages_received"] / elapsed if elapsed > 0 else 0
    
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Messages received:  {stats['messages_received']}")
    print(f"Messages pushed:    {stats['messages_pushed']}")
    print(f"Errors:             {stats['errors']}")
    print(f"Runtime:            {elapsed:.1f} seconds")
    print(f"Message rate:       {rate:.2f} msg/sec")
    print("="*60 + "\n")


# ============================================================================
# SIGNAL HANDLERS (for graceful shutdown)
# ============================================================================

def signal_handler(sig, frame):
    """
    Handle Ctrl+C gracefully.
    
    When user presses Ctrl+C, we want to:
    1. Stop receiving new messages
    2. Print final statistics
    3. Close connections cleanly
    4. Exit
    """
    global shutdown_flag
    print("\n\n🛑 Shutdown signal received (Ctrl+C)")
    print("Stopping gracefully...")
    shutdown_flag = True


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


# ============================================================================
# MAIN WEBSOCKET LOGIC
# ============================================================================

async def consume_binance_stream(redis_client):
    """
    Connect to Binance WebSocket and consume the price stream.
    
    This is the main async function that:
    1. Connects to Binance WebSocket
    2. Listens for messages
    3. Parses each message
    4. Pushes to Redis
    5. Handles errors and reconnects
    
    Args:
        redis_client: RedisClient instance
    """
    reconnect_attempts = 0
    
    while not shutdown_flag and reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
        try:
            print(f"\n🔌 Connecting to Binance WebSocket...")
            print(f"   URL: {BINANCE_WS_URL}")
            print(f"   Symbol: BTCUSDT (Bitcoin/USDT)")
            print(f"   Stream: Trade (every transaction)")
            print()
            
            # Connect to WebSocket
            # This is an async context manager - connection stays open
            async with websockets.connect(BINANCE_WS_URL) as websocket:
                print("✅ Connected to Binance!")
                print(f"💾 Pushing data to Redis list: {REDIS_LIST_NAME}")
                print(f"📊 Logging every {LOG_EVERY_N_MESSAGES} messages\n")
                
                reconnect_attempts = 0  # Reset on successful connection
                
                # Main message loop
                while not shutdown_flag:
                    try:
                        # Wait for next message from WebSocket
                        # This is async - doesn't block the event loop
                        message = await websocket.recv()
                        
                        # Update counter
                        stats["messages_received"] += 1
                        
                        # Parse the message
                        data = parse_binance_trade_message(message)
                        
                        if data:
                            # Push to Redis
                            success = redis_client.push_to_list(REDIS_LIST_NAME, data)
                            
                            if success:
                                stats["messages_pushed"] += 1
                                
                                # Log periodically (not every message - too much spam)
                                if stats["messages_received"] % LOG_EVERY_N_MESSAGES == 0:
                                    print(f"✅ [{stats['messages_received']}] "
                                          f"BTC: ${data['price']} | "
                                          f"Vol: {data['volume']} | "
                                          f"Time: {format_timestamp(data['timestamp'])}")
                    
                    except ConnectionClosed:
                        print("⚠️  WebSocket connection closed")
                        break  # Exit inner loop, trigger reconnect
                        
                    except Exception as e:
                        print(f"❌ Error processing message: {e}")
                        stats["errors"] += 1
                        # Continue receiving messages despite error
                        
        except ConnectionClosed:
            if not shutdown_flag:
                reconnect_attempts += 1
                print(f"\n⚠️  Connection lost. Reconnecting in {RECONNECT_DELAY_SECONDS}s "
                      f"(attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS})...")
                await asyncio.sleep(RECONNECT_DELAY_SECONDS)
                
        except WebSocketException as e:
            if not shutdown_flag:
                reconnect_attempts += 1
                print(f"\n❌ WebSocket error: {e}")
                print(f"   Reconnecting in {RECONNECT_DELAY_SECONDS}s "
                      f"(attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS})...")
                await asyncio.sleep(RECONNECT_DELAY_SECONDS)
                
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            stats["errors"] += 1
            if not shutdown_flag:
                reconnect_attempts += 1
                await asyncio.sleep(RECONNECT_DELAY_SECONDS)
    
    # If we exit the loop
    if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
        print(f"\n❌ Max reconnection attempts ({MAX_RECONNECT_ATTEMPTS}) reached. Exiting.")
    
    print_statistics()


# ============================================================================
# ENTRY POINT
# ============================================================================

async def main():
    """
    Main entry point for the script.
    
    Sets up:
    1. Redis connection
    2. WebSocket consumer
    3. Graceful shutdown
    """
    print("\n" + "="*60)
    print("BINANCE WEBSOCKET PRICE INGESTION")
    print("="*60)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nPress Ctrl+C to stop gracefully\n")
    
    try:
        # Create Redis client
        print("📡 Initializing Redis client...")
        redis_client = get_redis_client()
        
        # Test connection
        if not redis_client.ping():
            print("❌ Failed to connect to Redis. Check your .env file!")
            sys.exit(1)
        
        # Start consuming the stream
        await consume_binance_stream(redis_client)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print_statistics()
        sys.exit(1)
    
    print("\n👋 Shutdown complete. Goodbye!\n")


if __name__ == "__main__":
    # Run the async main function
    # asyncio.run() handles the event loop for us
    asyncio.run(main())