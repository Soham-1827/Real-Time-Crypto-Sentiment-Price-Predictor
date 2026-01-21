"""
Price Feature Calculator Module

This module calculates technical indicators from raw price data:
- Moving averages (SMA, EMA)
- RSI (Relative Strength Index)
- Volatility (rolling standard deviation)
- Momentum (rate of change)
- VWAP (Volume Weighted Average Price)

Architecture:
    raw:prices data → PriceFeatureCalculator → price features dict

Usage:
    from src.processing.price_features import PriceFeatureCalculator

    calculator = PriceFeatureCalculator()

    # Add price data points
    for price_data in prices:
        calculator.add_price(price_data)

    # Get calculated features
    features = calculator.get_features()
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Deque

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Window sizes in number of data points
# Assuming 1 trade per second, these translate to:
MA_SHORT_WINDOW = 300      # 5 minutes (300 seconds)
MA_LONG_WINDOW = 900       # 15 minutes (900 seconds)
RSI_PERIOD = 14            # Standard RSI period
VOLATILITY_WINDOW = 3600   # 1 hour (3600 seconds)
MOMENTUM_WINDOW = 300      # 5 minutes

# Buffer size: keep 2 hours of data in memory
BUFFER_SIZE = 7200  # 2 hours * 60 minutes * 60 seconds


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PricePoint:
    """Single price data point."""
    symbol: str
    price: float
    volume: float
    timestamp: int  # milliseconds

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PricePoint":
        """Create PricePoint from raw price data dict."""
        return cls(
            symbol=data.get("symbol", "UNKNOWN"),
            price=float(data.get("price", 0)),
            volume=float(data.get("volume", 0)),
            timestamp=int(data.get("timestamp", 0))
        )


@dataclass
class PriceFeatures:
    """Calculated price features."""
    symbol: str
    timestamp: int
    price: float

    # Moving averages
    price_ma_5m: Optional[float] = None
    price_ma_15m: Optional[float] = None
    price_ema_5m: Optional[float] = None

    # RSI
    rsi_14: Optional[float] = None

    # Volatility and momentum
    volatility_1h: Optional[float] = None
    momentum_5m: Optional[float] = None

    # VWAP
    vwap: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "price": self.price,
            "price_ma_5m": self.price_ma_5m,
            "price_ma_15m": self.price_ma_15m,
            "price_ema_5m": self.price_ema_5m,
            "rsi_14": self.rsi_14,
            "volatility_1h": self.volatility_1h,
            "momentum_5m": self.momentum_5m,
            "vwap": self.vwap,
        }


# ============================================================================
# PRICE FEATURE CALCULATOR
# ============================================================================

class PriceFeatureCalculator:
    """
    Calculates technical indicators from price data.

    This class maintains a rolling buffer of price data and calculates
    features on demand. It's designed for streaming data processing.

    Attributes:
        buffer_size: Maximum number of price points to keep
        min_data_points: Minimum data points needed for feature calculation
    """

    def __init__(
        self,
        buffer_size: int = BUFFER_SIZE,
        ma_short: int = MA_SHORT_WINDOW,
        ma_long: int = MA_LONG_WINDOW,
        rsi_period: int = RSI_PERIOD,
        volatility_window: int = VOLATILITY_WINDOW,
        momentum_window: int = MOMENTUM_WINDOW
    ):
        """
        Initialize the calculator.

        Args:
            buffer_size: Maximum price points to keep in memory
            ma_short: Short moving average window (default 5 min)
            ma_long: Long moving average window (default 15 min)
            rsi_period: RSI calculation period (default 14)
            volatility_window: Volatility window (default 1 hour)
            momentum_window: Momentum window (default 5 min)
        """
        self.buffer_size = buffer_size
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.volatility_window = volatility_window
        self.momentum_window = momentum_window

        # Rolling buffer for price data
        self._buffer: Deque[PricePoint] = deque(maxlen=buffer_size)

        # EMA state (for incremental calculation)
        self._ema_short: Optional[float] = None
        self._ema_alpha = 2.0 / (ma_short + 1)  # EMA smoothing factor

        # RSI state
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self._prev_price: Optional[float] = None

        logger.debug(
            f"Initialized PriceFeatureCalculator: "
            f"buffer={buffer_size}, ma_short={ma_short}, ma_long={ma_long}"
        )

    def add_price(self, data: Dict[str, Any]) -> None:
        """
        Add a new price data point to the buffer.

        Args:
            data: Raw price data dictionary from Redis
        """
        point = PricePoint.from_dict(data)
        self._update_ema(point.price)
        self._update_rsi(point.price)
        self._buffer.append(point)

    def add_prices(self, data_list: List[Dict[str, Any]]) -> None:
        """
        Add multiple price data points to the buffer.

        Args:
            data_list: List of raw price data dictionaries
        """
        for data in data_list:
            self.add_price(data)

    def get_features(self) -> Optional[PriceFeatures]:
        """
        Calculate and return current features.

        Returns:
            PriceFeatures object or None if insufficient data
        """
        if len(self._buffer) == 0:
            return None

        latest = self._buffer[-1]
        prices = self._get_price_array()
        volumes = self._get_volume_array()

        features = PriceFeatures(
            symbol=latest.symbol,
            timestamp=latest.timestamp,
            price=latest.price,
            price_ma_5m=self._calculate_sma(prices, self.ma_short),
            price_ma_15m=self._calculate_sma(prices, self.ma_long),
            price_ema_5m=self._ema_short,
            rsi_14=self._get_rsi(),
            volatility_1h=self._calculate_volatility(prices, self.volatility_window),
            momentum_5m=self._calculate_momentum(prices, self.momentum_window),
            vwap=self._calculate_vwap(prices, volumes)
        )

        return features

    def get_features_dict(self) -> Optional[Dict[str, Any]]:
        """
        Get features as a dictionary.

        Returns:
            Dictionary of features or None if insufficient data
        """
        features = self.get_features()
        return features.to_dict() if features else None

    def clear(self) -> None:
        """Clear all buffered data and reset state."""
        self._buffer.clear()
        self._ema_short = None
        self._avg_gain = None
        self._avg_loss = None
        self._prev_price = None

    @property
    def buffer_length(self) -> int:
        """Return current buffer length."""
        return len(self._buffer)

    # ========================================================================
    # PRIVATE METHODS - Feature Calculations
    # ========================================================================

    def _get_price_array(self) -> np.ndarray:
        """Get numpy array of prices from buffer."""
        return np.array([p.price for p in self._buffer])

    def _get_volume_array(self) -> np.ndarray:
        """Get numpy array of volumes from buffer."""
        return np.array([p.volume for p in self._buffer])

    def _calculate_sma(
        self,
        prices: np.ndarray,
        window: int
    ) -> Optional[float]:
        """
        Calculate Simple Moving Average.

        Args:
            prices: Array of prices
            window: Number of periods

        Returns:
            SMA value or None if insufficient data
        """
        if len(prices) < window:
            return None

        return float(np.mean(prices[-window:]))

    def _update_ema(self, price: float) -> None:
        """
        Update Exponential Moving Average incrementally.

        EMA formula: EMA_today = price * alpha + EMA_yesterday * (1 - alpha)
        where alpha = 2 / (window + 1)

        Args:
            price: Latest price
        """
        if self._ema_short is None:
            self._ema_short = price
        else:
            self._ema_short = (
                price * self._ema_alpha +
                self._ema_short * (1 - self._ema_alpha)
            )

    def _update_rsi(self, price: float) -> None:
        """
        Update RSI calculation incrementally using Wilder's smoothing.

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss

        Args:
            price: Latest price
        """
        if self._prev_price is None:
            self._prev_price = price
            return

        # Calculate price change
        change = price - self._prev_price
        gain = max(0, change)
        loss = max(0, -change)

        # First calculation: simple average
        if self._avg_gain is None:
            self._avg_gain = gain
            self._avg_loss = loss
        else:
            # Wilder's smoothing (similar to EMA)
            alpha = 1.0 / self.rsi_period
            self._avg_gain = gain * alpha + self._avg_gain * (1 - alpha)
            self._avg_loss = loss * alpha + self._avg_loss * (1 - alpha)

        self._prev_price = price

    def _get_rsi(self) -> Optional[float]:
        """
        Get current RSI value.

        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if self._avg_gain is None or self._avg_loss is None:
            return None

        if len(self._buffer) < self.rsi_period:
            return None

        if self._avg_loss == 0:
            return 100.0 if self._avg_gain > 0 else 50.0

        rs = self._avg_gain / self._avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return round(rsi, 2)

    def _calculate_volatility(
        self,
        prices: np.ndarray,
        window: int
    ) -> Optional[float]:
        """
        Calculate rolling volatility (standard deviation of returns).

        Args:
            prices: Array of prices
            window: Number of periods

        Returns:
            Volatility as a decimal (0.02 = 2%) or None if insufficient data
        """
        if len(prices) < window:
            window = len(prices)

        if window < 2:
            return None

        # Calculate log returns
        window_prices = prices[-window:]
        returns = np.diff(np.log(window_prices))

        if len(returns) == 0:
            return None

        volatility = float(np.std(returns))
        return round(volatility, 6)

    def _calculate_momentum(
        self,
        prices: np.ndarray,
        window: int
    ) -> Optional[float]:
        """
        Calculate price momentum (rate of change).

        Momentum = (current_price - price_n_periods_ago) / price_n_periods_ago

        Args:
            prices: Array of prices
            window: Number of periods

        Returns:
            Momentum as a decimal or None if insufficient data
        """
        if len(prices) < window:
            return None

        current_price = prices[-1]
        past_price = prices[-window]

        if past_price == 0:
            return None

        momentum = (current_price - past_price) / past_price
        return round(float(momentum), 6)

    def _calculate_vwap(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> Optional[float]:
        """
        Calculate Volume Weighted Average Price.

        VWAP = sum(price * volume) / sum(volume)

        Args:
            prices: Array of prices
            volumes: Array of volumes

        Returns:
            VWAP value or None if no data
        """
        if len(prices) == 0 or len(volumes) == 0:
            return None

        total_volume = np.sum(volumes)

        if total_volume == 0:
            # If no volume, return simple average
            return float(np.mean(prices))

        vwap = np.sum(prices * volumes) / total_volume
        return round(float(vwap), 2)


# ============================================================================
# BATCH PROCESSING HELPER
# ============================================================================

def calculate_features_for_batch(
    price_data: List[Dict[str, Any]],
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Calculate features for a batch of price data.

    This is a convenience function for processing historical data.

    Args:
        price_data: List of price data dictionaries
        **kwargs: Additional arguments for PriceFeatureCalculator

    Returns:
        Dictionary of calculated features or None
    """
    if not price_data:
        return None

    calculator = PriceFeatureCalculator(**kwargs)
    calculator.add_prices(price_data)
    return calculator.get_features_dict()


# ============================================================================
# TEST/DEMO CODE
# ============================================================================

if __name__ == "__main__":
    import random
    import time

    print("\n" + "=" * 60)
    print("PRICE FEATURE CALCULATOR TEST")
    print("=" * 60 + "\n")

    # Generate sample data
    calculator = PriceFeatureCalculator(
        ma_short=5,  # Short windows for demo
        ma_long=10,
        volatility_window=20,
        momentum_window=5
    )

    base_price = 43000.0
    base_time = int(time.time() * 1000)

    print("Adding sample price data...")

    for i in range(30):
        # Simulate price movement
        change = random.uniform(-50, 50)
        price = base_price + change + (i * 10)  # Slight uptrend
        volume = random.uniform(0.01, 1.0)

        data = {
            "symbol": "BTCUSDT",
            "price": str(price),
            "volume": str(volume),
            "timestamp": base_time + (i * 1000)
        }

        calculator.add_price(data)

        if i % 5 == 0:
            print(f"  Added {i+1} prices, latest: ${price:.2f}")

    print(f"\nBuffer size: {calculator.buffer_length}")
    print("\nCalculated features:")

    features = calculator.get_features()
    if features:
        print(f"  Symbol: {features.symbol}")
        print(f"  Price: ${features.price:.2f}")
        print(f"  MA (5): {features.price_ma_5m:.2f}" if features.price_ma_5m else "  MA (5): N/A")
        print(f"  MA (10): {features.price_ma_15m:.2f}" if features.price_ma_15m else "  MA (10): N/A")
        print(f"  EMA (5): {features.price_ema_5m:.2f}" if features.price_ema_5m else "  EMA (5): N/A")
        print(f"  RSI (14): {features.rsi_14}" if features.rsi_14 else "  RSI (14): N/A")
        print(f"  Volatility: {features.volatility_1h}" if features.volatility_1h else "  Volatility: N/A")
        print(f"  Momentum: {features.momentum_5m}" if features.momentum_5m else "  Momentum: N/A")
        print(f"  VWAP: ${features.vwap:.2f}" if features.vwap else "  VWAP: N/A")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60 + "\n")
