"""
Reddit RSS Ingestion Module

This module polls Reddit RSS feeds for crypto-related subreddits,
analyzes sentiment using FinBERT (via Modal GPU or local CPU),
and pushes the data to Redis.

Why RSS instead of Reddit API?
- No API key required
- No rate limiting concerns
- Simpler implementation
- Sufficient for our needs (new posts, not real-time)

Usage:
    # Run as main script
    python -m src.ingestion.reddit_rss

    # Or import and run
    from src.ingestion.reddit_rss import RedditRSSIngestion
    ingestion = RedditRSSIngestion()
    await ingestion.run()
"""

import asyncio
import hashlib
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any

import feedparser
from dateutil import parser as date_parser

from src.storage.redis_client import get_redis_client
from src.processing.sentiment import analyze_sentiment_batch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Subreddits to monitor (these have active crypto discussions)
SUBREDDITS = [
    "cryptocurrency",      # Main crypto subreddit
    "bitcoin",             # Bitcoin-specific
    "ethereum",            # Ethereum-specific
    "CryptoMarkets",       # Trading/markets focus
]

# RSS feed URL template
# Reddit RSS format: https://www.reddit.com/r/{subreddit}/new/.rss
RSS_URL_TEMPLATE = "https://www.reddit.com/r/{subreddit}/new/.rss"

# Polling configuration
POLL_INTERVAL_SECONDS = 300  # 5 minutes (respect Reddit's servers)
MAX_POSTS_PER_FEED = 25      # RSS typically returns 25 posts

# Redis configuration
REDIS_LIST_NAME = "raw:reddit"

# Deduplication: how long to remember seen posts (in seconds)
SEEN_POSTS_TTL = 86400  # 24 hours


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RedditPost:
    """Represents a Reddit post from RSS feed."""
    post_id: str
    subreddit: str
    title: str
    author: str
    url: str
    published_timestamp: int  # Unix timestamp in milliseconds
    fetched_at: int           # When we fetched it (Unix ms)

    # Sentiment fields (filled after analysis)
    sentiment_score: Optional[float] = None
    sentiment_confidence: Optional[float] = None
    sentiment_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "post_id": self.post_id,
            "subreddit": self.subreddit,
            "title": self.title,
            "author": self.author,
            "url": self.url,
            "timestamp": self.published_timestamp,
            "sentiment_score": self.sentiment_score,
            "sentiment_confidence": self.sentiment_confidence,
            "sentiment_label": self.sentiment_label,
            "fetched_at": self.fetched_at,
        }


# ============================================================================
# RSS PARSER
# ============================================================================

class RSSParser:
    """Parses Reddit RSS feeds into RedditPost objects."""

    def __init__(self, user_agent: str = "CryptoSentimentBot/1.0"):
        """
        Initialize RSS parser.

        Args:
            user_agent: User-Agent header for requests (be respectful)
        """
        self.user_agent = user_agent

    def parse_feed(self, subreddit: str) -> List[RedditPost]:
        """
        Fetch and parse RSS feed for a subreddit.

        Args:
            subreddit: Name of the subreddit (without r/)

        Returns:
            List of RedditPost objects
        """
        url = RSS_URL_TEMPLATE.format(subreddit=subreddit)
        logger.debug(f"Fetching RSS feed: {url}")

        try:
            # feedparser handles the HTTP request and XML parsing
            # We set the user-agent to be respectful
            feed = feedparser.parse(
                url,
                request_headers={"User-Agent": self.user_agent}
            )

            if feed.bozo:
                # bozo flag indicates parsing errors
                logger.warning(f"Feed parsing warning for {subreddit}: {feed.bozo_exception}")

            posts = []
            fetched_at = int(time.time() * 1000)

            for entry in feed.entries[:MAX_POSTS_PER_FEED]:
                try:
                    post = self._parse_entry(entry, subreddit, fetched_at)
                    if post:
                        posts.append(post)
                except Exception as e:
                    logger.warning(f"Failed to parse entry: {e}")
                    continue

            logger.info(f"Parsed {len(posts)} posts from r/{subreddit}")
            return posts

        except Exception as e:
            logger.error(f"Failed to fetch feed for r/{subreddit}: {e}")
            return []

    def _parse_entry(
        self,
        entry: Dict,
        subreddit: str,
        fetched_at: int
    ) -> Optional[RedditPost]:
        """
        Parse a single RSS entry into a RedditPost.

        Args:
            entry: RSS entry from feedparser
            subreddit: Subreddit name
            fetched_at: Timestamp when we fetched this

        Returns:
            RedditPost or None if parsing fails
        """
        # Extract post ID from the entry ID
        # Reddit entry IDs look like: "t3_abc123"
        entry_id = entry.get("id", "")
        post_id = entry_id.split("_")[-1] if "_" in entry_id else entry_id

        # If no valid ID, generate one from title hash
        if not post_id:
            post_id = hashlib.md5(entry.get("title", "").encode()).hexdigest()[:12]

        # Parse published time
        published = entry.get("published", "")
        try:
            if published:
                dt = date_parser.parse(published)
                published_ts = int(dt.timestamp() * 1000)
            else:
                published_ts = fetched_at
        except Exception:
            published_ts = fetched_at

        # Extract author (Reddit RSS includes "u/username")
        author = entry.get("author", "unknown")
        if author.startswith("/u/"):
            author = author[3:]
        elif author.startswith("u/"):
            author = author[2:]

        return RedditPost(
            post_id=post_id,
            subreddit=subreddit,
            title=entry.get("title", "").strip(),
            author=author,
            url=entry.get("link", ""),
            published_timestamp=published_ts,
            fetched_at=fetched_at,
        )


# ============================================================================
# MAIN INGESTION CLASS
# ============================================================================

class RedditRSSIngestion:
    """
    Main class for Reddit RSS ingestion with sentiment analysis.

    This class:
    1. Polls RSS feeds from configured subreddits
    2. Deduplicates posts (tracks what we've seen)
    3. Batches posts for efficient sentiment analysis
    4. Pushes enriched posts to Redis
    """

    def __init__(
        self,
        subreddits: List[str] = None,
        poll_interval: int = POLL_INTERVAL_SECONDS,
        use_modal: bool = True,
    ):
        """
        Initialize Reddit RSS ingestion.

        Args:
            subreddits: List of subreddits to monitor (default: SUBREDDITS)
            poll_interval: Seconds between polls (default: 300)
            use_modal: Whether to use Modal for sentiment (default: True)
        """
        self.subreddits = subreddits or SUBREDDITS
        self.poll_interval = poll_interval
        self.use_modal = use_modal

        # Initialize components
        self.parser = RSSParser()
        self.redis = None  # Lazy initialization
        self.seen_posts: Set[str] = set()  # Track seen post IDs

        # Shutdown handling
        self.running = False
        self._setup_signal_handlers()

        logger.info(f"Initialized Reddit RSS ingestion for: {self.subreddits}")
        logger.info(f"Poll interval: {self.poll_interval}s, Modal: {self.use_modal}")

    def _setup_signal_handlers(self):
        """Set up handlers for graceful shutdown."""
        # Signal handlers will be set up in the async run method
        pass

    def _request_shutdown(self):
        """Request graceful shutdown."""
        logger.info("Shutdown requested...")
        self.running = False

    def _get_redis(self):
        """Lazy initialization of Redis client."""
        if self.redis is None:
            self.redis = get_redis_client()
        return self.redis

    def _generate_post_key(self, post: RedditPost) -> str:
        """Generate unique key for deduplication."""
        return f"{post.subreddit}:{post.post_id}"

    def _is_seen(self, post: RedditPost) -> bool:
        """Check if we've already processed this post."""
        key = self._generate_post_key(post)
        return key in self.seen_posts

    def _mark_seen(self, post: RedditPost):
        """Mark a post as seen."""
        key = self._generate_post_key(post)
        self.seen_posts.add(key)

    def _cleanup_seen_posts(self):
        """
        Periodically clean up seen posts to prevent memory bloat.

        For simplicity, we just clear the set periodically.
        In production, you'd use a time-based cache or Redis SET.
        """
        if len(self.seen_posts) > 10000:
            logger.info("Cleaning up seen posts cache")
            # Keep most recent 5000 (arbitrary, but prevents unbounded growth)
            self.seen_posts = set(list(self.seen_posts)[-5000:])

    async def poll_once(self) -> int:
        """
        Perform a single poll cycle.

        Returns:
            Number of new posts processed
        """
        all_new_posts: List[RedditPost] = []

        # Fetch from all subreddits
        for subreddit in self.subreddits:
            posts = self.parser.parse_feed(subreddit)

            # Filter to only new posts
            new_posts = [p for p in posts if not self._is_seen(p)]
            all_new_posts.extend(new_posts)

            logger.debug(f"r/{subreddit}: {len(posts)} total, {len(new_posts)} new")

        if not all_new_posts:
            logger.info("No new posts found")
            return 0

        logger.info(f"Processing {len(all_new_posts)} new posts")

        # Batch sentiment analysis for efficiency
        titles = [p.title for p in all_new_posts]
        try:
            sentiment_results = analyze_sentiment_batch(titles, use_modal=self.use_modal)

            # Attach sentiment to posts
            for post, sentiment in zip(all_new_posts, sentiment_results):
                post.sentiment_score = sentiment["score"]
                post.sentiment_confidence = sentiment["confidence"]
                post.sentiment_label = sentiment["label"]

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            # Continue with neutral sentiment
            for post in all_new_posts:
                post.sentiment_score = 0.0
                post.sentiment_confidence = 0.0
                post.sentiment_label = "neutral"

        # Push to Redis
        redis = self._get_redis()
        success_count = 0

        for post in all_new_posts:
            try:
                if redis.push_to_list(REDIS_LIST_NAME, post.to_dict()):
                    self._mark_seen(post)
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to push post to Redis: {e}")

        logger.info(f"Pushed {success_count}/{len(all_new_posts)} posts to Redis")

        # Cleanup seen posts cache periodically
        self._cleanup_seen_posts()

        return success_count

    async def _interruptible_sleep(self, seconds: int):
        """
        Sleep that can be interrupted by shutdown request.

        Checks the running flag every second so we can exit quickly.
        """
        try:
            for _ in range(seconds):
                if not self.running:
                    break
                await asyncio.sleep(1)
        except (asyncio.CancelledError, KeyboardInterrupt):
            self.running = False

    async def run(self):
        """
        Run the ingestion loop continuously.

        This will poll RSS feeds every POLL_INTERVAL_SECONDS until
        interrupted by SIGINT or SIGTERM.
        """
        self.running = True
        logger.info("Starting Reddit RSS ingestion loop...")
        logger.info("Press Ctrl+C to stop...")

        # Set up signal handlers
        # Note: add_signal_handler doesn't work on Windows, use signal.signal instead
        import sys
        if sys.platform != "win32":
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._request_shutdown)
        else:
            # Windows fallback - works but may show KeyboardInterrupt
            signal.signal(signal.SIGINT, lambda s, f: self._request_shutdown())
            signal.signal(signal.SIGTERM, lambda s, f: self._request_shutdown())

        poll_count = 0
        total_posts = 0

        while self.running:
            try:
                poll_count += 1
                logger.info(f"--- Poll #{poll_count} ---")

                new_posts = await self.poll_once()
                total_posts += new_posts

                logger.info(
                    f"Poll complete. New posts: {new_posts}, "
                    f"Total processed: {total_posts}"
                )

                if self.running:
                    logger.info(f"Sleeping for {self.poll_interval} seconds... (Ctrl+C to stop)")
                    await self._interruptible_sleep(self.poll_interval)

            except (asyncio.CancelledError, KeyboardInterrupt):
                logger.info("Ingestion loop cancelled")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in poll cycle: {e}")
                if self.running:
                    # Wait before retrying on error
                    await self._interruptible_sleep(30)

        logger.info(f"Ingestion stopped. Total polls: {poll_count}, Total posts: {total_posts}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point for Reddit RSS ingestion."""
    import argparse

    parser = argparse.ArgumentParser(description="Reddit RSS Ingestion for Crypto Sentiment")
    parser.add_argument(
        "--interval",
        type=int,
        default=POLL_INTERVAL_SECONDS,
        help=f"Poll interval in seconds (default: {POLL_INTERVAL_SECONDS})"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force local sentiment analysis (no Modal)"
    )
    parser.add_argument(
        "--subreddits",
        nargs="+",
        default=None,
        help=f"Subreddits to monitor (default: {SUBREDDITS})"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run only one poll cycle and exit"
    )

    args = parser.parse_args()

    ingestion = RedditRSSIngestion(
        subreddits=args.subreddits,
        poll_interval=args.interval,
        use_modal=not args.local,
    )

    if args.once:
        # Single poll mode (useful for testing)
        await ingestion.poll_once()
    else:
        # Continuous mode
        await ingestion.run()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("REDDIT RSS INGESTION")
    print("=" * 60 + "\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
