"""
Ingestion Module

This module contains data ingestion components for the crypto ML pipeline.
"""

from src.ingestion.reddit_rss import RedditRSSIngestion, RedditPost, RSSParser

__all__ = [
    "RedditRSSIngestion",
    "RedditPost",
    "RSSParser",
]
