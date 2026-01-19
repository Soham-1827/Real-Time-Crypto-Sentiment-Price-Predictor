"""
Sentiment Analysis Module

This module provides sentiment analysis functionality using FinBERT,
a BERT model fine-tuned for financial sentiment analysis.

Why FinBERT instead of TextBlob?
- FinBERT understands financial context ("Bitcoin crashed" = negative)
- TextBlob is generic and might miss domain-specific sentiment
- Better accuracy for crypto/finance text

Architecture:
- By default, tries to use Modal.com for GPU-accelerated inference
- Falls back to local CPU inference if Modal is unavailable
- Use `use_modal=False` to force local inference

Usage:
    from src.processing.sentiment import analyze_sentiment, analyze_sentiment_batch

    # Single text
    result = analyze_sentiment("Bitcoin hits new ATH!")
    # Returns: {"score": 0.85, "confidence": 0.92, "label": "positive"}

    # Batch (more efficient)
    results = analyze_sentiment_batch(["text1", "text2"])
"""

import logging
import os
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# FinBERT model from HuggingFace
# ProsusAI/finbert is the most popular financial sentiment model
FINBERT_MODEL_NAME = "ProsusAI/finbert"

# Label mapping for FinBERT output
# FinBERT outputs: 0=positive, 1=negative, 2=neutral
LABEL_MAP = {
    0: "positive",
    1: "negative",
    2: "neutral"
}

# Score mapping: convert labels to numeric scores
# positive = +1, negative = -1, neutral = 0
SCORE_MAP = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0
}


# ============================================================================
# SENTIMENT RESULT DATACLASS
# ============================================================================

@dataclass
class SentimentResult:
    """
    Result of sentiment analysis.

    Attributes:
        score: Numeric score from -1 (negative) to +1 (positive)
        confidence: Model's confidence in the prediction (0-1)
        label: Human-readable label (positive/negative/neutral)
    """
    score: float
    confidence: float
    label: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "score": self.score,
            "confidence": self.confidence,
            "label": self.label
        }


# ============================================================================
# SENTIMENT ANALYZER CLASS
# ============================================================================

class SentimentAnalyzer:
    """
    Singleton class for sentiment analysis using FinBERT.

    This class:
    1. Lazy-loads the model on first use
    2. Caches the model to avoid reloading
    3. Handles GPU/CPU detection automatically

    Why singleton pattern?
    - Model loading is expensive (~500MB, takes several seconds)
    - We want to load once and reuse
    - Multiple calls should use the same model instance
    """

    _instance: Optional['SentimentAnalyzer'] = None
    _model = None
    _tokenizer = None
    _device = None

    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self) -> None:
        """
        Load FinBERT model and tokenizer.

        This is called lazily on first analysis request.
        The model is downloaded from HuggingFace on first run (~500MB).
        """
        if self._model is not None:
            return  # Already loaded

        logger.info(f"Loading FinBERT model: {FINBERT_MODEL_NAME}")
        logger.info("This may take a moment on first run (downloading ~500MB)...")

        # Detect device (GPU if available, else CPU)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self._device}")

        try:
            # Load tokenizer (converts text to tokens)
            self._tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)

            # Load model (the neural network)
            self._model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)

            # Move model to appropriate device
            self._model.to(self._device)

            # Set to evaluation mode (disables dropout, etc.)
            self._model.eval()

            logger.info("FinBERT model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of the given text.

        Args:
            text: Text to analyze (e.g., Reddit post title)

        Returns:
            SentimentResult with score, confidence, and label

        Example:
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("Bitcoin hits new ATH!")
            print(result.score)  # 0.85 (positive)
        """
        # Ensure model is loaded
        self._load_model()

        # Handle empty or invalid text
        if not text or not text.strip():
            return SentimentResult(score=0.0, confidence=0.0, label="neutral")

        try:
            # Tokenize the text
            # - return_tensors="pt" = return PyTorch tensors
            # - truncation=True = cut off text if too long
            # - max_length=512 = BERT's maximum sequence length
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            # Move inputs to same device as model
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Run inference (no gradient computation needed)
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Get probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

            # Get the predicted class and confidence
            confidence, predicted_class = torch.max(probabilities, dim=1)

            # Convert to Python types
            predicted_class = predicted_class.item()
            confidence = confidence.item()

            # Get label from class index
            label = LABEL_MAP.get(predicted_class, "neutral")

            # Calculate weighted score
            # We weight by probability of each class to get a continuous score
            probs = probabilities[0].cpu().numpy()
            # score = P(positive) * 1 + P(negative) * -1 + P(neutral) * 0
            score = probs[0] * 1.0 + probs[1] * -1.0 + probs[2] * 0.0

            return SentimentResult(
                score=round(float(score), 4),
                confidence=round(confidence, 4),
                label=label
            )

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            # Return neutral on error
            return SentimentResult(score=0.0, confidence=0.0, label="neutral")

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment for a batch of texts (more efficient than single).

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentResult objects

        Example:
            analyzer = SentimentAnalyzer()
            results = analyzer.analyze_batch(["Bitcoin up!", "Crash coming"])
        """
        # Ensure model is loaded
        self._load_model()

        if not texts:
            return []

        results = []
        batch_size = 32  # Process in mini-batches for memory efficiency

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Handle empty strings
            processed_texts = [t if t and t.strip() else "neutral" for t in batch_texts]

            try:
                # Tokenize batch
                inputs = self._tokenizer(
                    processed_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )

                # Move to device
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                # Run inference
                with torch.no_grad():
                    outputs = self._model(**inputs)

                # Get probabilities
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

                # Process each result
                for probs in probabilities:
                    probs_np = probs.cpu().numpy()

                    # Get predicted class and confidence
                    predicted_class = int(probs_np.argmax())
                    confidence = float(probs_np[predicted_class])

                    # Get label
                    label = LABEL_MAP.get(predicted_class, "neutral")

                    # Calculate weighted score
                    score = float(probs_np[0] * 1.0 + probs_np[1] * -1.0 + probs_np[2] * 0.0)

                    results.append(SentimentResult(
                        score=round(score, 4),
                        confidence=round(confidence, 4),
                        label=label
                    ))

            except Exception as e:
                logger.error(f"Error in batch analysis: {e}")
                # Return neutral for this batch
                for _ in batch_texts:
                    results.append(SentimentResult(score=0.0, confidence=0.0, label="neutral"))

        return results


# ============================================================================
# MODAL INTEGRATION
# ============================================================================

# Flag to track if Modal is available
_modal_available: Optional[bool] = None
_modal_model = None


def _check_modal_available() -> bool:
    """Check if Modal is available and configured."""
    global _modal_available

    if _modal_available is not None:
        return _modal_available

    # Check if Modal should be disabled via environment variable
    if os.getenv("DISABLE_MODAL", "").lower() in ("1", "true", "yes"):
        logger.info("Modal disabled via DISABLE_MODAL environment variable")
        _modal_available = False
        return False

    try:
        import modal
        # Check if modal package is installed
        _modal_available = True
        logger.info("Modal package is available")
        return True
    except ImportError:
        logger.info("Modal not installed, using local inference")
        _modal_available = False
        return False


def _get_modal_model():
    """Get the Modal FinBERT model instance."""
    global _modal_model

    if _modal_model is not None:
        return _modal_model

    try:
        import modal
        # Use the new Modal API: Cls.from_name instead of Cls.lookup
        model_cls = modal.Cls.from_name("finbert-sentiment", "FinBERTModel")
        _modal_model = model_cls()
        logger.info("Modal model instance created")
        return _modal_model
    except Exception as e:
        logger.warning(f"Failed to get Modal model: {e}")
        return None


def analyze_sentiment_batch_modal(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Analyze sentiment using Modal (GPU-accelerated).

    Args:
        texts: List of texts to analyze

    Returns:
        List of dicts with score, confidence, label

    Raises:
        Exception if Modal is not available
    """
    model = _get_modal_model()
    if model is None:
        raise RuntimeError("Modal model not available")

    return model.analyze_batch.remote(texts)


# ============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTION
# ============================================================================

# Global analyzer instance (lazy initialization)
_analyzer: Optional[SentimentAnalyzer] = None


def get_analyzer() -> SentimentAnalyzer:
    """
    Get the global SentimentAnalyzer instance.

    Returns:
        SentimentAnalyzer instance (creates one if needed)
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer


def analyze_sentiment(text: str, use_modal: bool = True) -> Dict[str, Any]:
    """
    Convenience function to analyze sentiment.

    This is the main function other modules should use.
    By default, tries to use Modal (GPU) and falls back to local (CPU).

    Args:
        text: Text to analyze
        use_modal: If True, try Modal first; if False, use local only

    Returns:
        Dictionary with keys: score, confidence, label

    Example:
        from src.processing.sentiment import analyze_sentiment

        result = analyze_sentiment("Ethereum upgrade successful!")
        print(result)
        # {"score": 0.72, "confidence": 0.89, "label": "positive"}
    """
    # Try Modal first if requested
    if use_modal and _check_modal_available():
        try:
            results = analyze_sentiment_batch_modal([text])
            if results:
                return results[0]
        except Exception as e:
            logger.warning(f"Modal inference failed, falling back to local: {e}")

    # Fall back to local
    analyzer = get_analyzer()
    result = analyzer.analyze(text)
    return result.to_dict()


def analyze_sentiment_batch(texts: List[str], use_modal: bool = True) -> List[Dict[str, Any]]:
    """
    Analyze sentiment for a batch of texts (more efficient).

    By default, tries to use Modal (GPU) and falls back to local (CPU).
    Use this function when processing multiple texts for better efficiency.

    Args:
        texts: List of texts to analyze
        use_modal: If True, try Modal first; if False, use local only

    Returns:
        List of dictionaries with score, confidence, label

    Example:
        from src.processing.sentiment import analyze_sentiment_batch

        results = analyze_sentiment_batch([
            "Bitcoin hits new ATH!",
            "Market crashes hard"
        ])
        # [{"score": 0.85, ...}, {"score": -0.72, ...}]
    """
    if not texts:
        return []

    # Try Modal first if requested
    if use_modal and _check_modal_available():
        try:
            return analyze_sentiment_batch_modal(texts)
        except Exception as e:
            logger.warning(f"Modal batch inference failed, falling back to local: {e}")

    # Fall back to local
    analyzer = get_analyzer()
    results = analyzer.analyze_batch(texts)
    return [r.to_dict() for r in results]


# ============================================================================
# TEST/DEMO CODE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test FinBERT sentiment analysis")
    parser.add_argument("--local", action="store_true", help="Force local inference (no Modal)")
    args = parser.parse_args()

    use_modal = not args.local

    print("\n" + "=" * 60)
    print("FINBERT SENTIMENT ANALYSIS TEST")
    print(f"Mode: {'Local (CPU)' if args.local else 'Modal (GPU) with local fallback'}")
    print("=" * 60 + "\n")

    # Test texts with expected sentiments
    test_texts = [
        ("Bitcoin hits new all-time high!", "positive"),
        ("Crypto market crashes 20% overnight", "negative"),
        ("Trading volume remains stable", "neutral"),
        ("Ethereum upgrade successful, network fees drop", "positive"),
        ("SEC announces investigation into major exchange", "negative"),
        ("BTC price moving sideways", "neutral"),
    ]

    print("Loading model (first run may download ~500MB)...\n")

    # Test single analysis
    print("--- Single Analysis ---\n")
    for text, expected in test_texts[:2]:
        result = analyze_sentiment(text, use_modal=use_modal)
        status = "OK" if result["label"] == expected else "MISMATCH"

        print(f"Text: \"{text}\"")
        print(f"  Score: {result['score']:+.3f}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Label: {result['label']} (expected: {expected}) [{status}]")
        print()

    # Test batch analysis
    print("--- Batch Analysis ---\n")
    texts_only = [t[0] for t in test_texts]
    expected_labels = [t[1] for t in test_texts]

    results = analyze_sentiment_batch(texts_only, use_modal=use_modal)

    for text, result, expected in zip(texts_only, results, expected_labels):
        status = "OK" if result["label"] == expected else "MISMATCH"

        print(f"Text: \"{text}\"")
        print(f"  Score: {result['score']:+.3f}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Label: {result['label']} (expected: {expected}) [{status}]")
        print()

    print("=" * 60)
    print("Test complete!")
    print("=" * 60 + "\n")
