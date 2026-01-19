"""
Modal-hosted FinBERT Sentiment Analysis

This module deploys FinBERT on Modal.com for serverless GPU inference.
Modal provides $30/month free credits, perfect for this project.

Why Modal instead of local inference?
- GPU inference is ~10x faster than CPU
- Serverless = no server management
- Pay only for compute time (~$0.01/hour at our volume)
- Cold starts ~2s (acceptable for batch processing)

Deployment:
    modal deploy src/processing/sentiment_modal.py

Usage:
    from src.processing.sentiment_modal import analyze_sentiment_batch

    results = analyze_sentiment_batch.remote(["Bitcoin up!", "Crash incoming"])

Note:
    You must run `modal token new` once to authenticate before deploying.
"""

import modal

# ============================================================================
# MODAL APP CONFIGURATION
# ============================================================================

# Create the Modal app
app = modal.App("finbert-sentiment")

# Define the container image with all dependencies
# This gets cached after first build, so subsequent starts are fast
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.1.2",
    "transformers==4.36.2",
    "numpy==1.26.2",
)

# Model configuration
FINBERT_MODEL_NAME = "ProsusAI/finbert"

# Label mapping for FinBERT output
# FinBERT outputs: 0=positive, 1=negative, 2=neutral
LABEL_MAP = {
    0: "positive",
    1: "negative",
    2: "neutral"
}


# ============================================================================
# MODAL CLASS FOR FINBERT
# ============================================================================

@app.cls(
    image=image,
    gpu="T4",  # NVIDIA T4 - cheapest GPU option, plenty fast for inference
    container_idle_timeout=300,  # Keep warm for 5 minutes (reduces cold starts)
)
class FinBERTModel:
    """
    Modal class that hosts the FinBERT model.

    The @modal.enter() decorator ensures the model loads once when the
    container starts, not on every request.
    """

    @modal.enter()
    def load_model(self):
        """
        Load FinBERT model when container starts.

        This runs once on container startup and the model stays loaded
        for subsequent requests (until container idles out).
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        print(f"Loading FinBERT model: {FINBERT_MODEL_NAME}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)

        # Move to GPU and set to eval mode
        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.model.eval()

        print("FinBERT model loaded successfully on GPU!")

    @modal.method()
    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """
        Analyze sentiment for a batch of texts.

        Batch processing is more efficient than single texts because:
        - Reduces API call overhead
        - Better GPU utilization
        - Amortizes cold start cost

        Args:
            texts: List of texts to analyze

        Returns:
            List of dicts with score, confidence, label for each text
        """
        import torch
        import numpy as np

        if not texts:
            return []

        results = []

        # Process in mini-batches of 32 for memory efficiency
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Handle empty strings
            processed_texts = [t if t and t.strip() else "neutral" for t in batch_texts]

            # Tokenize batch
            inputs = self.tokenizer(
                processed_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            # Move to GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

            # Process each result in batch
            for j, probs in enumerate(probabilities):
                probs_np = probs.cpu().numpy()

                # Get predicted class and confidence
                predicted_class = int(np.argmax(probs_np))
                confidence = float(probs_np[predicted_class])

                # Get label
                label = LABEL_MAP.get(predicted_class, "neutral")

                # Calculate weighted score
                # score = P(positive) * 1 + P(negative) * -1 + P(neutral) * 0
                score = float(probs_np[0] * 1.0 + probs_np[1] * -1.0 + probs_np[2] * 0.0)

                results.append({
                    "score": round(score, 4),
                    "confidence": round(confidence, 4),
                    "label": label
                })

        return results

    @modal.method()
    def analyze_single(self, text: str) -> dict:
        """
        Analyze sentiment for a single text.

        For efficiency, prefer analyze_batch() when processing multiple texts.

        Args:
            text: Text to analyze

        Returns:
            Dict with score, confidence, label
        """
        results = self.analyze_batch([text])
        return results[0] if results else {"score": 0.0, "confidence": 0.0, "label": "neutral"}


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

@app.function(image=image)
def analyze_sentiment_batch(texts: list[str]) -> list[dict]:
    """
    Convenience function to analyze a batch of texts.

    This creates a FinBERTModel instance and calls analyze_batch.

    Usage:
        from src.processing.sentiment_modal import analyze_sentiment_batch
        results = analyze_sentiment_batch.remote(["text1", "text2"])
    """
    model = FinBERTModel()
    return model.analyze_batch.remote(texts)


@app.function(image=image)
def analyze_sentiment_single(text: str) -> dict:
    """
    Convenience function to analyze a single text.

    Usage:
        from src.processing.sentiment_modal import analyze_sentiment_single
        result = analyze_sentiment_single.remote("Bitcoin is awesome!")
    """
    model = FinBERTModel()
    return model.analyze_single.remote(text)


# ============================================================================
# LOCAL ENTRY POINT FOR TESTING
# ============================================================================

@app.local_entrypoint()
def main():
    """
    Test the sentiment analysis locally.

    Run with: modal run src/processing/sentiment_modal.py
    """
    test_texts = [
        "Bitcoin hits new all-time high!",
        "Crypto market crashes 20% overnight",
        "Trading volume remains stable",
        "Ethereum upgrade successful, network fees drop",
        "SEC announces investigation into major exchange",
        "BTC price moving sideways",
    ]

    print("\n" + "=" * 60)
    print("MODAL FINBERT SENTIMENT ANALYSIS TEST")
    print("=" * 60 + "\n")

    # Test batch processing
    print("Testing batch processing...")
    model = FinBERTModel()
    results = model.analyze_batch.remote(test_texts)

    for text, result in zip(test_texts, results):
        print(f"\nText: \"{text}\"")
        print(f"  Score: {result['score']:+.4f}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Label: {result['label']}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60 + "\n")
