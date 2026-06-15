import pytest
from fastapi.testclient import TestClient
import numpy as np

# We need to mock the SentenceTransformer to prevent it from loading the model during tests
class MockSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
        
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        # Return random normalized embeddings
        embeddings = np.random.randn(len(texts), 384).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

# Inject the mock
import sentence_transformers  # noqa: E402
sentence_transformers.SentenceTransformer = MockSentenceTransformer

# Now import the app safely
from app.main import app  # noqa: E402

@pytest.fixture
def app_client():
    """Provides a FastAPI TestClient."""
    with TestClient(app) as client:
        yield client

@pytest.fixture
def sample_embeddings():
    """Returns a small batch of deterministic normalized embeddings."""
    np.random.seed(42)
    embeddings = np.random.randn(10, 384).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

@pytest.fixture
def sample_documents():
    """Returns a small set of mock research paper documents."""
    return [
        "Attention Is All You Need\n\nWe propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
        "BERT: Pre-training of Deep Bidirectional Transformers\n\nWe introduce a new language representation model called BERT.",
        "YOLO9000: Better, Faster, Stronger\n\nWe introduce YOLO9000, a state-of-the-art, real-time object detection system.",
        "Deep Residual Learning for Image Recognition\n\nDeeper neural networks are more difficult to train. We present a residual learning framework.",
        "Generative Adversarial Nets\n\nWe propose a new framework for estimating generative models via an adversarial process.",
        "Adam: A Method for Stochastic Optimization\n\nWe introduce Adam, an algorithm for first-order gradient-based optimization.",
        "XGBoost: A Scalable Tree Boosting System\n\nTree boosting is a highly effective and widely used machine learning method.",
        "Playing Atari with Deep Reinforcement Learning\n\nWe present the first deep learning model to successfully learn control policies.",
        "Mastering the game of Go with deep neural networks and tree search\n\nThe game of Go has long been viewed as the most challenging of classic games.",
        "Language Models are Few-Shot Learners\n\nWe demonstrate that scaling up language models greatly improves task-agnostic, few-shot performance."
    ]
