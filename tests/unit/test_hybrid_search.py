import pytest
import numpy as np
from app.hybrid_search import BM25Index, HybridSearcher
from app.vector_store import build_index

@pytest.fixture
def test_data():
    docs = [
        "apple banana orange",
        "apple computer laptop",
        "grape kiwi fruit"
    ]
    # Fake embeddings for the docs
    emb = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    bm25 = BM25Index()
    bm25.fit(docs)
    
    # Fake FAISS index
    index = build_index(emb, labels=[0, 1, 2])
    
    return docs, bm25, index

def test_hybrid_fusion_rrf(test_data):
    docs, bm25, index = test_data
    searcher = HybridSearcher(bm25, rrf_k=60)
    
    query = "apple"
    # Matches doc 0 in FAISS perfectly
    q_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    indices, scores, details = searcher.search(query, q_emb, index, docs, limit=2, offset=0)
    
    assert len(indices) == 2
    # Since doc 0 has exact match in FAISS and BM25 ("apple banana orange"), it should be rank 1
    assert indices[0] == 0
    assert details[0]["hybrid_score"] > 0
    assert "dense_rank" in details[0]
    assert "sparse_rank" in details[0]

def test_hybrid_pagination(test_data):
    docs, bm25, index = test_data
    searcher = HybridSearcher(bm25, rrf_k=60)
    
    query = "apple"
    q_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    # Get all 3 results
    indices_all, _, _ = searcher.search(query, q_emb, index, docs, limit=3, offset=0)
    
    # Get offset 1, limit 1
    indices_page, _, _ = searcher.search(query, q_emb, index, docs, limit=1, offset=1)
    
    assert len(indices_page) == 1
    assert indices_page[0] == indices_all[1]

def test_candidate_merging_from_both_retrievers(test_data):
    docs, bm25, index = test_data
    searcher = HybridSearcher(bm25, rrf_k=60)
    
    # Query matches doc 1 textually ("apple"), but doc 2 semantically (vector match)
    query = "apple"
    q_emb = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    
    indices, scores, details = searcher.search(query, q_emb, index, docs, limit=3, offset=0)
    
    assert len(indices) == 3
    assert set(indices) == {0, 1, 2}
