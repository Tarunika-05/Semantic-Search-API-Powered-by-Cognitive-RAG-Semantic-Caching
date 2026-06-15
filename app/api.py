from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import numpy as np

# Create the router
router = APIRouter()

# ─────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────

from app.rate_limit import limiter  # noqa: E402
from app.logger import get_logger  # noqa: E402
import time  # noqa: E402
from app.analytics import log_query, get_analytics_stats  # noqa: E402

logger = get_logger("api")

class QueryRequest(BaseModel):
    query: str
    generate: bool = True  # Whether to generate an LLM answer
    limit: int = 5
    offset: int = 0


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: str | None
    similarity_score: float | None
    result: str
    generated_answer: str | None = None
    citations: list[dict] = []
    dominant_cluster: int
    search_mode: str = "dense"  # "dense" | "hybrid"
    limit: int
    offset: int


class HybridQueryRequest(BaseModel):
    query: str
    generate: bool = True  # Whether to generate an LLM answer
    limit: int = 5
    offset: int = 0


# ─────────────────────────────────────────────────────────────────────
# Handlers (Functions previously inside main.py)
# ─────────────────────────────────────────────────────────────────────

def process_query(req: Request, query: str):
    """
    Embed a query string and determine its dominant cluster.
    """
    state = req.app.state
    model = state.model
    pca = state.pca
    gmm = state.gmm

    # Embed (1, 384)
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # Reduce for clustering (1, 50)
    query_reduced = pca.transform([query_embedding])

    # Soft cluster assignment (15,)
    cluster_probs = gmm.predict_proba(query_reduced)[0]
    dominant_cluster = int(np.argmax(cluster_probs))

    return query_embedding, dominant_cluster, cluster_probs


def get_result_from_corpus(request: Request, query_embedding: np.ndarray, category_filter: int | None = None, limit: int = 5, offset: int = 0) -> tuple[str, list[str], list[int], list[float]]:
    """
    Search FAISS index and return the most relevant document snippet, top-k docs, their indices, and scores.
    """
    from app.vector_store import search_index, search_with_filter
    
    state = request.app.state
    index_data = state.index
    documents = state.documents

    if category_filter is not None:
        distances, indices = search_with_filter(
            index_data, query_embedding, category_filter=category_filter, limit=limit, offset=offset
        )
    else:
        distances, indices = search_index(index_data, query_embedding, limit=limit, offset=offset)

    if len(indices) == 0:
        return "No matching documents found.", [], [], []

    top_docs = [documents[idx] for idx in indices]
    # Build result from top match snippet
    result = top_docs[0].strip()
    return result, top_docs, list(indices), list(distances)


def get_hybrid_result(request: Request, query: str, query_embedding: np.ndarray, limit: int = 5, offset: int = 0) -> tuple:
    """
    Search using hybrid (BM25 + Dense) scoring via RRF.
    """
    state = request.app.state
    hybrid = state.hybrid_searcher
    documents = state.documents
    index_data = state.index

    indices, scores, details = hybrid.search(
        query=query,
        query_embedding=query_embedding,
        faiss_index=index_data,
        documents=documents,
        limit=limit,
        offset=offset
    )

    if len(indices) == 0:
        return "No matching documents found.", [], [], [], details

    top_docs = [documents[idx] for idx in indices]
    result = top_docs[0].strip()
    return result, top_docs, list(indices), list(scores), details


# ─────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────
from app.auth import require_role, Role

@router.post("/query", response_model=QueryResponse, dependencies=[Depends(require_role(Role.USER))])
@limiter.limit("60/minute")
async def query_endpoint(request: Request, payload: QueryRequest):
    start_time = time.time()
    logger.info("Processing /query", query=payload.query)
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query = payload.query.strip()
    cache = request.app.state.cache

    query_embedding, dominant_cluster, cluster_probs = process_query(request, query)

    cached_entry = cache.lookup(query_embedding, dominant_cluster, cluster_probs)

    if cached_entry is not None:
        similarity = float(np.dot(query_embedding, cached_entry.embedding))
        
        latency_ms = (time.time() - start_time) * 1000
        log_query(query, "dense", True, latency_ms, dominant_cluster)
        
        return QueryResponse(
            query=query,
            cache_hit=True,
            matched_query=cached_entry.query,
            similarity_score=round(similarity, 4),
            result=cached_entry.result,
            generated_answer=getattr(cached_entry, "generated_answer", None),
            citations=getattr(cached_entry, "citations", []),
            dominant_cluster=dominant_cluster,
            search_mode="dense",
            limit=payload.limit,
            offset=payload.offset
        )

    result, top_docs, indices, scores = get_result_from_corpus(request, query_embedding, limit=payload.limit, offset=payload.offset)

    generated_answer = None
    citations = []
    if payload.generate and top_docs:
        from app.llm import generate_answer, AzureOpenAIProvider
        provider = AzureOpenAIProvider()
        generated_answer = generate_answer(query, top_docs, provider=provider)
        # Build strong citations
        for doc, idx, score in zip(top_docs, indices, scores):
            parts = doc.split('\n\n', 1)
            citations.append({
                "title": parts[0].strip(),
                "snippet": parts[1][:150].strip() + "..." if len(parts) > 1 else "",
                "paper_id": int(idx),
                "retrieval_score": round(float(score), 4)
            })

    cache.store(
        query=query,
        embedding=query_embedding,
        result=result,
        dominant_cluster=dominant_cluster,
        cluster_probs=cluster_probs,
        generated_answer=generated_answer,
        citations=citations
    )

    latency_ms = (time.time() - start_time) * 1000
    log_query(query, "dense", False, latency_ms, dominant_cluster)

    return QueryResponse(
        query=query,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result,
        generated_answer=generated_answer,
        citations=citations,
        dominant_cluster=dominant_cluster,
        search_mode="dense",
        limit=payload.limit,
        offset=payload.offset
    )


@router.post("/hybrid-query", dependencies=[Depends(require_role(Role.USER))])
@limiter.limit("60/minute")
async def hybrid_query_endpoint(request: Request, payload: HybridQueryRequest):
    start_time = time.time()
    logger.info("Processing /hybrid-query", query=payload.query)
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query = payload.query.strip()
    query_embedding, dominant_cluster, _ = process_query(request, query)

    result, top_docs, indices, scores, score_details = get_hybrid_result(
        request, query, query_embedding, limit=payload.limit, offset=payload.offset
    )

    generated_answer = None
    citations = []
    if payload.generate and top_docs:
        from app.llm import generate_answer, AzureOpenAIProvider
        provider = AzureOpenAIProvider()
        generated_answer = generate_answer(query, top_docs, provider=provider)
        for doc, idx, score in zip(top_docs, indices, scores):
            parts = doc.split('\n\n', 1)
            citations.append({
                "title": parts[0].strip(),
                "snippet": parts[1][:150].strip() + "..." if len(parts) > 1 else "",
                "paper_id": int(idx),
                "retrieval_score": round(float(score), 4)
            })

    latency_ms = (time.time() - start_time) * 1000
    log_query(query, "hybrid", False, latency_ms, dominant_cluster)

    return {
        "query": query,
        "cache_hit": False,
        "result": result,
        "generated_answer": generated_answer,
        "citations": citations,
        "search_mode": "hybrid",
        "dominant_cluster": dominant_cluster,
        "score_breakdown": score_details,
        "limit": payload.limit,
        "offset": payload.offset
    }


class FilteredQueryRequest(BaseModel):
    query: str
    category: int = 0
    generate: bool = True
    limit: int = 5
    offset: int = 0


@router.post("/filtered-query", dependencies=[Depends(require_role(Role.USER))])
@limiter.limit("60/minute")
async def filtered_query_endpoint(request: Request, payload: FilteredQueryRequest):
    start_time = time.time()
    logger.info("Processing /filtered-query", query=payload.query, category=payload.category)
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query = payload.query.strip()
    query_embedding, dominant_cluster, _ = process_query(request, query)
    result, top_docs, indices, scores = get_result_from_corpus(
        request, query_embedding, category_filter=payload.category, limit=payload.limit, offset=payload.offset
    )

    generated_answer = None
    citations = []
    if payload.generate and top_docs:
        from app.llm import generate_answer, AzureOpenAIProvider
        provider = AzureOpenAIProvider()
        generated_answer = generate_answer(query, top_docs, provider=provider)
        for doc, idx, score in zip(top_docs, indices, scores):
            parts = doc.split('\n\n', 1)
            citations.append({
                "title": parts[0].strip(),
                "snippet": parts[1][:150].strip() + "..." if len(parts) > 1 else "",
                "paper_id": int(idx),
                "retrieval_score": round(float(score), 4)
            })

    latency_ms = (time.time() - start_time) * 1000
    log_query(query, f"filtered_{payload.category}", False, latency_ms, dominant_cluster)

    return {
        "query": query,
        "cache_hit": False,
        "result": result,
        "generated_answer": generated_answer,
        "citations": citations,
        "search_mode": "filtered",
        "category_filter": payload.category,
        "dominant_cluster": dominant_cluster,
        "limit": payload.limit,
        "offset": payload.offset
    }


@router.get("/cache/stats", dependencies=[Depends(require_role(Role.ADMIN))])
@limiter.limit("60/minute")
async def cache_stats(request: Request):
    logger.info("Fetching cache stats")
    return request.app.state.cache.get_stats()


@router.delete("/cache", dependencies=[Depends(require_role(Role.ADMIN))])
async def clear_cache(request: Request):
    logger.warning("Clearing semantic cache")
    request.app.state.cache.flush()
    return {"message": "Cache cleared successfully.", "status": "ok"}


@router.patch("/cache/threshold", dependencies=[Depends(require_role(Role.ADMIN))])
async def update_threshold(request: Request, threshold: float):
    logger.warning("Updating cache threshold", new_threshold=threshold)
    if not (0.0 < threshold <= 1.0):
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1.")
    request.app.state.cache.set_threshold(threshold)
    return {"message": f"Cache threshold updated to {threshold}", "status": "ok"}


@router.get("/clusters/analysis", dependencies=[Depends(require_role(Role.ADMIN))])
@limiter.limit("10/minute")
def cluster_analysis(request: Request):
    logger.info("Running cluster analysis")
    from app.clustering import get_full_analysis
    state = request.app.state
    return get_full_analysis(
        state.documents,
        state.cluster_probs,
        state.dominant_clusters,
        state.embeddings
    )


@router.get("/analytics", dependencies=[Depends(require_role(Role.ADMIN))])
async def get_analytics(request: Request):
    logger.info("Fetching analytics stats")
    return await get_analytics_stats()


@router.get("/evaluate", dependencies=[Depends(require_role(Role.ADMIN))])
async def evaluate_ir_metrics(request: Request):
    logger.info("Fetching IR evaluation metrics")
    import json
    import os
    results_path = "experiments/results/evaluation_results.json"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    else:
        return {"error": "Evaluation results not found. Please run experiments/evaluate_search.py first."}


@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@router.get("/health")
async def health(request: Request):
    state = request.app.state
    return {
        "status": "ok",
        "documents_loaded": len(getattr(state, "documents", [])),
        "cache_entries": len(getattr(state, "cache", [])),
        "bm25_vocab_size": len(getattr(state, "bm25", __import__("app.hybrid_search", fromlist=["BM25Index"]).BM25Index()).df),
        "search_modes": ["dense", "hybrid", "filtered"],
    }
