from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import time
import uuid
import structlog
from dotenv import load_dotenv

load_dotenv()

from app.api import router
from app.dataset import load_documents
from app.embeddings import get_model, embed_documents, save_embeddings, load_embeddings
from app.vector_store import build_index, save_index, load_index
from app.clustering import (
    reduce_dimensions, fit_gmm, get_cluster_distributions,
    get_dominant_cluster, save_clustering, load_clustering,
    analyze_clusters
)
from app.cache import SemanticCache
from app.hybrid_search import BM25Index, HybridSearcher, save_bm25, load_bm25
from app.logger import setup_logging, get_logger
from app.rate_limit import setup_rate_limiting
from app.analytics import init_db

# Setup structured logging
setup_logging()
logger = get_logger("main")

# ─────────────────────────────────────────────────────────────────────
# App Lifespan (Startup / Shutdown context)
# ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application...")
    
    # Initialize analytics database
    await init_db()

    # Store objects natively on app.state
    
    # ── Step 1: Embedding model ──
    app.state.model = get_model()

    # ── Step 2: Documents + Embeddings ──
    embeddings, documents, labels = load_embeddings()

    if embeddings is None:
        logger.info("No cached embeddings found. Building from scratch...")
        documents, labels, label_names = load_documents()
        embeddings = embed_documents(documents, app.state.model)
        save_embeddings(embeddings, documents, labels)

    app.state.documents = documents
    app.state.embeddings = embeddings
    app.state.labels = labels

    # ── Step 3: FAISS Index (with metadata) ──
    index_data = load_index()
    if index_data is None:
        logger.info("No cached FAISS index found. Building...")
        index_data = build_index(embeddings, labels)
        save_index(index_data)
        
    app.state.index = index_data

    # ── Step 4: Clustering ──
    gmm, pca, cluster_probs, dominant_clusters = load_clustering()
    if gmm is None:
        logger.info("No cached clustering found. Fitting GMM...")
        reduced, pca = reduce_dimensions(embeddings)
        gmm = fit_gmm(reduced, n_clusters=15)
        cluster_probs = get_cluster_distributions(gmm, reduced)
        dominant_clusters = get_dominant_cluster(cluster_probs)
        save_clustering(gmm, pca, cluster_probs, dominant_clusters)
        analyze_clusters(documents, cluster_probs, dominant_clusters, embeddings)

    app.state.gmm = gmm
    app.state.pca = pca
    app.state.cluster_probs = cluster_probs
    app.state.dominant_clusters = dominant_clusters

    # ── Step 5: BM25 Index (for hybrid search) ──
    bm25 = load_bm25()
    if bm25 is None:
        logger.info("No cached BM25 index found. Building...")
        bm25 = BM25Index()
        bm25.fit(documents)
        save_bm25(bm25)
        
    app.state.bm25 = bm25

    # ── Step 6: Semantic Cache ──
    app.state.cache = SemanticCache(threshold=0.85)

    # ── Step 7: Hybrid Searcher ──
    app.state.hybrid_searcher = HybridSearcher(bm25_index=bm25)

    logger.info("Service ready.")

    yield  # Server runs here

    # Shutdown
    logger.info("Shutting down...")
    app.state._state.clear()


# ─────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Cognitive RAG Semantic Caching API",
    description="A high-performance semantic search and caching engine for ArXiv ML papers.",
    version="1.0.0",
    lifespan=lifespan
)

from app.config import settings

origins = settings.cors_origins.split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # allow_credentials cannot be True if allow_origins is ["*"]
    allow_credentials=False if "*" in origins else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_rate_limiting(app)

@app.middleware("http")
async def structlog_middleware(request: Request, call_next):
    # Create a unique request ID
    request_id = str(uuid.uuid4())
    
    # Bind request_id to structlog contextvars for the duration of the request
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else None
    )
    
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    # Log the request completion
    logger.info("Request completed", status_code=response.status_code, duration_ms=round(process_time, 2))
    
    # Return the request ID in headers for client tracing
    response.headers["X-Request-ID"] = request_id
    return response

app.include_router(router)
