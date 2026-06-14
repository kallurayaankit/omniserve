# ==================== OmniServe Hybrid RAG – Self-contained Runner ====================
import os, sys, traceback, logging

# 1. Setup working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 2. Configure logging to file AND console so we see errors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("omniserve.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("omniserve")

try:
    # ---------- Import everything ----------
    from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    import uuid, json, sqlite3, hashlib, tempfile, time
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance
    from sentence_transformers import SentenceTransformer
    from prometheus_client import Counter, Histogram, generate_latest
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder
    from pydantic import BaseModel
    import re

    # ---------- FastAPI app ----------
    app = FastAPI(title="OmniServe")

    # ---------- Prometheus metrics ----------
    CHAT_REQUESTS = Counter("omniserve_chat_requests", "Chat requests")
    ERRORS = Counter("omniserve_errors", "Errors")
    LATENCY = Histogram("omniserve_chat_latency", "Chat latency")

    # ---------- SQLite (in‑memory for simplicity, change to file if needed) ----------
    DB = sqlite3.connect("omniserve.db", check_same_thread=False)
    DB.row_factory = sqlite3.Row
    DB.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, email TEXT UNIQUE, password_hash TEXT)")
    DB.execute("CREATE TABLE IF NOT EXISTS agent_config (user_id INTEGER PRIMARY KEY, system_prompt TEXT, tools TEXT)")

    # ---------- Qdrant ----------
    qdrant_client = QdrantClient(path="omniserve_qdrant_data")
    COLLECTION_NAME = "omniserve_docs"
    # Create collection if missing
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")

    # ---------- Embedding model ----------
    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # ---------- Ollama model name ----------
    OLLAMA_MODEL = "mistral:latest"

    # ---------- BM25 ----------
    bm25_corpus = []
    bm25_index = None

    def build_bm25():
        global bm25_corpus, bm25_index
        all_points = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=10000, with_payload=True)[0]
        bm25_corpus = [p.payload.get("text", "") for p in all_points]
        if bm25_corpus:
            tokenized = [doc.lower().split() for doc in bm25_corpus]
            bm25_index = BM25Okapi(tokenized)
        else:
            bm25_index = None

    build_bm25()

    # ---------- Cross-encoder ----------
    logger.info("Loading cross-encoder (first time may download ~200MB)...")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # ---------- Chat endpoint ----------
    class ChatRequest(BaseModel):
        question: str
        message: str = ""

    def enforce_citations(query, context, citations, max_retries=3):
        import ollama
        citation_text = "\n".join([f"[{i+1}] {cid}" for i, cid in enumerate(citations)]) if citations else ""
        system_msg = """You are a helpful assistant. Answer in JSON: {"answer": "...", "citations": [citation numbers used]}."""
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        if citation_text:
            prompt = f"Available citations:\n{citation_text}\n\n{prompt}"
        for attempt in range(max_retries):
            try:
                resp = ollama.chat(model=OLLAMA_MODEL, messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ], format="json")
                raw = resp["message"]["content"]
                data = json.loads(raw)
                # extract valid citations
                cited_nums = data.get("citations", [])
                if context.strip() and not cited_nums:
                    raise ValueError("Missing citations")
                final = [citations[n-1] for n in cited_nums if 1 <= n <= len(citations)]
                return {"answer": data.get("answer", ""), "citations": final}
            except Exception as e:
                prompt = f"{prompt}\n\nPrevious attempt failed: {e}. Ensure valid JSON with citations."
        return {"answer": "I couldn't provide a properly cited answer.", "citations": []}

    @app.post("/chat")
    def chat(req: ChatRequest):
        try:
            query = req.question
            # Vector search
            qvec = embedding_model.encode(query).tolist()
            vec_hits = qdrant_client.query_points(collection_name=COLLECTION_NAME, query=qvec, limit=10).points
            # BM25 search
            bm25_hits = []
            if bm25_index:
                scores = bm25_index.get_scores(query.lower().split())
                bm25_hits = sorted(zip(bm25_corpus, scores), key=lambda x: x[1], reverse=True)[:10]
            # RRF merge
            all_pts = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=10000, with_payload=True)[0]
            text_to_id = {p.payload.get("text", ""): p.id for p in all_pts}
            vec_ids = [p.id for p in vec_hits]
            bm_ids = [text_to_id.get(t) for t,_ in bm25_hits if text_to_id.get(t)]
            k=60; scores = {}
            for rank, did in enumerate(vec_ids): scores[did] = scores.get(did,0) + 1/(k+rank+1)
            for rank, did in enumerate(bm_ids): scores[did] = scores.get(did,0) + 1/(k+rank+1)
            top_ids = sorted(scores, key=scores.get, reverse=True)[:5]
            fused = []
            for did in top_ids:
                for p in all_pts:
                    if p.id == did: fused.append(p); break
            # Cross-encoder rerank
            if fused:
                pairs = [[query, p.payload.get("text","")] for p in fused]
                ce_scores = cross_encoder.predict(pairs)
                reranked = sorted(zip(fused, ce_scores), key=lambda x: x[1], reverse=True)
                final_chunks = [p for p,_ in reranked[:3]]
            else:
                final_chunks = []
            context = "\n\n".join([p.payload.get("text","") for p in final_chunks])
            chunk_ids = [p.id for p in final_chunks]
            return enforce_citations(query, context, chunk_ids)
        except Exception as e:
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    # ---------- Health endpoint ----------
    @app.get("/")
    def health():
        return {"status": "ok", "service": "OmniServe"}

    # ---------- Start server ----------
    import uvicorn
    logger.info("Starting Uvicorn on http://127.0.0.1:8002")
    uvicorn.run(app, host="127.0.0.1", port=8002)

except Exception as e:
    logger.error("FATAL ERROR during startup", exc_info=True)
    print("\n\n*** OMNISERVE CRASHED ***\n", file=sys.stderr)
    traceback.print_exc()
    input("Press Enter to exit...")
    sys.exit(1)