import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import sqlite3, json, logging, traceback, re
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from pydantic import BaseModel
import ollama

# --- Setup logging to file AND console (so we see everything) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("omniserve.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("omniserve")

app = FastAPI(title="OmniServe RAG")

# --- Qdrant (auto-create collection) ---
qdrant = QdrantClient(path="omniserve_qdrant_data")
COLLECTION = "rag_docs"
if COLLECTION not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(COLLECTION, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    logger.info("Created Qdrant collection")

# --- Embedding model ---
embed = SentenceTransformer('all-MiniLM-L6-v2')

# --- BM25 index (built from Qdrant) ---
corpus = []
bm25 = None

def rebuild_bm25():
    global corpus, bm25
    points = qdrant.scroll(collection_name=COLLECTION, limit=10000, with_payload=True)[0]
    corpus = [p.payload.get("text", "") for p in points]
    if corpus:
        bm25 = BM25Okapi([doc.lower().split() for doc in corpus])

rebuild_bm25()

# --- Cross-encoder reranker ---
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# --- Chat request model ---
class ChatReq(BaseModel):
    question: str
    message: str = ""

# --- Citation enforcement ---
def enforce_citations(query, context, citations, max_retries=3):
    cite_text = "\n".join([f"[{i+1}] {cid}" for i,cid in enumerate(citations)]) if citations else ""
    sys_msg = 'You are a helpful assistant. Answer strictly in JSON: {"answer":"...", "citations":[citation numbers used]}'
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    if cite_text:
        prompt = f"Available citations:\n{cite_text}\n\n{prompt}"
    for _ in range(max_retries):
        try:
            resp = ollama.chat(model="mistral:latest", messages=[
                {"role":"system","content":sys_msg},
                {"role":"user","content":prompt}
            ], format="json")
            data = json.loads(resp["message"]["content"])
            nums = data.get("citations", [])
            if context.strip() and not nums:
                raise ValueError("Missing citations")
            final = [citations[n-1] for n in nums if 1 <= n <= len(citations)]
            return {"answer": data.get("answer",""), "citations": final}
        except Exception as e:
            prompt += f"\nPrevious attempt failed: {e}. Return valid JSON."
    return {"answer":"Could not produce a properly cited answer.","citations":[]}

@app.post("/chat")
def chat(req: ChatReq):
    try:
        q = req.question
        # Vector search
        qv = embed.encode(q).tolist()
        vec_hits = qdrant.query_points(collection_name=COLLECTION, query=qv, limit=10).points
        # BM25 search
        bm25_hits = []
        if bm25:
            scores = bm25.get_scores(q.lower().split())
            bm25_hits = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)[:10]
        # Merge with Reciprocal Rank Fusion
        all_pts = qdrant.scroll(collection_name=COLLECTION, limit=10000, with_payload=True)[0]
        text_to_id = {p.payload.get("text",""): p.id for p in all_pts}
        vec_ids = [p.id for p in vec_hits]
        bm_ids = [text_to_id.get(t) for t,_ in bm25_hits if text_to_id.get(t)]
        k=60; scores={}
        for rank, did in enumerate(vec_ids): scores[did]=scores.get(did,0)+1/(k+rank+1)
        for rank, did in enumerate(bm_ids): scores[did]=scores.get(did,0)+1/(k+rank+1)
        top_ids = sorted(scores, key=scores.get, reverse=True)[:5]
        fused = []
        for did in top_ids:
            for p in all_pts:
                if p.id == did: fused.append(p); break
        # Cross-encoder rerank
        if fused:
            pairs = [[q, p.payload.get("text","")] for p in fused]
            ce_scores = reranker.predict(pairs)
            ranked = sorted(zip(fused, ce_scores), key=lambda x: x[1], reverse=True)
            final = [p for p,_ in ranked[:3]]
        else:
            final = []
        ctx = "\n\n".join([p.payload.get("text","") for p in final])
        cids = [p.id for p in final]
        return enforce_citations(q, ctx, cids)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def health():
    return {"status":"ok","service":"OmniServe RAG"}