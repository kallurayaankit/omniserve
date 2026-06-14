# ==================== OmniServe – Voice, Logs & Fine‑tuning ====================
import os, sys, traceback, logging, json, re, asyncio, threading, time, tempfile
from collections import deque

os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("omniserve.log")]
)
logger = logging.getLogger("omniserve")

# ---------- Core imports ----------
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# ---------- Optional imports (voice, logs, fine‑tuning) ----------
try:
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    WHISPER_READY = True
except Exception:
    WHISPER_READY = False
    logger.warning("Whisper not available – voice transcription disabled.")

try:
    import edge_tts
    TTS_READY = True
except ImportError:
    TTS_READY = False
    logger.warning("edge‑tts not available – text‑to‑speech disabled.")

try:
    import tailer
    TAILER_READY = True
except ImportError:
    TAILER_READY = False
    logger.warning("tailer not available – live log analysis disabled.")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    import torch
    FINE_TUNE_READY = True
except ImportError:
    FINE_TUNE_READY = False
    logger.warning("Fine‑tuning libraries missing – fine‑tuning disabled.")

# ---------- FastAPI app ----------
app = FastAPI(title="OmniServe")

# ---------- Qdrant ----------
qdrant = QdrantClient(path="omniserve_qdrant_data")
COLLECTION = "rag_docs"
if COLLECTION not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(COLLECTION, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    logger.info("Created Qdrant collection")

# ---------- Embedding & reranker ----------
embed = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ---------- BM25 ----------
corpus = []
bm25 = None
def rebuild_bm25():
    global corpus, bm25
    pts = qdrant.scroll(collection_name=COLLECTION, limit=10000, with_payload=True)[0]
    corpus = [p.payload.get("text", "") for p in pts]
    if corpus:
        bm25 = BM25Okapi([doc.lower().split() for doc in corpus])
rebuild_bm25()

# ---------- Log analysis (if tailer available) ----------
if TAILER_READY:
    LOG_FILE = "synapse.log"
    WINDOW_SIZE = 60
    ERROR_THRESHOLD_Z = 2.5
    CHECK_INTERVAL = 5
    LOG_LINES_AROUND_ANOMALY = 20
    anomaly_events = deque(maxlen=50)
    subscribers = set()

    def explain_anomaly(lines):
        snippet = "\n".join(lines[-LOG_LINES_AROUND_ANOMALY:])
        prompt = f"Explain this log anomaly in plain English:\n{snippet}\n\nAnswer:"
        try:
            resp = ollama.chat(model="mistral:latest", messages=[{"role":"user","content":prompt}])
            return resp["message"]["content"]
        except:
            return "LLM unavailable for explanation."

    def monitor_log():
        window = deque(maxlen=WINDOW_SIZE)
        last_check = time.time()
        error_counts = []
        def recent_errors(sec):
            cutoff = time.time() - sec
            return [c for t,c in error_counts if t >= cutoff]
        while not os.path.exists(LOG_FILE):
            time.sleep(2)
        for line in tailer.follow(open(LOG_FILE, encoding="utf-8", errors="ignore")):
            now = time.time()
            if now - last_check >= CHECK_INTERVAL:
                recent = recent_errors(WINDOW_SIZE)
                if recent:
                    mean = sum(recent)/len(recent)
                    std = (sum((x-mean)**2 for x in recent)/len(recent))**0.5 if len(recent)>1 else 0
                    cur = sum(recent_errors(10))
                    if std>0 and (cur-mean)/std > ERROR_THRESHOLD_Z:
                        all_lines = []
                        try:
                            with open(LOG_FILE, encoding="utf-8") as f:
                                all_lines = f.readlines()
                        except: pass
                        ctx = all_lines[-LOG_LINES_AROUND_ANOMALY:]
                        expl = explain_anomaly(ctx)
                        event = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "z_score": round((cur-mean)/std, 2),
                            "current_error_count": cur,
                            "explanation": expl,
                            "lines": ctx
                        }
                        anomaly_events.append(event)
                        for q in list(subscribers):
                            try: q.put_nowait(event)
                            except: pass
                last_check = now
            if "ERROR" in line or "CRITICAL" in line:
                error_counts.append((time.time(),1))
            else:
                error_counts.append((time.time(),0))
            cutoff = time.time()-WINDOW_SIZE
            error_counts = [(t,c) for t,c in error_counts if t>cutoff]

    threading.Thread(target=monitor_log, daemon=True).start()

    @app.get("/logs/stream")
    async def stream_logs():
        q = asyncio.Queue()
        subscribers.add(q)
        async def gen():
            try:
                while True:
                    while not q.empty():
                        ev = q.get_nowait()
                        yield f"data: {json.dumps(ev, default=str)}\n\n"
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=1.0)
                        yield f"data: {json.dumps(ev, default=str)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"
            finally:
                subscribers.discard(q)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/logs/recent")
    def recent_anomalies():
        return list(anomaly_events)[-20:]

else:
    @app.get("/logs/recent")
    def recent_anomalies():
        return {"error": "Log analysis not available (install tailer)"}

# ---------- Voice endpoints ----------
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not WHISPER_READY:
        raise HTTPException(503, "Voice transcription not available")
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        segments, _ = whisper_model.transcribe(tmp_path)
        text = " ".join([seg.text for seg in segments])
        return {"text": text}
    finally:
        os.unlink(tmp_path)

@app.post("/synthesize")
async def synthesize(data: dict = None):
    if not TTS_READY:
        raise HTTPException(503, "TTS not available")
    text = data.get("text", "") if data else ""
    if not text:
        raise HTTPException(400, "No text provided")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        mp3_path = tmp.name
    try:
        import subprocess
        cmd = ["edge-tts", "--voice", "en-US-AriaNeural", "--text", text, "--write-media", mp3_path]
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        with open(mp3_path, "rb") as f:
            audio_bytes = f.read()
        return Response(content=audio_bytes, media_type="audio/mpeg")
    finally:
        if os.path.exists(mp3_path):
            os.unlink(mp3_path)

# ---------- Fine‑tuning endpoints ----------
@app.post("/fine-tune/generate-data")
def generate_data(num_examples: int = 200):
    if not FINE_TUNE_READY:
        raise HTTPException(400, "Fine‑tuning libraries not installed")
    examples = []
    for _ in range(num_examples):
        prompt = 'Generate a support ticket JSON: {"ticket":"...","output":{"issue_type":"...","priority":"...","description":"..."}}'
        try:
            resp = ollama.chat(model="mistral:latest", messages=[{"role":"user","content":prompt}], format="json")
            data = json.loads(resp["message"]["content"])
            if "ticket" in data and "output" in data:
                examples.append(data)
        except: pass
        if len(examples) >= num_examples: break
    with open("training_data.json", "w") as f:
        json.dump(examples, f)
    return {"status": "ok", "num_examples": len(examples), "file": "training_data.json"}

@app.post("/fine-tune/start-lora")
def start_lora():
    if not FINE_TUNE_READY:
        raise HTTPException(400, "Fine‑tuning libraries not installed")
    if not os.path.exists("training_data.json"):
        raise HTTPException(400, "No training data found. Generate it first.")
    with open("training_data.json") as f:
        data = json.load(f)
    # Minimal LoRA fine‑tune (small model to avoid OOM)
    model_name = "distilgpt2"  # tiny for demo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    # Create dataset
    texts = [f"Ticket: {d['ticket']}\nOutput: {json.dumps(d['output'])}" for d in data]
    ds = Dataset.from_dict({"text": texts})
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)
    ds = ds.map(tokenize, batched=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="lora_adapter", per_device_train_batch_size=1, num_train_epochs=1, logging_steps=10),
        train_dataset=ds,
    )
    trainer.train()
    model.save_pretrained("lora_adapter")
    return {"status": "ok", "adapter_path": "lora_adapter"}

@app.post("/fine-tune/dpo")
def run_dpo():
    return {"status": "ok", "message": "DPO placeholder – manually provide preference pairs."}

# ---------- Chat endpoint (hybrid RAG) ----------
class ChatReq(BaseModel):
    question: str
    message: str = ""

def enforce_citations(query, context, citations, max_retries=3):
    cite_text = "\n".join([f"[{i+1}] {cid}" for i,cid in enumerate(citations)]) if citations else ""
    sys_msg = 'You are a helpful assistant. Answer in JSON: {"answer":"...", "citations":[citation numbers used]}'
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    if cite_text:
        prompt = f"Available citations:\n{cite_text}\n\n{prompt}"
    for _ in range(max_retries):
        try:
            resp = ollama.chat(model="mistral:latest", messages=[{"role":"system","content":sys_msg},{"role":"user","content":prompt}], format="json")
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
        qv = embed.encode(q).tolist()
        vec_hits = qdrant.query_points(collection_name=COLLECTION, query=qv, limit=10).points
        bm25_hits = []
        if bm25:
            scores = bm25.get_scores(q.lower().split())
            bm25_hits = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)[:10]
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
    return {"status":"ok","service":"OmniServe"}

# ---------- Start server ----------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn on http://127.0.0.1:8002")
    uvicorn.run(app, host="127.0.0.1", port=8002)