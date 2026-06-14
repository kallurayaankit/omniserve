import streamlit as st
import requests
import time

BACKEND = "http://127.0.0.1:8002"

st.set_page_config(page_title="OmniServe", layout="wide")
st.title("🧠 OmniServe – All Features")

# ---------- Sidebar status ----------
st.sidebar.header("Backend Status")
try:
    resp = requests.get(f"{BACKEND}/", timeout=3)
    st.sidebar.success(f"Backend online (status {resp.status_code})")
except:
    st.sidebar.error("Backend unreachable")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🎤 Voice", "📈 Logs", "🧪 Fine‑tune"])

# ==================== TAB 1 – Chat ====================
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = requests.post(f"{BACKEND}/chat", json={"question":prompt}, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer","")
                    citations = data.get("citations",[])
                    st.markdown(answer)
                    if citations:
                        st.caption(f"Sources: {', '.join(citations)}")
                    st.session_state.messages.append({"role":"assistant","content":answer})
                    # TTS
                    try:
                        tts = requests.post(f"{BACKEND}/synthesize", json={"text":answer}, timeout=30)
                        if tts.status_code == 200:
                            st.audio(tts.content, format="audio/mp3")
                    except: pass
                else:
                    st.error(f"Error: {resp.text}")

# ==================== TAB 2 – Voice ====================
with tab2:
    st.header("🎤 Voice Transcription")
    audio_file = st.file_uploader("Upload audio (WAV, MP3...)", type=["wav","mp3","ogg","flac","m4a"])
    if audio_file:
        with st.spinner("Transcribing..."):
            files = {"file": (audio_file.name, audio_file.getvalue())}
            resp = requests.post(f"{BACKEND}/transcribe", files=files)
            if resp.status_code == 200:
                text = resp.json()["text"]
                st.text_area("Transcribed text:", text, height=150)
            else:
                st.error(f"Transcription failed: {resp.text}")

# ==================== TAB 3 – Logs ====================
with tab3:
    st.header("📈 Live Log Anomalies")
    if "log_anomalies" not in st.session_state:
        st.session_state.log_anomalies = []
    def fetch_sse():
        def listen():
            while True:
                try:
                    resp = requests.get(f"{BACKEND}/logs/recent", timeout=2)
                    if resp.status_code == 200:
                        new_anomalies = resp.json()
                        if isinstance(new_anomalies, list) and len(new_anomalies) != len(st.session_state.log_anomalies):
                            st.session_state.log_anomalies = new_anomalies
                except: pass
                time.sleep(2)
        import threading; threading.Thread(target=listen, daemon=True).start()
    if "sse_started" not in st.session_state:
        fetch_sse(); st.session_state.sse_started = True
    anomalies = st.session_state.log_anomalies
    if anomalies:
        for ev in reversed(anomalies[-5:]):
            with st.expander(f"Anomaly at {ev['timestamp']} (z={ev['z_score']})"):
                st.write(f"Error count: {ev['current_error_count']}")
                st.success(ev['explanation'])
                st.code("\n".join(ev['lines'][-10:]))
    else:
        st.info("No anomalies yet.")

# ==================== TAB 4 – Fine‑tuning ====================
with tab4:
    st.header("🧪 Fine‑tuning Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Generate Training Data"):
            with st.spinner("Generating..."):
                resp = requests.post(f"{BACKEND}/fine-tune/generate-data?num_examples=100", timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Generated {data['num_examples']} examples → training_data.json")
                else:
                    st.error(resp.text)
    with col2:
        if st.button("Start LoRA Fine‑tuning"):
            with st.spinner("Training (may take several minutes)..."):
                resp = requests.post(f"{BACKEND}/fine-tune/start-lora", timeout=600)
                if resp.status_code == 200:
                    st.success("LoRA adapter saved to lora_adapter/")
                else:
                    st.error(resp.text)
    with col3:
        if st.button("Run DPO (placeholder)"):
            resp = requests.post(f"{BACKEND}/fine-tune/dpo", timeout=60)
            if resp.status_code == 200:
                st.info(resp.json().get("message",""))
            else:
                st.error(resp.text)