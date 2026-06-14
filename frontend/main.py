import streamlit as st
import requests
import time

BACKEND = "http://127.0.0.1:8002"

st.set_page_config(page_title="OmniServe", layout="wide")
st.title("🧠 OmniServe Chat + Voice + Logs")

# Sidebar
st.sidebar.header("Backend Status")
try:
    resp = requests.get(f"{BACKEND}/", timeout=3)
    st.sidebar.success(f"Backend online (status {resp.status_code})")
except Exception as e:
    st.sidebar.error(f"Backend unreachable: {e}")

# Voice Assistant
st.header("🎤 Voice Assistant")
audio_file = st.file_uploader("Upload audio (WAV, MP3...)", type=["wav","mp3","ogg","flac","m4a"])
if audio_file:
    with st.spinner("Transcribing..."):
        files = {"file": (audio_file.name, audio_file.getvalue())}
        resp = requests.post(f"{BACKEND}/transcribe", files=files)
        if resp.status_code == 200:
            transcribed = resp.json()["text"]
            st.text_area("Transcribed:", transcribed, height=100)
            st.session_state["voice_query"] = transcribed
        else:
            st.error(f"Transcription failed: {resp.text}")

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if "voice_query" in st.session_state and st.session_state.voice_query:
    prompt = st.session_state.voice_query
    st.session_state.voice_query = None
else:
    prompt = st.chat_input("Ask about your documents...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = requests.post(f"{BACKEND}/chat", json={"question": prompt, "message": prompt}, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "")
                citations = data.get("citations", [])
                st.markdown(answer)
                if citations:
                    st.caption(f"Sources: {', '.join(citations)}")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                # TTS
                st.markdown("🔊 **Listen:**")
                try:
                    tts_resp = requests.post(f"{BACKEND}/synthesize", json={"text": answer}, timeout=30)
                    if tts_resp.status_code == 200:
                        st.audio(tts_resp.content, format="audio/mp3")
                    else:
                        st.warning("TTS unavailable")
                except Exception as e:
                    st.warning(f"TTS error: {e}")
            else:
                st.error(f"Chat error: {resp.text}")

# Live Log Analysis
st.header("📈 Live Log Analysis")
if "log_anomalies" not in st.session_state:
    st.session_state.log_anomalies = []

def fetch_sse():
    import threading
    def listen():
        while True:
            try:
                resp = requests.get(f"{BACKEND}/logs/recent", timeout=2)
                if resp.status_code == 200:
                    new_anomalies = resp.json()
                    if len(new_anomalies) != len(st.session_state.log_anomalies):
                        st.session_state.log_anomalies = new_anomalies
            except:
                pass
            time.sleep(2)
    t = threading.Thread(target=listen, daemon=True)
    t.start()

if "sse_started" not in st.session_state:
    fetch_sse()
    st.session_state.sse_started = True

anomalies = st.session_state.log_anomalies
if anomalies:
    for event in reversed(anomalies[-5:]):
        with st.expander(f"Anomaly at {event['timestamp']} (z={event['z_score']})"):
            st.write(f"Error count: {event['current_error_count']}")
            st.success(event['explanation'])
            st.code("\n".join(event['lines'][-10:]))
else:
    st.info("No anomalies yet. Waiting for log data...")