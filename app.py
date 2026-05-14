import hashlib
import streamlit as st
from modules.stt import transcribe
from modules.tts import speak
from modules.intent_router import classify_intent
from modules.rag_agent import answer_query
from modules.heart_agent import (
    FEATURES, get_next_question, parse_value,
    is_complete, predict_heart_disease
)

st.set_page_config(page_title="Medical Voice Assistant", page_icon="🏥")
st.title("🏥 Medical Voice Assistant")

st.markdown("""
<style>
/* Hide the playback bar inside the mic widget */
[data-testid="stAudioInput"] audio { display: none !important; }
/* Hide the response audio player visually — audio still plays */
[data-testid="stAudio"] {
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "idle"
if "collected" not in st.session_state:
    st.session_state.collected = {}
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None
if "pending_audio" not in st.session_state:
    st.session_state.pending_audio = None

# ── Sidebar: reset button ─────────────────────────────────────────────────────
with st.sidebar:
    if st.button("🗑️ Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.mode = "idle"
        st.session_state.collected = {}
        st.session_state.last_audio_hash = None
        st.session_state.pending_audio = None
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Play response audio (invisible player, audio still plays) ─────────────────
if st.session_state.pending_audio:
    st.audio(st.session_state.pending_audio, autoplay=True)
    st.session_state.pending_audio = None

# ── Mic — always available ────────────────────────────────────────────────────
audio_file = st.audio_input("🎤 Press to speak")

if audio_file is not None:
    audio_bytes = audio_file.getbuffer().tobytes()
    audio_hash = hashlib.md5(audio_bytes).hexdigest()

    if audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = audio_hash

        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)
        user_text = transcribe("temp_audio.wav")
        st.session_state.messages.append({"role": "user", "content": user_text})

        if st.session_state.mode == "collecting":
            current_idx = len(st.session_state.collected)
            feature = FEATURES[current_idx]
            value = parse_value(user_text, feature)

            if value is None:
                response = "Sorry, I couldn't understand. Please repeat your answer."
            else:
                st.session_state.collected[feature["key"]] = value

                if is_complete(st.session_state.collected):
                    result = predict_heart_disease(st.session_state.collected)
                    if result == 1:
                        response = "Based on your inputs, I recommend you visit a cardiologist soon."
                    else:
                        response = "Based on your inputs, you do not show signs of heart disease. Stay healthy!"
                    st.session_state.mode = "idle"
                    st.session_state.collected = {}
                else:
                    response = get_next_question(st.session_state.collected)
        else:
            intent = classify_intent(user_text)
            if intent == "heart_diagnosis":
                st.session_state.mode = "collecting"
                st.session_state.collected = {}
                response = "Sure, I will help you with that. " + FEATURES[0]["question"]
            else:
                response = answer_query(user_text)

        st.session_state.messages.append({"role": "assistant", "content": response})
        speak(response, "output.wav")
        with open("output.wav", "rb") as f:
            st.session_state.pending_audio = f.read()
        st.rerun()
