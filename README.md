# 🏥 HealthCare AI Calling Assistant
 
A locally-running, voice-powered medical assistant built with Python and Streamlit. Speak to it — it listens, understands, and responds. Capable of diagnosing heart disease risk using a trained XGBoost model and answering general medical queries using PDF-based RAG (Retrieval Augmented Generation).


---

## 📽️ Demo Flow

```
You speak  →  Whisper transcribes  →  LLM routes intent
                                              │
                          ┌───────────────────┴──────────────────┐
                          ▼                                       ▼
               Heart Diagnosis                           General Query
               (asks 13 questions)                    (searches PDF knowledge base)
               → XGBoost predicts                     → LLM answers from context
                          │                                       │
                          └───────────────────┬──────────────────┘
                                              ▼
                                    TTS speaks the response
                                    Audio plays in browser
```

---

## ✨ Features

- 🎤 **Voice Input** — speak directly into the mic via Streamlit
- 🧠 **Intent Detection** — LLM automatically routes between heart diagnosis and general queries
- ❤️ **Heart Disease Prediction** — conversationally collects 13 clinical features and runs your trained XGBoost model
- 📄 **PDF RAG** — answers hospital/doctor queries from your own uploaded PDF knowledge base
- 🔊 **Voice Response** — replies are spoken back using text-to-speech
- 💻 **Fully Local** — no external APIs, everything runs on your machine

---

## 🏗️ Architecture

```
medical-voice-agent/
├── app.py                   # Streamlit frontend & main logic
├── setup_rag.py             # One-time PDF indexing script
├── requirements.txt
│
├── modules/
│   ├── stt.py               # Speech-to-Text (Whisper)
│   ├── tts.py               # Text-to-Speech (pyttsx3)
│   ├── llm.py               # LLM loader & chat (Qwen2.5-1.5B)
│   ├── intent_router.py     # Classifies user intent
│   ├── heart_agent.py       # Feature collection + XGBoost prediction
│   └── rag_agent.py         # PDF retrieval + LLM answering
│
├── models/
│   ├── xgboost_heart.pkl    # Trained binary XGBoost model
│   ├── scaler.pkl           # Fitted StandardScaler
│   └── feature_columns.pkl  # Column order from training
│
└── rag/
    ├── pdfs/                # Place your PDF knowledge base here
    └── vectorstore/         # FAISS index (auto-generated)
```

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Speech to Text | OpenAI Whisper (small) |
| Language Model | Qwen2.5-1.5B-Instruct |
| Heart Prediction | XGBoost (binary classification) |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Store | FAISS (local) |
| Text to Speech | pyttsx3 |
| PDF Parsing | pypdf |

---

## ⚙️ Requirements

- Python **3.11** (3.13 not supported by PyTorch yet)
- NVIDIA GPU with CUDA support (tested on RTX 3050 4GB VRAM)
- **ffmpeg** installed and added to system PATH
- ~3.5 GB free disk space for model downloads

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/healthcare-ai-calling.git
cd healthcare-ai-calling
```

### 2. Create virtual environment with Python 3.11
```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### 3. Install PyTorch with CUDA
```bash
# Check your CUDA version first: nvidia-smi
# Then install matching PyTorch (example: CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install remaining dependencies
```bash
pip install -r requirements.txt
```

### 5. Install ffmpeg (Windows)
Download from https://github.com/BtbN/FFmpeg-Builds/releases, extract, and add the `bin/` folder to your system PATH.

Verify:
```bash
ffmpeg -version
```

---

## 🤖 Model Setup

### Trained Models (XGBoost)
Add these three files to the `models/` folder. Generate them by running the training notebook and adding these cells at the end:

```python
import joblib, os
os.makedirs("models", exist_ok=True)

joblib.dump(scaler,          "models/scaler.pkl")
joblib.dump(list(X.columns), "models/feature_columns.pkl")
joblib.dump(xgb_b,           "models/xgboost_heart.pkl")
```

### AI Models (auto-downloaded on first run)
| Model | Size | Cached Location |
|-------|------|----------------|
| Whisper small | ~460 MB | `~/.cache/whisper/` |
| Qwen2.5-1.5B-Instruct | ~3 GB | `~/.cache/huggingface/hub/` |
| all-MiniLM-L6-v2 | ~90 MB | `~/.cache/huggingface/hub/` |

> Models download once. Every subsequent run loads from cache.

---

## 📄 PDF Knowledge Base

Place your PDF files in `rag/pdfs/` (any filename works), then run the indexer:

```bash
python setup_rag.py
```

This reads all PDFs, chunks the text, generates embeddings, and saves a FAISS index to `rag/vectorstore/`. Re-run this script whenever you add new PDFs.

**Recommended PDFs to include:**
- Hospital & Doctor Directory
- Disease & Symptoms Guide
- Medical Tests & Lab Reference
- Emergency First Aid Guide
- Heart Disease Patient Guide

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

> First launch takes **30–60 seconds** to load all models. Subsequent launches are faster.

---

## 🗣️ How to Use

**General Medical Query:**
> Press mic → *"What are the timings for the cardiology department?"*
> Assistant searches your PDFs and speaks the answer.

**Heart Disease Check:**
> Press mic → *"I want to check if I have heart disease"*
> Assistant asks 13 clinical questions one by one, then predicts risk using XGBoost.

> ℹ️ This is a **push-to-talk** interface. Press the mic button once per message.

---

## 📊 Model Performance (Training Results)

| Model | Binary Accuracy | Precision | Recall | F1 |
|-------|----------------|-----------|--------|----|
| Decision Tree | 78.8% | 79.4% | 83.3% | 81.3% |
| Random Forest | 85.3% | 84.4% | 90.2% | 87.2% |
| SVM | 73.9% | 75.0% | 79.4% | 77.1% |
| **XGBoost** | **85.3%** | **83.8%** | **91.2%** | **87.3%** |

Dataset: [Heart Disease UCI](https://archive.ics.uci.edu/dataset/45/heart+disease) — 920 records, 13 features, binary classification (disease / no disease).

---

## 🖥️ VRAM Usage (RTX 3050 4GB)

| Component | Device | VRAM |
|-----------|--------|------|
| Whisper small | GPU | ~500 MB |
| Qwen2.5-1.5B (fp16) | GPU | ~1.2 GB |
| MiniLM Embeddings | CPU | — |
| FAISS Index | CPU/RAM | — |
| pyttsx3 TTS | CPU | — |
| **Total GPU** | | **~1.7 GB** ✅ |

---

## ⚠️ Disclaimer

This application is for **educational and demonstration purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

## 📜 License

MIT License — free to use, modify, and distribute with attribution.
