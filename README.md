---
title: OCR & RAG Agent
emoji: 🛢️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8501
---
# Agent

## Run locally on Windows (Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

Open http://localhost:8502. Configure `.env` with the OpenAI-compatible vLLM
server used for chat and vision:

```env
VLLM_API_BASE=http://192.168.1.100:8700/v1
VLLM_API_KEY=empty
VLLM_MODEL=qwen3-vl-8b
LOCAL_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
LOCAL_EMBEDDING_DIM=768
LOCAL_EMBEDDING_CACHE=database/model_cache
LITELLM_LOCAL_MODEL_COST_MAP=True
DATABASE_URL=postgresql://drake@127.0.0.1:5433/drake_ai
```

Embeddings run locally on the Windows computer. The embedding model is
downloaded into `database/model_cache` on first use, then reused offline.

PostgreSQL tables and the local Guest account are created automatically on
startup. Start the dedicated Windows database before launching Streamlit:

```powershell
.\start-postgres.ps1
```

Stop it later with `.\stop-postgres.ps1`. The current app runs as a shared
Guest demo. Documents and chat history are stored in PostgreSQL; local document
vector indexes are stored under `database/`.
