# Intelligent Customer Support Agent

An enterprise-grade, highly efficient AI Agent built with LangGraph and FastAPI, featuring multi-domain architecture and semantic caching via RedisVL. 

This project demonstrates a production-ready approach to building LLM applications where frontend presentation and intelligent backend orchestration are seamlessly integrated.

## 🌟 Core Features
- **Semantic Caching:** Uses RedisVL and specialized embedding models (`ep-m-2026...`) to cache and rapidly return answers to semantically similar questions, drastically cutting down token costs.
- **Workflow Orchestration:** Powered by LangGraph, capable of complex reasoning, iterative reflection, and multi-step research iterations before replying to the user.
- **Full-Stack Integration:** FastAPI effectively serves both the complex backend AI logics and the modern, glassmorphic UI frontend, eliminating cross-origin (CORS) hassle and reducing deployment complexity.
- **Premium User Experience:** Advanced UI with dark/light mode parsing, real-time glowing "Thinking" states, and beautifully crafted macro components.

## 📂 Architecture & Directory Structure
Unlike standard toy examples, this repository embraces Domain-Driven Design (DDD). We do not simply divide code into `frontend` and `backend`. **The `src/` directory is the backend brain of the application.**

```text
.
├── frontend/             # Static UI Assets (HTML, CSS, JS with Glassmorphism)
└── src/                  # The Backend Core (Python)
    ├── api/              # Domain: Gateway & Routing (FastAPI server, Auth, Mounts)
    ├── workflow/         # Domain: Intelligence (LangGraph nodes, edges, states, prompts)
    ├── cache/            # Domain: Memory & Performance (RedisVL Semantic Caching)
    ├── knowledge/        # Domain: Libraries (Embeddings & Document Parsers)
    ├── common/           # Domain: Infrastructure (Logging, Security, Environments)
    └── testing/          # Domain: Evaluation (Automated scenario runners and logging)
```

## 🚀 Quick Start / Deployment

### 1. Prerequisites
- **Python 3.11+**
- **Redis Server** running locally or a provided remote `REDIS_URL`
- **ARK API Key** (Volcengine/Doubao models) 

### 2. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Start the Unified Server
Because the frontend is cleanly mounted into the FastAPI application, a single command boots the entire stack:
```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Open in Browser
Visit **[http://127.0.0.1:8000](http://127.0.0.1:8000)** and start chatting with your agent!
