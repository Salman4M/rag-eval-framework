We are building a RAG Eval Framework — a standalone evaluation system that wraps an existing RAG project (rag-document-assistant, connected via GitHub). The goal is to build a professional eval pipeline in phases, learning everything deeply as we go.

The RAG Project (already built)
The existing project is a document Q&A API. Key details:

FastAPI backend, PostgreSQL + SQLAlchemy, ChromaDB vector store
fastembed (BAAI/bge-small-en-v1.5, 384 dims) for embeddings
BAAI/bge-reranker-base cross-encoder for reranking (retrieves 10, reranks to 4)
Qwen2.5 via Ollama for generation
pdfplumber + pypdf for PDF extraction (text + tables)
Per-user isolation in ChromaDB via user_id metadata filter
Personal facts memory — LLM extracts facts from conversations, stored in PostgreSQL
SSE upload progress streaming
Rate limiting via slowapi (20/min on /ask, 5/hour on /upload)
Key endpoints: POST /upload, POST /ask, GET /documents, DELETE /documents/{filename}, DELETE /clear
/ask returns {"answer": "...", "sources": [{"text": "...", "page": N, "filename": "..."}]}


The Eval Framework We Are Building
A new separate project called rag-eval-framework. Here is the full planned architecture:
rag-eval-framework/
├── datasets/
│   ├── raw/                    # source PDFs used for test cases
│   ├── eval_dataset.json       # hand-crafted + generated Q&A pairs
│   └── baselines/              # saved results per version/date
│
├── evaluators/
│   ├── ragas_evaluator.py      # Ragas + Ollama as judge
│   └── custom_evaluator.py     # custom metrics built from scratch
│
├── judges/
│   └── llm_judge.py            # LLM-as-judge core logic (reusable)
│
├── dataset_builder/
│   ├── generator.py            # auto-generate Q&A from PDFs via LLM
│   └── reviewer.py             # CLI to review/approve generated cases
│
├── reporters/
│   ├── console.py              # terminal output with rich tables
│   └── html_reporter.py        # visual HTML report with trends
│
├── runner.py                   # main entry point
├── config.yaml                 # models, thresholds, API URL, paths
├── compare.py                  # regression detection, baseline diff
└── .github/workflows/eval.yml  # CI integration

The 6 Phases (in order)
Phase 1 — Dataset Foundation
Build the dataset schema and tooling. eval_dataset.json contains question, expected answer, source document, page number, difficulty (factual / reasoning / multi-hop), category. Build generator.py — feeds a PDF to Ollama and auto-generates candidate Q&A pairs. Build reviewer.py — CLI to approve/reject/edit generated pairs. Target: 30-50 high quality test cases.
Phase 2 — Ragas + Ollama Evaluator
Run Ragas fully locally using Ollama as the judge model. ragas_evaluator.py hits the live RAG API, collects answers + retrieved chunks, scores with Ragas. Metrics: faithfulness, answer relevancy, context precision, context recall. Output: JSON results file.
Phase 3 — Custom Evaluator From Scratch
Build the same metrics without Ragas — raw LLM calls only. llm_judge.py sends (question, answer, context) to Ollama with structured scoring prompts and parses scores back. Custom faithfulness, hallucination detection, retrieval scoring. Compare custom scores vs Ragas scores on same dataset.
Phase 4 — Regression Detection
Save baseline results with timestamp + git commit hash. compare.py diffs two result files and shows delta per metric per test case. config.yaml defines minimum acceptable thresholds — fail if below. runner.py --compare baseline produces a delta table.
Phase 5 — CI Integration
GitHub Actions workflow — runs eval suite on push to main, fails build if metrics drop below threshold vs baseline, posts summary as PR comment.
Phase 6 — Reporting
Console reporter with rich colored tables. HTML reporter with metric trends over time, per-case breakdown, worst performing cases highlighted.

Tech Stack

Language: Python
RAG API: existing rag-document-assistant running locally on http://localhost:8000
Eval framework: Ragas (Phase 2) + custom from scratch (Phase 3)
LLM judge: Ollama — DeepSeek R1 7B (deepseek-r1:7b)
Dataset format: JSON
CI: GitHub Actions
Reporting: rich (terminal) + Jinja2 (HTML)
Config: PyYAML


Hardware

GPU: ~8GB VRAM
Models that run comfortably: up to 14B parameters (Q4_K_M)
Currently using Qwen2.5 for generation, planning to test Qwen3 8B as upgrade


Where We Start
Phase 1 — dataset schema and generator.py first.
Start by defining the eval_dataset.json schema, then build the generator that takes a PDF and produces candidate Q&A pairs using Ollama. Then build the CLI reviewer. Do not skip ahead — the dataset quality determines everything else.