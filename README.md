# rag-eval-framework

Evaluation framework for [rag-document-assistant](../rag-document-asistant).


### Setup
```bash
cd rag-eval-framework
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy the example config and fill in your token:
```bash
cp config.example.yaml config.yaml
```

Add your RAG API token to '.env':
```
RAG_API_TOKEN=your_access_token_here
```

---

## Phase 1 — Dataset Builder


Make sure Ollama is running with qwen2.5 pulled:
```bash
ollama pull qwen2.5
```

### 1. Drop your PDFs
Put source PDFs in `datasets/raw/`.

### 2. Generate candidates
```bash
python -m dataset_builder.generator --pdf datasets/raw/yourfile.pdf
```

Options:
- `--pages 1-10` — only process pages 1 through 10
- `--questions-per-page 5` — override the default (3) from config.yaml
- `--overwrite` — start fresh instead of appending to existing candidates

Outputs to `datasets/raw_candidates.json`.

### 3. Review candidates
```bash
python -m dataset_builder.reviewer
```

Controls: `(a)pprove  (e)dit  (r)eject  (s)kip  (q)uit`

Approved cases land in `datasets/eval_dataset.json`.
Progress is saved automatically — you can quit and resume at any time.

---

## Phase 2 — Ragas Evaluator

Runs in two stages to avoid loading two models at the same time due to hardware limitations on my laptop. (Even you can use better models for judge and handle proccess depend on the hardware)

Make sure `python_tutorial.pdf` (or your source PDF) is uploaded to the RAG API for your test user.


### Stage 1 - Collect RAG API answers
RAG API must be running. Ollama should be idle.
```bash
python runner.py --eval ragas
```
Saves answers to 'datasets/baselines/TIMESTAMP_collected.json'

### Stage 2 - Score with Ragas
Stop the RAG API first.Then:
```bash
python runner.py --eval ragas --score datasets/baselines/TIMESTAMP_collected.json
```

Requires `deepseek-r1:7b` in Ollama and `bge-small-en-v1.5` from HuggingFace
```bash
ollama pull deepseek-r1:7b
uv add langchain-huggingface
```

Results saved to 'datasets/baselines/TIMESTAMP_ragas.json'.

---


## Phase 3 - Cusom Evaluator

Rebuilds the same metrics from scratch using raw Ollama calls and cosine similarity. No Ragas dependency

Score with custom judge:
```bash
python runner.py --eval custom --score datasets/baselines/TIMESTAMP_collected.json
```

Score and compare against Ragas baseline:
```bash
python runner.py --eval custom \ 
    --score datasets/baselines/TIMESTAMP_collected.json \
    --compare datasets/baselines/TIMESTAMP_ragas.json
```

Results saved to `datasets/baselines/TIMESTAMP_custom.json`.

---

## Phase 4 - Regression Dedection

Diffs two Ragas baseline files and flags metric drops. Exits with code 1 on regression.

```bash
python compare.py \
    -- baseline datasets/baselines/OLD_ragas.json \ 
    -- current datasets/baselines/NEW_ragas.json
```

Optional - override the allowed drop threshold (default is 0.05):
```bash
python compare.py --baseline OLD.json --current NEW.json --threshold 0.10
```


## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Dataset Foundation | Done |
| 2 | Ragas + Ollama Evaluator | Done |
| 3 | Custom Evaluator From Scratch | Done |
| 4 | Regression Detection | Done |
| 5 | CI Integration | Pending |
| 6 | Reporting | Pending |