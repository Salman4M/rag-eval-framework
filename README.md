# rag-eval-framework

Evaluation framework for [rag-document-assistant](../rag-document-asistant).

## Phase 1 — Dataset Builder

### Setup
```bash
cd rag-eval-framework
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

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

### Target
30–50 approved cases before moving to Phase 2 (Ragas evaluation).

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Dataset Foundation | In progress |
| 2 | Ragas + Ollama Evaluator | Pending |
| 3 | Custom Evaluator From Scratch | Pending |
| 4 | Regression Detection | Pending |
| 5 | CI Integration | Pending |
| 6 | Reporting | Pending |