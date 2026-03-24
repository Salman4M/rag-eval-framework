import argparse
import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

METRIC_LOOKUP = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "context_precision": "llm_context_precision_without_reference",
    "context_recall": "context_recall",
}

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Eval Report</title>
  <style>
    :root {
      --bg: #f5f4ee;
      --card: #fffdf5;
      --ink: #1f2937;
      --muted: #4b5563;
      --line: #d6d3c4;
      --good: #166534;
      --bad: #991b1b;
      --accent: #075985;
    }
    body { margin: 0; background: radial-gradient(circle at 10% 10%, #ede9d5 0, #f5f4ee 45%, #f2efe4 100%); color: var(--ink); font-family: "Georgia", "Times New Roman", serif; }
    .container { max-width: 1080px; margin: 0 auto; padding: 28px 16px 40px; }
    .header { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 18px; box-shadow: 0 4px 18px rgba(0,0,0,.05); }
    h1, h2 { margin: 0 0 10px; }
    p { margin: 4px 0; color: var(--muted); }
    .card { margin-top: 18px; background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px; box-shadow: 0 4px 18px rgba(0,0,0,.05); }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { text-align: left; padding: 10px 8px; border-bottom: 1px solid #ebe7d6; }
    th { color: var(--accent); font-weight: 700; }
    .good { color: var(--good); font-weight: 700; }
    .bad { color: var(--bad); font-weight: 700; }
    .mono { font-family: "Courier New", monospace; }
    @media (max-width: 740px) {
      th, td { font-size: 12px; padding: 8px 4px; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>RAG Evaluation Report</h1>
      <p>Generated at: <span class="mono">{{ generated_at }}</span></p>
      <p>Total runs included: <strong>{{ runs|length }}</strong></p>
    </div>

    <div class="card">
      <h2>Metric Trends</h2>
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Evaluator</th>
            {% for metric in metrics %}
            <th>{{ metric }}</th>
            {% endfor %}
          </tr>
        </thead>
        <tbody>
          {% for run in runs %}
          <tr>
            <td class="mono">{{ run.name }}</td>
            <td>{{ run.evaluator }}</td>
            {% for metric in metrics %}
            {% set value = run.metric_values.get(metric) %}
            <td>{% if value is none %}<span class="mono">N/A</span>{% else %}<span class="mono">{{ "%.4f"|format(value) }}</span>{% endif %}</td>
            {% endfor %}
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>Latest Run Summary</h2>
      <p><span class="mono">{{ latest.name }}</span> ({{ latest.evaluator }})</p>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Average</th>
            <th>Threshold</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {% for row in latest_summary %}
          <tr>
            <td>{{ row.metric }}</td>
            <td class="mono">{{ row.average }}</td>
            <td class="mono">{{ row.threshold }}</td>
            <td class="{{ 'good' if row.status == 'PASS' else 'bad' }}">{{ row.status }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>Worst Performing Cases (Latest Run)</h2>
      {% for group in worst_cases %}
      <h3>{{ group.metric }}</h3>
      <table>
        <thead>
          <tr>
            <th>Case ID</th>
            <th>Score</th>
            <th>Question</th>
          </tr>
        </thead>
        <tbody>
          {% for row in group.rows %}
          <tr>
            <td class="mono">{{ row.case_id }}</td>
            <td class="mono">{{ row.score }}</td>
            <td>{{ row.question }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% endfor %}
    </div>
  </div>
</body>
</html>
"""


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_timestamp(value: str, fallback_name: str) -> datetime:
    if value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    try:
        stem = fallback_name.split("_")[0] + "_" + fallback_name.split("_")[1]
        return datetime.strptime(stem, "%Y%m%d_%H%M%S")
    except Exception:
        return datetime.min


def discover_run_files(baselines_dir: Path) -> list[Path]:
    files = []
    for path in sorted(baselines_dir.glob("*_ragas.json")):
        files.append(path)
    for path in sorted(baselines_dir.glob("*_custom.json")):
        files.append(path)
    for alias in ["reference_ragas.json", "latest_ragas.json"]:
        alias_path = baselines_dir / alias
        if alias_path.exists():
            files.append(alias_path)
    return files


def read_runs(baselines_dir: Path) -> list[dict]:
    runs = []
    for path in discover_run_files(baselines_dir):
        try:
            data = load_json(path)
        except json.JSONDecodeError:
            continue

        averages = data.get("averages", {})
        values = {
            "faithfulness": averages.get(METRIC_LOOKUP["faithfulness"]),
            "answer_relevancy": averages.get(METRIC_LOOKUP["answer_relevancy"]),
            "context_precision": averages.get(METRIC_LOOKUP["context_precision"]),
            "context_recall": averages.get(METRIC_LOOKUP["context_recall"]),
        }
        runs.append(
            {
                "name": path.name,
                "path": path,
                "evaluator": data.get("evaluator", "unknown"),
                "timestamp": parse_timestamp(data.get("timestamp", ""), path.name),
                "raw": data,
                "metric_values": values,
            }
        )
    runs.sort(key=lambda x: x["timestamp"])
    return runs


def latest_summary_rows(latest_raw: dict) -> list[dict]:
    averages = latest_raw.get("averages", {})
    thresholds = latest_raw.get("thresholds", {})
    rows = []
    for metric in METRICS:
        key = METRIC_LOOKUP[metric]
        avg = averages.get(key)
        threshold_key = "context_precision" if key == "llm_context_precision_without_reference" else key
        threshold = thresholds.get(threshold_key, 0.6)
        status = "FAIL"
        avg_display = "N/A"
        if avg is not None:
            avg_display = f"{float(avg):.4f}"
            status = "PASS" if float(avg) >= float(threshold) else "FAIL"
        rows.append(
            {
                "metric": metric,
                "average": avg_display,
                "threshold": f"{float(threshold):.4f}",
                "status": status,
            }
        )
    return rows


def worst_case_groups(latest_raw: dict, limit: int = 5) -> list[dict]:
    groups = []
    results = latest_raw.get("results", [])
    for metric in METRICS:
        key = METRIC_LOOKUP[metric]
        scored = []
        for item in results:
            score = (item.get("scores") or {}).get(key)
            if score is None:
                continue
            q = item.get("question", "")
            scored.append(
                {
                    "case_id": str(item.get("id", "n/a")),
                    "score_value": float(score),
                    "score": f"{float(score):.4f}",
                    "question": q if len(q) <= 110 else f"{q[:107]}...",
                }
            )
        scored.sort(key=lambda x: x["score_value"])
        rows = scored[:limit]
        for row in rows:
            row.pop("score_value", None)
        groups.append({"metric": metric, "rows": rows})
    return groups


def generate_report(config: dict, output_path: Path | None = None) -> Path:
    baselines_dir = Path(config["paths"]["baselines_dir"])
    runs = read_runs(baselines_dir)
    if not runs:
        raise RuntimeError("No valid baseline files found for HTML reporting.")

    latest = runs[-1]
    latest_summary = latest_summary_rows(latest["raw"])
    worst_cases = worst_case_groups(latest["raw"])

    env = Environment(autoescape=True, trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(HTML_TEMPLATE)
    html = template.render(
        generated_at=datetime.utcnow().isoformat() + "Z",
        runs=runs,
        latest=latest,
        latest_summary=latest_summary,
        worst_cases=worst_cases,
        metrics=METRICS,
    )

    path = output_path or Path("reports/eval_report.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def run(config: dict, output_path: Path | None = None) -> int:
    try:
        path = generate_report(config, output_path=output_path)
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 1
    print(f"HTML report generated: {path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML report from eval baselines")
    parser.add_argument("--baselines-dir", default="datasets/baselines")
    parser.add_argument("--output", default="reports/eval_report.html")
    args = parser.parse_args()
    config = {"paths": {"baselines_dir": args.baselines_dir}}
    raise SystemExit(run(config, output_path=Path(args.output)))


if __name__ == "__main__":
    main()
