import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

METRIC_LABELS = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_without_reference": "context_precision",
    "context_precision": "context_precision",
    "context_recall": "context_recall",
}


def load_baseline(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def pick_latest_baseline(baselines_dir: Path) -> Path | None:
    candidates = sorted(baselines_dir.glob("*_ragas.json")) + sorted(
        baselines_dir.glob("*_custom.json")
    )
    if not candidates:
        return None
    return sorted(candidates)[-1]


def metric_threshold_for(metric: str, thresholds: dict) -> float:
    if metric == "llm_context_precision_without_reference":
        return thresholds.get("context_precision", 0.6)
    return thresholds.get(metric, 0.6)


def render_summary(console: Console, baseline: dict, source: Path) -> None:
    title = (
        f"Evaluation Report: {source.name} "
        f"({baseline.get('evaluator', 'unknown')} | {baseline.get('timestamp', 'n/a')})"
    )
    console.print(f"\n[bold]{title}[/bold]")

    averages = baseline.get("averages", {})
    thresholds = baseline.get("thresholds", {})

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Average", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Status", justify="center")

    for metric, value in averages.items():
        label = METRIC_LABELS.get(metric, metric)
        threshold = metric_threshold_for(metric, thresholds)
        if value is None:
            avg_str = "N/A"
            status = "[yellow]MISSING[/yellow]"
        else:
            avg_str = f"{value:.4f}"
            status = "[green]PASS[/green]" if value >= threshold else "[red]FAIL[/red]"
        table.add_row(label, avg_str, f"{threshold:.4f}", status)

    console.print(table)


def render_worst_cases(console: Console, baseline: dict, worst_limit: int) -> None:
    results = baseline.get("results", [])
    if not results:
        console.print("\n[yellow]No per-case results found in this baseline.[/yellow]")
        return

    metrics = list((baseline.get("averages") or {}).keys())
    for metric in metrics:
        label = METRIC_LABELS.get(metric, metric)
        scored = []
        for item in results:
            score = (item.get("scores") or {}).get(metric)
            if score is None:
                continue
            scored.append((float(score), item))

        if not scored:
            continue

        scored.sort(key=lambda x: x[0])
        worst = scored[:worst_limit]

        table = Table(
            show_header=True,
            header_style="bold",
            title=f"Worst {len(worst)} cases by {label}",
        )
        table.add_column("Case ID", style="magenta")
        table.add_column("Score", justify="right")
        table.add_column("Question", style="white")

        for score, item in worst:
            q = item.get("question", "")
            short_q = q if len(q) <= 80 else f"{q[:77]}..."
            table.add_row(str(item.get("id", "n/a")), f"{score:.4f}", short_q)
        console.print(table)


def run(config: dict, input_path: Path | None, worst_limit: int = 5) -> int:
    console = Console()
    baselines_dir = Path(config["paths"]["baselines_dir"])

    target = input_path if input_path else pick_latest_baseline(baselines_dir)
    if target is None:
        console.print("[red]No baseline files found for reporting.[/red]")
        return 1
    if not target.exists():
        console.print(f"[red]Baseline file not found: {target}[/red]")
        return 1

    try:
        baseline = load_baseline(target)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid JSON in {target}: {exc}[/red]")
        return 1

    render_summary(console, baseline, target)
    render_worst_cases(console, baseline, worst_limit)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Render console report for an eval baseline")
    parser.add_argument("--input", default=None, help="Path to baseline JSON file")
    parser.add_argument("--baselines-dir", default="datasets/baselines")
    parser.add_argument("--worst-limit", type=int, default=5)
    args = parser.parse_args()

    config = {"paths": {"baselines_dir": args.baselines_dir}}
    input_path = Path(args.input) if args.input else None
    raise SystemExit(run(config, input_path=input_path, worst_limit=args.worst_limit))


if __name__ == "__main__":
    main()
