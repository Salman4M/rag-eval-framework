import argparse
import json
import sys
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config()->dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)
    

def load_baseline(path:Path) -> dict:
    with open(path) as f:
        return json.load(f)
    

def get_metric_names(baseline:dict) ->list[str]:
    return list(baseline.get("averages",{}).keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Eval regression detector")
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to lder baseline JSON (the reference)"
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to newer baseline JSON (the one being tested)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Maximum allowed drop pet metric before flagging regression {default: 0.05}"
    )
    args = parser.parse_args()

    config = load_config()
    baseline_path = Path(args.baseline)
    current_path = Path(args.current)

    if not baseline_path.exists() or not current_path.exists():
        print(f"[error] Current or Baseline file is not found \nBaseline path: {baseline_path} \nCurrent path: {current_path}")
        sys.exit(1)
    
    baseline = load_baseline(baseline_path)
    current = load_baseline(current_path)

    thresholds = config.get("thresholds",{})
    regression_threshold = args.threshold

    print(f"\n{'='*70}")
    print(f"REGRESSION DETECTION")
    print(f"\n{'='*70}")
    print(f"Baseline: {baseline_path.name}")
    print(f"Current: {current_path.name}")
    print(f"Max allowed drop per metric: {regression_threshold}")
    print(f"\n{'='*70}")

    print(f"  {'metric':<45} {'baseline':>9} {'current':>9} {'delta':>9} {'status':>10}")
    print(f"  {'-'*45} {'-'*9} {'-'*9} {'-'*9} {'-'*10}")

    metrics = get_metric_names(baseline)
    regressions = []

    for metric in metrics:
        baseline_score = baseline["averages"].get(metric)
        current_score = current["averages"].get(metric)

        if baseline_score is None or current_score is None:
            print(f"  {'?'} {metric:<45} {'N/A':>9} {'N/A':>9} {'N/A':>9} {'SKIP':>10}")
            continue

        delta = round(current_score - baseline_score, 4)
        delta_str = f"{delta:+.4f}"
        baseline_str = f"{baseline_score:.4f}"
        current_str = f"{current_score:.4f}"

        min_threshold = thresholds.get(metric.replace("llm_context_precision_without_reference","context_precision"),0.6)
        is_regression = delta < -regression_threshold
        is_below_min = current_score < min_threshold

        if is_regression:
            status = "REGRESSION"
            regressions.append({
                "metric":metric,
                "baseline":baseline_score,
                "current":current_score,
                "delta":delta
            })
        
        elif is_below_min:
            status = "BELOW MIN"
            regressions.append({
                "metric":metric,
                "baseline":baseline_score,
                "current":current_score,
                "delta":delta
            })

        elif delta >=0:
            status = "OK"
        else:
            status = "WARN"
        
        print(f"{metric:<45} {baseline_str:>9} {current_str:>9} {delta_str:>9} {status:>10}")

    print(f"\n{'='*70}")


    if regressions:
        print(f"\n FAILED - {len(regressions)} regressions(s) detected: \n")
        for r in regressions:
            print(f"x {r['metric']}")
            print(f"baseline: {r['baseline']:.4f}")
            print(f"current: {r['current']:.4f}")
            print(f"delta: {r['delta']:+.4f}")
        print(f"{'='*70}\n")
        sys.exit(1)
    else:
        print(f"\n PASSED - no regression detected")
        print(f"{'='*70}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()






