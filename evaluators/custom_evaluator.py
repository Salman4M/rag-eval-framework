import json
from datetime import datetime,timezone
from pathlib import Path
from judges.llm_judge import judge


def score(config:dict, collected_path:Path) -> Path:
    baselines_dir = Path(config["paths"]["baselines_dir"])

    with open(collected_path) as f:
        collected = json.load(f)

    if not collected:
        print("[error] No collected answers found.")
        return None
    
    print(f"\n{'='*60}")
    print(f"PHASE 3 - Custom Evaluator")
    print(f"Judge model:{config['ollama']['judge_model']}")
    print(f"Metrics: faithfulness, answer_relevancy,context_precision, context_recall")
    print(f"\n{'='*60}")

    results = []
    passed = 0
    failed = 0

    for i,item in enumerate(collected):
        print(f"[{i+1}/{len(collected)}] {item['question'][:70]}")

        scores = judge(
            question=item["question"],
            answer=item["answer"],
            contexts=item["contexts"],
            expected_answer=item["expected_answer"],
            config=config
        )
        print(f"scores: {scores}")

        results.append({**item,"scores":scores})
        passed+=1
    
    #averages

    metric_names = ["faithfulness", "answer_relevancy","context_precision", "context_recall"]
    averages = {}
    for metric in metric_names:
        values = [
            r["scores"][metric]
            for r in results
            if r["scores"].get(metric) is not None
        ]
        averages[metric] = round(sum(values) / len(values),4) if values else None

    thresholds = config.get("thresholds",{})
    thresholds_map = {
        "faithfulness":thresholds.get("faithfulness",0.7),
        "answer_relevancy":thresholds.get("answer_relevancy",0.7),
        "context_precision":thresholds.get("context_precision",0.6),
        "context_recall":thresholds.get("context_recall",0.6)
    }
    
    passed_thresholds = {
        k: (v >= thresholds_map.get(k,0) if v is not None else False)
        for k,v in averages.items()
    }

    timestamp = datetime.now(timezone.ufc).strftime("%Y%m%d_%H%M%S")
    output = {
        "evaluator":"custom",
        "timestamp":datetime.now(timezone.utc).isoformat(),
        "judge_model":config["ollama"]["judge_model"],
        "total_cases":len(collected),
        "passed":passed,
        "failed":failed,
        "averages":averages,
        "thresholds":thresholds_map,
        "passed_thresholds":passed_thresholds,
        "results":results
    }

    output_path = baselines_dir / f"{timestamp}_custom.json"
    with open(output_path,"w") as f:
        json.dump(output,f,indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"RAGAS EVALUATION SUMMARY")
    print(f"\n{'='*60}")
    print(f"Cases evaluated : {passed}/{len(collected)}")
    print(f"Failed          : {failed}")

    for metric_name,avg in averages.items():
        threshold = thresholds_map.get(metric,0)
        status = "PASSED" if passed_thresholds.get(metric) else "DENIED"
        print(f"{status} {metric_name:<45} {avg if avg is not None else 'N/A'} (threshold: {threshold})")

    print()
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    return output_path



# to compare custom against ragas basline side by side
def compare(custom_path: Path, ragas_path: Path) -> None:
    with open(custom_path) as f:
        custom = json.load(f)
    with open(ragas_path) as f:
        ragas = json.load(f)

    print(f"{'='*60}")
    print(f"Comparison - Custom and Ragas")
    print(f"{'='*60}")
    print(f"{'metric':<45} {'custom':>8} {'ragas':>8} {'delta':>8}")
    print(f"{'-'*45} {'-'*8} {'-'*8} {'-'*8}")

    metrics = ["faithfulness", "answer_relevancy","context_precision", "context_recall"]
    ragas_metric_map = {
        "context_precision":"llm_context_precision_without_reference",
        "answer_relevancy":"response_relevancy"
    }

    for metric in metrics:
        custom_avg = custom["averages"].get(metric)
        ragas_key = ragas_metric_map.get(metric,metric)
        ragas_avg = ragas["averages"].get(ragas_key)

        if custom_avg is not None and ragas_avg is not None:
            delta = round(custom_avg-ragas_avg,4)
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "N/A"

    custom_str = f"{custom_avg:.4f}" if not custom_avg is not None else "N/A"
    ragas_str = f"{ragas_avg: 4f}" if ragas_avg is not None else "N/A"

    print(f" {metric:<45} {custom_str:>8} {ragas_str} {delta_str:>8}")

print(f"{'='*60}")
    



