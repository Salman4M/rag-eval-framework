import argparse
import asyncio
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)
    
def get_token(config: dict) ->str:
    token = os.getenv("RAG_API_TOKEN") or config.get("api",{}).get("token","")
    if not token:
        print("[error] No API token found. Set RAG_API_TOKEN in .env or config.yaml")
        sys.exit(1)
    return token

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Eval Framework runner")
    parser.add_argument(
        "--eval",
        choices=["ragas","custom"],
        required=True,
        help="Which evalutator to run"
    )
    parser.add_argument(
        "--score",
        default=None,
        help="Path to a collected answers JSON file (Stage 2 only). Skips RAG API calls."
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Path to Ragss baseline JSON to compare custom scores against"
    )

    args = parser.parse_args()

    config = load_config()

    if args.eval == "ragas":
        from evaluators.ragas_evaluator import collect,score
        if args.score:
            collected_path = Path(args.score)
            if not collected_path.exists():
                print(f"[error] File not found: {collected_path}")
                sys.exit(1)
            asyncio.run(score(config,collected_path))
        else:
            token = get_token(config)
            collect(config,token)

    elif args.eval == "custom":
        from evaluators.custom_evaluator import score, compare

        if not args.score:
            print(f"[error] --score is required for custom scores against")
            print("Usage: python runner.py --eval custom --score datasets/baselines/TIMESTAMP_collected.json")
        sys.exit(1)

        collected_path = Path(args.score)
        if not collected_path.exists():
            print(f"[error] File not found: {collected_path}")
            sys.exit(1)
        
        output_path = score(config, collected_path)

        if args.compare and output_path:
            ragas_path  = Path(args.compare)
            if not ragas_path.exists():
                print(f"[error] Ragas baseline not found: {ragas_path}")
                sys.exit(1)
            compare(output_path,ragas_path)


if __name__ == "__main__":
    main()

