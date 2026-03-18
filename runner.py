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
        print("Custom evaluator not built yet - coming in Phase 3")
        sys.exit(0)


if __name__ == "__main__":
    main()

