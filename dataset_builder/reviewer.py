import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime,timezone
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
PROGRESS_FILE = Path(__file__).parent.parent / "datasets" / ".reviewer_progress.json"

console = Console()

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)
    
def load_json(path: Path, default):
    if path.exists() and path.stat().st_size > 2:
        with open(path) as f:
            return json.load(f)
    return default

def save_json(data, path:Path)-> None:
    with open(path,"w") as f:
        json.dump(data,f,indent=2,ensure_ascii=False)

def load_progress()->dict:
    return load_json(PROGRESS_FILE, {"reviewed_ids":[]})

def save_progress(progress:dict) -> None:
    save_json(progress, PROGRESS_FILE)

def load_eval_dataset(path:Path)->dict:
    data = load_json(path, None)
    if data is None:
        return {
            "version":"1.0",
            "created_at":datetime.now(timezone.utc).isoformat(),
            "cases":[]
        }
    return data

def save_eval_dataset(dataset:dict, path:Path) -> None:
    save_json(dataset, path)


def edit_in_editor(candidate:dict) -> dict | None:
    editor = os.environ.get("EDITOR","nano")
    with tempfile.NamedTemporaryFile(
        mode = "w", suffix=".json",delete=False, encoding="utf-8"
    ) as f:
        json.dump(candidate,f,indent=2,ensure_ascii=False)
        tmp_path = f.name

    try:
        result = subprocess.run([editor,tmp_path])
        if result.returncode != 0:
            console.print("[yellow]Editor exited with non-zero code. Skipping edit.[/yellow]")
            return None

        with open(tmp_path,encoding="utf-8") as f:
            edited = json.load(f)
        return edited
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON after editing: {e}[/red]")
        return None
    finally:
        os.unlink(tmp_path)


def display_candidate(candidate: dict, index: int, total:int, approved_count:int)-> None:
    difficulty_colors = {
        "factual":"cyan",
        "reasoning":"yellow",
        "multi_hop":"magenta"
    }
    diff = candidate.get("difficulty","factual")
    diff_color = difficulty_colors.get(diff,"white")

    header = (
        f"[bold]{index + 1}/{total}[/bold]  "
        f"[{diff_color}]{diff.upper()}[/{diff_color}]  "
        f"[dim]{candidate.get('source_document','')} · page {candidate.get('page_number','?')}[/dim]  "
        f"[green]✓ {approved_count} approved[/green]"
    )
    body = Text()
    body.append("Q: ", style="bold blue")
    body.append(candidate.get("question",""),style="white")
    body.append("\n\n")
    body.append("A:", style="bold green")
    body.append(candidate.get("expected_answer",""),style="dim white")

    if candidate.get("category"):
        body.append(f"\n\n[dim]category: {candidate['category']}[/dim]")

    console.print(Panel(body,title=header, border_style="dim"))


def display_controls() -> None:
    console.print(
        "  [bold cyan](a)[/bold cyan]pprove  "
        "[bold yellow](e)[/bold yellow]dit  "
        "[bold red](r)[/bold red]eject  "
        "[dim](s)[/dim]kip  "
        "[dim](q)[/dim]uit"
    )

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Interactive Q&A candidate reviewer")
    parser.add_argument(
        "--candidates",
        default=None,
        help="Path to raw_candidates.json.Defaults to config path"
    )
    args = parser.parse_args()

    config = load_config()
    candidates_path = Path(args.candidates or config["paths"]["raw_candidates"])
    eval_path = Path(config["paths"]["eval_dataset"])
    rejected_path = Path(config["paths"]["rejected"])

    candidates: list[dict] = load_json(candidates_path,[])
    if not candidates:
        console.print("[red]No candidates found. Run generator.py first.[/red]")
        sys.exit(1)

    
    progress = load_progress()
    reviewed_ids: set = set(progress.get("reviewed_ids",[]))

    dataset = load_eval_dataset(eval_path)
    rejected: list[dict] = load_json(rejected_path,[])

    #filter to unreviewed candidates
    pending = [c for c in candidates if c.get("id") not in reviewed_ids]

    if not pending:
        console.print("[green]All candidates have been reviewed:[/green]")
        console.print(f"Approved: {len(dataset['cases'])}")
        console.print(f"Rejected: {len(rejected)}")
        sys.exit(0)

    console.print(f"\n[bold]RAG Eval - Candidate Reviewer[/bold]")
    console.print(f"  {len(pending)} candidates to review | {len(dataset['cases'])} already approved\n")

    approved_count = len(dataset["cases"])

    for i,candidate in enumerate(pending):
        console.clear()
        display_candidate(candidate,i,len(pending),approved_count)
        display_controls()

        while True:
            choice = Prompt.ask(" >", default = "s").strip().lower()

            if choice =="a":
                candidate["reviewed_by"] = "human"
                candidate["approved"] = True
                dataset["cases"].append(candidate)
                reviewed_ids.add(candidate["id"])
                approved_count +=1
                save_eval_dataset(dataset,eval_path)
                save_progress({"reviewed_ids": list(reviewed_ids)})
                console.print(" [green]✓ Approved[/green]")
                break

            elif choice =="e":
                edited = edit_in_editor(candidate)
                if edited:
                    edited["id"] = candidate["id"]
                    edited["reviewed_by"] = "human"
                    edited["approved"] = True
                    dataset["cases"].append(edited)
                    reviewed_ids.add(candidate["id"])
                    approved_count +=1
                    save_eval_dataset(dataset,eval_path)
                    save_progress({"reviewed_ids": list(reviewed_ids)})
                    console.print(" [green]✓ Edited and Approved[/green]")
                else:
                    console.print("  [yellow]Edit cancelled. Skipping.[/yellow]")
                break

            elif choice == "r":
                candidate["reviewed_by"] = "human"
                candidate["approved"] = False
                rejected.append(candidate)
                reviewed_ids.add(candidate["id"])
                save_json(rejected,rejected_path)
                save_progress({"reviewed_ids": list(reviewed_ids)})
                console.print(" [red]✓ Rejected[/red]")
                break

            elif choice == "s":
                console.print("  [dim]Skipped[/dim]")
                break
            
            elif choice == "q":
                save_progress({"reviewed_ids": list(reviewed_ids)})
                console.print(f"\n[bold]Session saved.[/bold]")
                console.print(f"  Approved this session: {approved_count - len(load_eval_dataset(eval_path)['cases']) + approved_count}")
                console.print(f"  Total approved: {len(dataset['cases'])}")
                sys.exit(0)
            else:
                console.print("  [dim]Invalid choise. Use a / e / r / s / q[/dim]")

    save_progress({"reviewed_ids":list(reviewed_ids)})
    console.print(f"\n[bold green]All candidates reviewed![/bold green]")
    console.print(f"  Total approved: {len(dataset['cases'])}")
    console.print(f"  Total rejected: {len(rejected)}")
    console.print(f"\n  evaldataset.json is ready at: {eval_path}")



if __name__ == "__main__":
    main()

