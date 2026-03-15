import argparse
import json
import re
import sys
import uuid
from pathlib import Path

import httpx
import pdfplumber
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def exctract_pages(pdf_path: Path)->list[dict]:
    pages=[]
    with pdfplumber.open(pdf_path) as pdf:
        for i,page in enumerate(pdf.pages,start=1):
            text=page.extract_text() or ""

            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    row_text = " | ".join(str(cell or "").strip() for cell in row)
                    if row_text.strip():
                        text += "\n" + row_text

            text = text.strip()
            if text:
                pages.append({"page_number":i, "text":text})

    return pages


#to get rid of <think> tags in output of chain of thought in deepseek
def strip_think_tags(raw:str) -> str:
    cleaned=re.sub(r"<think>.*?</think>",raw,flags=re.DOTALL)
    return cleaned.strip()

def extract_json_array(text:str) -> str:
    start = text.find("[")
    end = text.find("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in model output:\n{text[:500]}")
    return text[start:end+1]


GENERATION_PROMPT = """\
You are an expert at creating evulation datasets for RAG (Retrieval-Augmented Generation) systems.

Given the document excerpt below, generate exactly {n} question-answer pairs.

Rules:
- Questions must be answerable ONLY from the provided excerpt - no outside knowledge
- Answers must be consise and factually grounded in the excerpt
- Vary the difficulty: include factual lookups, reasoning questions, and (if possible) questions that require connecting multiple pieces of information
- Each question must be distinct - no rephrasing of the same question
- Do NOT include about page numbers of document structure

Difficulty definitions:
- "factual": a single fact stated explicitly in the text
- "reasonnig": requires interpreting or combining information from the excerpt
- "multi_hop": requires connecting two or more seperate facts from the excerpt

Return ONLY a valid JSON array with no preamble, no explanations, no markdown fences.
Each item must have exactly these keys:
  "question","excepted_answer","difficulty","category"

Where "category" is a short topic label (e.g "specifications","safety","installation","overview").

Document excerpt (page {page_number}):
---
{text}
---

Json array:"""

def call_ollama(prompt:str, config:dict)->str:
    url=f"{config['ollama']['base_url']}/api/generate"
    payload={
        "model":config["ollama"]["judge_model"],
        "prompt":prompt,
        "stream":False,
        "options":{"num_predict":1024, "temperature":0.3}
    }

    with httpx.Client(timeout=config["ollama"]["timeout"]) as client:
        response = client.post(url,json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response","")
    
#for a single page
def generate_candidates_for_page(
        page:dict,
        config:dict,
        questions_per_page:int,
        source_document:str,
)->list[dict]:
    prompt = GENERATION_PROMPT.format(
        n=questions_per_page,
        page_number = page["page_number"],
        text = page["text"][:3000] # # cap to avoid huge prompts on dense pages    
    )
    raw = call_ollama(prompt,config)
    cleaned = strip_think_tags(raw)
    
    try:
        json_str = extract_json_array(cleaned)
        items = json.loads(json_str)
    except (ValueError,json.JSONDecodeError) as e:
        print(f"[!] Failed to parse JSON  for page {page['page_number']}: {e}")
        print(f" Raw output: {cleaned[:300]}")
        return []
    
    candidates = []
    for item in items:
        if not isinstance(item,dict):
            continue
        question = item.get("question","").strip()
        excepted_answer = item.get("excepted_answer","").strip()
        if not question or not excepted_answer:
            continue

        candidates.append({
            "id":f"case_{uuid.uuid4().hex[:8]}",
            "question":question,
            "excepted_answer":excepted_answer,
            "source_document":source_document,
            "page_number":page["page_number"],
            "difficulty":item.get("difficulty","factual"),
            "category":item.get("category","general"),
            "notes":"",
            "generated_by":config["ollama"]["judge_model"],
            "reviewed_by":None,
            "approved":False
        })

    return candidates

def load_existing_candidates(path: Path) ->list[dict]:
    if path.exists() and path.stat().st_size > 2:
        with open(path) as f:
            return json.load(f)
    return []


def save_candidates(candidates: list[dict], path: Path) -> None:
    with open(path,"w") as f:
        json.dump(candidates,f,indent=2,ensure_ascii=False)


def parse_range_range(pages_arg:str | None, total_pages:int) -> list[int]:
    if not pages_arg:
        return list(range(1,total_pages + 1))
    
    page_nums = []
    for part in pages_arg.split(","):
        part = part.strip()
        if "-" in part:
            start,end = part.split("-",1)
            page_nums.extend(range(int(start),int(end) + 1))
        else:
            page_nums.append(int(part))
    
    return sorted(set(page_nums))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Q&A candidate pairs from a PDF using Ollama"
    )
    parser.add_argument("--pdf",required=True,help="Path to the source PDF")
    parser.add_argument(
        "--pages",
        default=None,
        help="Page range to process, e.g. '1-10' or '1,3,5'. Defaults to all pages." 
    )
    parser.add_argument(
        "--question-per-page",
        type=int,
        default=None,
        help="Number of Q&A pairs per page. Overrides config.yaml."
        )
    parser.add_argument(
        "--append",
        action="store_true",
        default=True,
        help="Append to existing raw_candidates.json instead of overwriting (default: true)"
    )
    args = parser.parse_args()
    
    config = load_config()
    pdf_path = Path(args.pdf)
    candidates_path = Path(config["paths"]["raw_candidates"])
    questions_per_page = args.questions_per_page or config["generation"]["questions_per_page"]
    
    if not pdf_path.exists():
        print(f"[error] PDF not found: {pdf_path}")
        sys.exit()

    print(f"\n{'='*60}")
    print(f"Generator - {[pdf_path.name]}")
    print(f"Model: {config['ollama']['judge_model']}")
    print(f"Questions per pag: {questions_per_page}")
    print(f"\n{'='*60}")
    
    print("Exctracting pages from PDF...")
    pages = exctract_pages(pdf_path)
    print(f"Found {len(pages)} non-empty pages\n")

    if not pages:
        print(["[error] No text could be extracted from this PDF."])
        sys.exit(1)
    
    #filter to requested page range
    requested = parse_range_range(args.pages, max(p["page_number"] for p in pages))
    pages = [p for p in pages if p["page_number"] in requested]
    print(f"Processing {len(pages)} pages\n")

    #to load existing candidaes if appending
    all_candidates = load_existing_candidates(candidates_path) if args.append else []
    new_count = 0

    for page in pages:
        print(f"Page {page["page_number"]:>3} / {pages[-1]['page_number']} ... ",end="",flush=True)
        candidates = generate_candidates_for_page(
            page, config, questions_per_page, pdf_path.name
        )
        all_candidates.extend(candidates)
        new_count +=len(candidates)
        print(f"{len(candidates)} Q&A pairs generated")

    save_candidates(all_candidates,candidates_path)

    print(f"\n{'='*60}")
    print(f"Done. {new_count} new candidates added.")
    print(f"Total in {candidates_path}: {len(all_candidates)}")
    print(f"Next step: python -m dataset_builder.reviewer")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()