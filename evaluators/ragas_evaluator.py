import asyncio
import json
import os
from datetime import datetime,timezone
from pathlib import Path


import time
import httpx
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import ResponseRelevancy
from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference
from ragas.metrics._context_recall import ContextRecall
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config()->dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)
    

def load_eval_dataset(path:Path)->list[dict]:
    with open(path) as f:
        data = json.load(f)
    cases = [c for c in data.get("cases",[]) if c.get("approved")]
    print(f"Loaded {len(cases)} approved test cases")
    return cases

def ask_rag_api(question:str,config:dict,token:str)->dict | None:
    url = f"{config['api']['base_url']}/ask"
    headers= {"Authorization":f"Bearer {token}"}
    payload = {"question":question}

    try:
        with httpx.Client(timeout=config["api"]["timeout"])as client:
            response=client.post(url,json=payload,headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        print(f"[!] API error {e.response.status_code} for question: {question[:60]}")
        return None
    except Exception as e:
        print(f"[!] Request failed: {e}")
        return None


def build_ollama_llm(config:dict):
    client=AsyncOpenAI(
        base_url=f"{config['ollama']['base_url']}/v1",
        api_key="ollama"
    )
    return llm_factory(config['ollama']['judge_model'],client=client)


def build_ollama_embeddings(config:dict):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return LangchainEmbeddingsWrapper(embeddings=embeddings)



async def score_sample(sample:SingleTurnSample,metrics: list) ->dict:
    scores = {}
    for metric in metrics:
        try:
            result = await metric.single_turn_ascore(sample)
            scores[metric.name] = round(float(result),4)
        except Exception as e:
            print(f"[!] Metric {metric.name} failed: {e}")
            scores[metric.name] = None
    return scores



def collect(config:dict, token:str)-> None:
    eval_path = Path(config["paths"]["eval_dataset"])
    baselines_dir = Path(config["paths"]["baselines_dir"])
    baselines_dir.mkdir(parents=True,exist_ok=True)

    cases = load_eval_dataset(eval_path)
    if not cases:
        print("[error] No approved cases found in eval_dataset.json")
        return
    
    print(f"\n{'='*60}")
    print(f"  STAGE 1 — Collecting RAG API answers")
    print(f"  Make sure ONLY the RAG API is running. Stop Ollama if needed.")
    print(f"{'='*60}\n")

    collected = []
    failed = 0 

   
    for i,case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] {case['question'][:70]}...")
        api_response = ask_rag_api(case["question"],config, token)
        if not api_response:
            print(f"[!] Skipping - API call failed")
            failed+=1
            time.sleep(30)
            continue

        answer = api_response.get("answer","")
        sources = api_response.get("sources",[])
        contexts = [s["text"]for s in sources if s.get("text")]

        if not answer or not contexts:
            print(f"[!] Skipping - empty answer or no contexts returned")
            failed+=1
            continue

        collected.append({
            "id":case["id"],
            "question":case["question"],
            "expected_answer":case["expected_answer"],
            "actual_answer":answer,
            "source_document":case["source_document"],
            "page_number":case["page_number"],
            "difficulty":case["difficulty"],
            "category":case["category"],
            "contexts":contexts,
        })
        print(f"completed collected")
        time.sleep(30)


    timestamp =  datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    staging_path = baselines_dir / f"{timestamp}_collected.json"
    with open(staging_path,"w") as f:
        json.dump(collected,f,indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Collected : {len(collected)}/{len(cases)}")
    print(f"  Failed    : {failed}")
    print(f"  Saved to  : {staging_path}")
    print(f"\n  Next step:")
    print(f"  1. Stop the RAG API")
    print(f"  2. Run: python runner.py --eval ragas --score {staging_path}")
    print(f"{'='*60}\n")


async def score(config:dict, collected_path:Path)->None:
    baselines_dir = Path(config["paths"]["baselines_dir"])

    with open(collected_path) as f:
        collected = json.load(f)
    
    if not collected:
        print("[error] No collected answers found in file.")
        return
    
    print(f"\n{'='*60}")
    print(f"  STAGE 2 — Scoring with Ragas")
    print(f"  Make sure RAG API is STOPPED. Only Ollama needs to run.")
    print(f"{'='*60}\n")
 
    print(f"  Setting up Ragas with Ollama ({config['ollama']['judge_model']})...")

    llm = build_ollama_llm(config)
    embeddings = build_ollama_embeddings(config)

    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(embeddings=embeddings,llm=llm),
        LLMContextPrecisionWithoutReference(llm=llm),
        ContextRecall(llm=llm)
    ]

    print(f"Metrics: {[m.name for m in metrics]}\n")
    print(f"{'='*60}")

    results = []
    passed = 0
    failed = 0

    for i,item in enumerate(collected):
        print(f"[{i+1}/{len(collected)}] {item['question'][:70]}")
        sample = SingleTurnSample(
            user_input=item["question"],
            response=item["actual_answer"],
            retrieved_contexts=item["contexts"],
            reference=item["expected_answer"]
        )

        scores = await score_sample(sample,metrics)
        print(f"scores: {scores}")

        results.append({**item,"scores":scores})
        passed+=1
    
    averages={}
    for metric in metrics:
        values = [r["scores"][metric.name] for r in results if r["scores"].get(metric.name) is not None]
        averages[metric.name] = round(sum(values) / len(values),4) if values else None

    thresholds = config.get("thresholds",{}) 
    thresholds_map = {
        "faithfulness":thresholds.get("faithfulness",0.7),
        "answer_relevancy":thresholds.get("answer_relevancy",0.7),
        "llm_context_precision_without_reference":thresholds.get("context_precision",0.6),
        "context_recall":thresholds.get("context_recall",0.6),
    }
    passed_thresholds = {
        k: (v >= thresholds_map.get(k,0) if v is not None else False)
        for k,v in averages.items()
    }

    timestamp =  datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = {
        "evaluator":"ragas",
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

    output_path = baselines_dir / f"{timestamp}_ragas.json"
    with open(output_path,"w") as f:
        json.dump(output,f,indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"RAGAS EVALUATION SUMMARY")
    print(f"\n{'='*60}")
    print(f"Cases evaluated : {passed}/{len(collected)}")
    print(f"Failed          : {failed}")

    for metric_name,avg in averages.items():
        threshold = thresholds_map.get(metric_name,0)
        status = "PASSED" if passed_thresholds.get(metric_name) else "DENIED"
        print(f"{status} {metric_name:<45} {avg if avg is not None else 'N/A'} (threshold: {threshold})")

    print()
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")



#####################

