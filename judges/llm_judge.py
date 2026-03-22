import json
import re
from pathlib import Path

import httpx
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

_embed_model = None

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)
    

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embed_model


def cosine_similarity(a: list[float],b:list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def strip_think_tags(raw:str) -> str:
    cleaned = re.sub(r"<think>.*?</think>","",raw, flags = re.DOTALL)
    return cleaned.strip()


def extract_json(text:str) -> dict:
    text = strip_think_tags(text)
    text = re.sub(r"```json|```","",text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in output:\n{text[:300]}")
    return json.loads(text[start:end])


def call_ollama_sync(prompt:str, config:dict) -> str:
    url = f"{config['ollama']['base_url']}/api/generate"
    payload={
        "model":config["ollama"]["judge_model"],
        "prompt":prompt,
        "stream":False,
        "options":{"num_predict":512,"temperature":0.0}
    }
    with httpx.Client(timeout=config["ollama"]["timeout"]) as client:
        response = client.post(url,json=payload)
        response.raise_for_status()
        return response.json().get("response","")
    


#faithfulness metric

FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI answer is faithful to its source context.

Context:
{context}

Answer:
{answer}

Task:
1.  List every factual claims made in the answer
2.  For each claim, check if it is directly supported by the context above.
3.  Count how many claims are supported vs total claims.

Return ONLY a JSON object, no preamble, no explanation:
{{"supported": <int>, "total": <int>, "reason": "<one sentence>"}}"""


def score_faithfulness(answer:str, contexts:list[str],config:dict)-> float:
    context_text = "\n\n".join(contexts)
    prompt = FAITHFULNESS_PROMPT.format(context=context_text,answer=answer)

    try:
        raw = call_ollama_sync(prompt,config)
        data = extract_json(raw)
        supported = int(data.get("supported",0))
        total = int(data.get("total",1))
        if total==0:
            return 1.0
        return round(supported / total, 4)
    except Exception as e:
        print(f"[!] faithfulness scoring failed: {e}")
        return None

#answer relevancy metric 

QUESTION_GENERATION_PROMPT = """\
Given the following answer, generate exactly 3 different questions that this answer would be a good response to.

Answer:
{answer}


Return ONLY a JSON array of 3 question strings, no preamble, no explanation:
["question 1", "question 2", "question 3"]"""

def score_answer_relevancy(question:str, answer:str, config:dict)->float:
    #generating three questions from the answer
    prompt = QUESTION_GENERATION_PROMPT.format(answer=answer)

    try:
        raw = call_ollama_sync(prompt,config)
        cleaned = strip_think_tags(raw)
        cleaned  = re.sub(r"```json|```","",cleaned).strip()
        start = cleaned.find("[")
        end = cleaned.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON array found")
        generated_questions = json.loads(cleaned[start:end])

        if not generated_questions:
            return None

        #embedding original question and original questions
        model = get_embed_model()
        original_embedding = model.encode(question).tolist()
        generated_embeddings = [model.encode(q).tolist() for q in generated_questions]

        #average cosine similarity
        similarities = [
            cosine_similarity(original_embedding,gen_emb)
            for gen_emb in generated_embeddings
        ]
        return round(float(np.mean(similarities)),4)
    
    except Exception as e:
        print(f"[!] answer_relevancy scoring failed: {e}")
        return None


#context precision metric

CONTEXT_PRECISION_PROMPT = """\
You are evaluating whether a retrieved document chunk is relevant to answering a question.

Question:
{question}

Chunk:
{chunk}

Is this chunk relevant to answering the question above?
Return ONLY a JSON object:
{{"relevant": true/false, "reason":"<one sentence"}}"""


def score_context_precision(question: str, contexts: list[str], config:dict)->dict:
    if not contexts:
        return 0.0
    
    relevant_count = 0

    for chunk in contexts:
        prompt = CONTEXT_PRECISION_PROMPT.format(question=question, chunk=chunk)
        try:
            raw = call_ollama_sync(prompt,config)
            data = extract_json(raw)
            if data.get("relevant") is True:
                relevant_count +=1
        
        except Exception as e:
            print(f"[!] context_precision chunk scoring failed: {e}")
            continue
    
    return round(relevant_count / len(contexts), 4)

#context recall metric

CONTEXT_RECALL_PROMPT = """\
You are evaluating wheter retrieved document chunks contain enough information to answer a question correctly.

Expected answer:
{expected_answer}

Retrieved chunks:
{context}

Task:
1.  Break the expected answer into individual factual statements.
2.  For each statement, check if it is covered by the retrieved chunks.
3.  Count how many statements are covered vs total statements.

Return ONLY a JSON object, no preamble, no explanation:
{{"covered":<int>, "total":<int>, "reason": "<one sentence>"}}"""

def score_context_recall(
        contexts:list[str], expected_answer: str, config: dict
) -> float:
    context_text = "\n\n".join(contexts)
    prompt = CONTEXT_RECALL_PROMPT.format(
        expected_answer = expected_answer, context = context_text
    )

    try:
        raw = call_ollama_sync(prompt,config)
        data = extract_json(raw)
        covered = int(data.get("covered",0))
        total = int(data.get("total",1))
        if total == 0:
            return 1.0
        return round(covered / total, 4)
    except Exception as e:
        print(f"[!] contexT_recall scoring failed: {e}")
        return None
    

#main judge

def judge(
        question:str,
        answer:str,
        contexts:list[str],
        expected_answer: str,
        config: dict,
) -> dict:
    return {
        "faithfulness": score_faithfulness(answer,contexts,config),
        "answer_relevancy":score_answer_relevancy(question,answer,config),
        "context_precision":score_context_precision(question,contexts,config),
        "context_recall":score_context_recall(contexts,expected_answer,config),

    }