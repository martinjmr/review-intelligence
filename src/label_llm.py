"""Étape 2 : faire étiqueter des avis par un LLM (cours 1, sections 4 à 7).

Usage :  python -m src.label_llm --part explo --prompt v1
Produit : data/labels_<part>_<prompt>.jsonl (une ligne par avis, reprise automatique).
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel

MODEL = "mistral-small-2603"   # figé : l'alias mistral-small-latest peut changer de ciblePOOL = Path("data/pool.parquet")
MAX_ATTEMPTS = 5
PAUSE_S = 2.2            # mistral-small : 20 000 tokens/min max -> ~28 avis/min
RETRY_CODES = {429, 500, 502, 503, 504}

Motif = Literal["reception", "gout_texture", "fraicheur_securite", "non_conforme", "prix", "autre"]


class Etiquetage(BaseModel):
    evidence: str          # généré en 1er : le passage de l'avis qui justifie
    motif_libre: str       # généré en 2e : le problème en 3 à 6 mots
    motifs: list[Motif]    # généré en dernier : l'étiquette


def load_prompt(version):
    return Path(f"prompts/{version}.txt").read_text(encoding="utf-8")


def build_messages(system_prompt, title, text):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Title: {title}\nReview: {text}"},
    ]


def check_rules(motifs):
    """Renvoie la liste des règles violées (liste vide = valide)."""
    problems = []
    if not motifs:
        problems.append("liste_vide")
    if "autre" in motifs and len(set(motifs)) > 1:
        problems.append("autre_combine")
    if len(motifs) != len(set(motifs)):
        problems.append("doublon")
    return problems


def call_with_retry(fn, max_attempts=MAX_ATTEMPTS, sleep=time.sleep):
    """Appelle fn() ; en cas d'erreur réessayable, attend 1, 2, 4, 8 s (backoff exponentiel).
    Renvoie (résultat, nombre d'essais)."""
    for attempt in range(max_attempts):
        try:
            return fn(), attempt + 1
        except Exception as e:
            code = getattr(e, "status_code", None)
            retryable = code in RETRY_CODES or (
                code is None and not isinstance(e, (ValueError, TypeError, KeyError))
            )
            if not retryable or attempt == max_attempts - 1:
                raise
            sleep(2 ** attempt)


def already_done(out_path):
    """Identifiants déjà étiquetés avec succès (les lignes en erreur seront refaites)."""
    if not out_path.exists():
        return set()
    with out_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["review_id"] for r in rows if "error" not in r}


def label_one(client, system_prompt, row):
    """Un appel, une ligne de résultat (dict prêt à écrire en JSONL)."""
    messages = build_messages(system_prompt, row["title"], row["text"])
    t0 = time.perf_counter()
    rep, attempts = call_with_retry(lambda: client.chat.parse(
        model=MODEL,
        response_format=Etiquetage,
        temperature=0,
        max_tokens=200,
        random_seed=42,
        messages=messages,
    ))
    latency = time.perf_counter() - t0
    parsed = rep.choices[0].message.parsed
    motifs = parsed.motifs if parsed else []
    return {
        "review_id": row["review_id"],
        "model": rep.model,
        "evidence": parsed.evidence if parsed else None,
        "motif_libre": parsed.motif_libre if parsed else None,
        "motifs": motifs,
        "problems": check_rules(motifs),
        "prompt_tokens": rep.usage.prompt_tokens,
        "completion_tokens": rep.usage.completion_tokens,
        "latency_s": round(latency, 3),
        "attempts": attempts,
    }


def label_part(client, df, system_prompt, prompt_version, out_path, pause_s=PAUSE_S):
    done = already_done(out_path)
    todo = df[~df["review_id"].isin(done)]
    print(f"{len(done)} déjà faits, {len(todo)} à faire")
    with out_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(todo.to_dict("records"), 1):
            try:
                result = label_one(client, system_prompt, row)
            except Exception as e:                       # erreur définitive : on note et on continue
                result = {"review_id": row["review_id"], "error": repr(e)[:300]}
            result["prompt_version"] = prompt_version
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()                                     # écrit sur disque tout de suite
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}")
            time.sleep(pause_s)


def main():
    from dotenv import load_dotenv
    from mistralai.client import Mistral

    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="explo")
    parser.add_argument("--prompt", default="v1")
    parser.add_argument("--limit", type=int, default=None, help="n'étiqueter que les N premiers")
    args = parser.parse_args()

    load_dotenv()
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    df = pd.read_parquet(POOL)
    df = df[df["part"] == args.part]
    if args.limit:
        df = df.head(args.limit)
    out = Path(f"data/labels_{args.part}_{args.prompt}.jsonl")
    label_part(client, df, load_prompt(args.prompt), args.prompt, out)
    print("Écrit :", out)


if __name__ == "__main__":
    main()
