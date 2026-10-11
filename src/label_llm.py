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

MODEL = "mistral-small-latest"
POOL = Path("data/pool.parquet")
MAX_ATTEMPTS = 5
PAUSE_S = 2.2            # mistral-small : 20 000 tokens/min max -> ~28 avis/min
RETRY_CODES = {429, 500, 502, 503, 504}

Motif = Literal["livraison", "emballage", "qualite", "prix", "service", "autre"]


class Etiquetage(BaseModel):
    # TODO : trois champs, DANS CET ORDRE (cours 1, section 7) :
    #   evidence (str), motif_libre (str), motifs (liste de Motif).
    # Pourquoi cet ordre ? Relis la règle 3 de la section 6.
    pass


def load_prompt(version):
    return Path(f"prompts/{version}.txt").read_text(encoding="utf-8")


def build_messages(system_prompt, title, text):
    # TODO : renvoie la liste de deux messages (rôles "system" puis "user").
    # Le message user contient le titre et le texte, par ex. "Title: ...\nReview: ...".
    raise NotImplementedError


def check_rules(motifs):
    """Renvoie la liste des règles violées (liste vide = valide)."""
    # TODO : construis une liste `problems` et ajoute :
    #   "liste_vide" si motifs est vide,
    #   "autre_combine" si "autre" est présent avec un autre motif,
    #   "doublon" si un motif apparaît deux fois.
    raise NotImplementedError


def call_with_retry(fn, max_attempts=MAX_ATTEMPTS, sleep=time.sleep):
    """Appelle fn() ; en cas d'erreur réessayable, attend 1, 2, 4, 8 s (backoff exponentiel)."""
    # TODO : boucle sur max_attempts essais.
    #   - try : return fn()
    #   - except Exception as e :
    #       code = getattr(e, "status_code", None)
    #       l'erreur est réessayable si code est dans RETRY_CODES,
    #       ou si code est None ET que e n'est pas un ValueError/TypeError/KeyError (= problème réseau).
    #       Si elle n'est pas réessayable, ou si c'était le dernier essai : `raise` (relance l'erreur).
    #       Sinon : sleep(2 ** attempt)  -> attend 1, 2, 4, 8 s.
    # Pourquoi `sleep` est un paramètre ? Pour que les tests vérifient les attentes sans attendre vraiment.
    raise NotImplementedError


def already_done(out_path):
    """Identifiants déjà présents dans le fichier de sortie (pour la reprise)."""
    # TODO : si out_path n'existe pas -> set() vide.
    # Sinon : ouvrir le fichier (encoding="utf-8"), et renvoyer l'ensemble des "review_id"
    # de chaque ligne non vide (json.loads(ligne)["review_id"]).
    raise NotImplementedError


def label_one(client, system_prompt, row):
    """Un appel, une ligne de résultat (dict prêt à écrire en JSONL)."""
    # TODO :
    #  1. t0 = time.perf_counter()
    #  2. rep = call_with_retry(lambda: client.chat.parse(...)) avec :
    #       model=MODEL, response_format=Etiquetage, temperature=0, max_tokens=200,
    #       random_seed=42, messages=build_messages(...)
    #  3. latence = time.perf_counter() - t0
    #  4. parsed = rep.choices[0].message.parsed   (peut être None si la réponse est vide)
    #  5. renvoyer un dict avec les clés :
    #       review_id, model, evidence, motif_libre, motifs, problems (check_rules),
    #       prompt_tokens, completion_tokens (dans rep.usage), latency_s (arrondie à 3 décimales)
    raise NotImplementedError


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
