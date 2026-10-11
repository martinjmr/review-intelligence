"""Étape 3 : analyser les étiquettes et décider si la liste de motifs tient (cours 1, section 9).

Usage :  python -m src.report data/labels_explo_v1.jsonl
"""
import argparse
import json
from collections import Counter

import pandas as pd

PRICE_IN = 0.15 / 1e6     # $ par token d'entrée  (à vérifier dans la console, noter la date)
PRICE_OUT = 0.60 / 1e6    # $ par token de sortie
MIN_SHARE = 0.05          # chaque motif doit couvrir au moins 5 % des avis
MAX_AUTRE = 0.25          # et "autre" au plus 25 %


def load_labels(path):
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def motif_shares(df):
    """Part des avis valides qui portent chaque motif (un avis peut en porter plusieurs)."""
    labeled = df[df["motifs"].map(len) > 0]
    n = len(labeled)
    counts = Counter()
    for m in labeled["motifs"]:
        counts.update(set(m))          # chaque motif compte une fois par avis
    return {motif: round(c / n, 3) for motif, c in counts.most_common()}

def cost_and_latency(df, n_target=100_000):
    ok = df.dropna(subset=["prompt_tokens"])
    cost = ok["prompt_tokens"].sum() * PRICE_IN + ok["completion_tokens"].sum() * PRICE_OUT
    return {
        "tokens_entree_moyen": round(float(ok["prompt_tokens"].mean()), 1),
        "tokens_sortie_moyen": round(float(ok["completion_tokens"].mean()), 1),
        "cout_total_$": round(float(cost), 4),
        f"cout_pour_{n_target}_avis_$": round(float(cost / len(ok) * n_target), 2),
        "latence_mediane_s": round(float(ok["latency_s"].median()), 2),
        "latence_p95_s": round(float(ok["latency_s"].quantile(0.95)), 2),
    }


def verdict(shares, expected=("livraison", "emballage", "qualite", "prix", "service")):
    """Liste des alertes ; liste vide = la liste de motifs tient."""
    alerts = []
    for motif in expected:
        if shares.get(motif, 0) < MIN_SHARE:
            alerts.append(f"{motif} < 5%")
    if shares.get("autre", 0) > MAX_AUTRE:
        alerts.append("autre > 25%")
    return alerts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    df = load_labels(args.path)
    if "error" in df:
        err = df["error"].notna()
        n_err = len(set(df.loc[err, "review_id"]) - set(df.loc[~err, "review_id"]))
        df = df[~err]
    else:
        n_err = 0
    df["motifs"] = df["motifs"].map(lambda m: m if isinstance(m, list) else [])
    print(f"{len(df)} avis étiquetés, {n_err} erreurs d'appel")
    print("Invalides :", round(float((df["problems"].map(len) > 0).mean()), 3))
    shares = motif_shares(df)
    print("Parts des motifs :", shares)
    print("Coût / latence :", cost_and_latency(df))
    alerts = verdict(shares)
    print("VERDICT :", "la liste tient" if not alerts else f"à revoir -> {alerts}")
    print("\nMotifs libres des avis 'autre' (pour trouver les motifs manquants) :")
    autres = df[df["motifs"].map(lambda m: m == ["autre"])]
    for m in autres["motif_libre"].head(40):
        print(" -", m)


if __name__ == "__main__":
    main()
