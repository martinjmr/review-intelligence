"""Étape 1 : construire le réservoir d'avis et le découper (cours 1, section 8).

Usage :  python -m src.data --n-read 500000
Produit : data/pool.parquet (20 000 avis, avec une colonne `part`).
"""
import argparse
import html
import re
from pathlib import Path

import pandas as pd

DATA_FILE = (
    "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "raw/review_categories/Grocery_and_Gourmet_Food.jsonl"
)
OUT = Path("data/pool.parquet")
PARTS = {"explo": 200, "test": 300, "train": 5000}   # le reste -> "reserve"


def stream_reviews(n_read):
    """Lit les n_read premiers avis du fichier, en streaming (fourni, rien à coder)."""
    from datasets import load_dataset

    flux = load_dataset("json", data_files=DATA_FILE, split="train", streaming=True)
    return pd.DataFrame(list(flux.take(n_read)))


def make_review_id(row):
    """Identifiant stable d'un avis (le jeu n'en fournit pas)."""
    return f"{row['user_id']}_{row['parent_asin']}_{row['timestamp']}"


def clean_text(text):
    """Remplace les balises HTML par des espaces, décode les entités, compacte les espaces."""
    if text is None:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_pool(df, pool_size=20_000, seed=42):
    """Filtre (notes 1-3, >= 5 mots, sans doublons) puis tire pool_size avis au hasard."""
    df = df[df["rating"] <= 3].copy()
    df["title"] = df["title"].map(clean_text)
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.split().str.len() >= 5]
    df = df.drop_duplicates(subset="text")
    df["review_id"] = df.apply(make_review_id, axis=1)
    df = df.drop_duplicates(subset="review_id")
    if len(df) < pool_size:
        raise ValueError(
            f"Seulement {len(df)} avis après filtrage, il en faut {pool_size} : augmente --n-read."
        )
    return df.sample(n=pool_size, random_state=seed).reset_index(drop=True)


def split_pool(pool, seed=42):
    """Ajoute une colonne `part` : explo / test / train / reserve, sans recouvrement."""
    shuffled = pool.sample(frac=1, random_state=seed)
    labels = []
    for part, n in PARTS.items():
        labels += [part] * n
    labels += ["reserve"] * (len(pool) - len(labels))
    shuffled["part"] = labels
    return shuffled


def diversity_report(df):
    """Indicateurs pour détecter un échantillon biaisé par l'ordre du fichier."""
    years = pd.to_datetime(df["timestamp"], unit="ms").dt.year
    return {
        "n_avis": len(df),
        "n_produits": df["parent_asin"].nunique(),
        "n_auteurs": df["user_id"].nunique(),
        "part_max_un_produit": round(df["parent_asin"].value_counts(normalize=True).iloc[0], 4),
        "annees": years.value_counts().sort_index().to_dict(),
        "notes": df["rating"].value_counts().sort_index().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-read", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = stream_reviews(args.n_read)
    print("Lecture brute :", diversity_report(raw))
    pool = split_pool(build_pool(raw, seed=args.seed), seed=args.seed)
    print("Réservoir :", diversity_report(pool))
    print(pool["part"].value_counts().to_dict())
    OUT.parent.mkdir(exist_ok=True)
    pool.to_parquet(OUT, index=False)
    print("Écrit :", OUT)


if __name__ == "__main__":
    main()
