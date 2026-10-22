"""Échauffement : des embeddings mistral-embed regardés de près, avant d'écrire le RAG.

Usage :
  python -m src.echauffement_embeddings          # 3 phrases courtes, comparées au TF-IDF (~20 tokens)
  python -m src.echauffement_embeddings --avis   # 2 avis réalistes contre 6 vrais passages du corpus (~1 000 tokens)
Chaque commande fait un seul appel (moins de 0,0002 $).
Test 1 : la norme des vecteurs (1 selon la documentation), et si l'embedding rapproche deux phrases de même sens sans
mot commun, là où TF-IDF ne voit que les mots.
Test 2 : sur des textes de la taille des vrais, le bon passage passe-t-il devant ceux qui partagent seulement le sujet
(le colis) ? Les deux avis sont inventés : aucun ne vient du jeu de test.
"""
import argparse
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.rag import charger_corpus, texte_a_indexer

PHRASES = ["the package arrived crushed", "box was damaged in shipping", "the package was too expensive"]

# Deux avis inventés, chacun avec le passage qu'une bonne réponse doit utiliser.
AVIS = [
    ("dommage", "P08",
     "Honey jar smashed\nOrdered two jars of raw honey as a gift. The package looked fine from the outside, but one "
     "jar was smashed inside and the honey had leaked all over the other jar and the box. Sticky mess, I had to throw "
     "everything away. Very disappointed."),
    ("prix", "P43",
     "Mostly air\nPaid almost $9 for this bag of chips and it was barely a third full when I opened it. The package "
     "is huge but there is almost nothing inside. Tastes fine, but this is not worth the price at all. Won't buy again."),
]
# Six passages : les deux bons, et des pièges qui partagent le sujet « colis » ou « prix ».
PASSAGES = ["P08", "P14", "P15", "P43", "P40", "P45"]


def client_mistral():
    from dotenv import load_dotenv
    from mistralai.client import Mistral

    load_dotenv()
    return Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def unitaires(rep):
    """Vecteurs de la réponse, renormalisés ligne par ligne."""
    E = np.array([d.embedding for d in rep.data])
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def test_phrases(client):
    rep = client.embeddings.create(model="mistral-embed", inputs=PHRASES)
    E = np.array([d.embedding for d in rep.data])
    print(f"Modèle : {rep.model} · dimension {E.shape[1]} · {rep.usage.prompt_tokens} tokens facturés")
    print("Normes :", np.linalg.norm(E, axis=1).round(6))

    En = unitaires(rep)
    S_emb = En @ En.T
    X = TfidfVectorizer().fit_transform(PHRASES)
    S_tfidf = (X @ X.T).toarray()          # TfidfVectorizer normalise déjà chaque ligne : produit scalaire = cosinus

    print(f"\n{'paire':<50} {'embedding':>10} {'TF-IDF':>8}")
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        print(f"{PHRASES[i][:22] + ' / ' + PHRASES[j][:22]:<50} {S_emb[i, j]:>10.3f} {S_tfidf[i, j]:>8.3f}")
    print("\nÀ noter : les trois normes, et si cos(crushed, damaged) > cos(crushed, expensive).")


def test_avis(client):
    par_id = {p["id"]: p for p in charger_corpus()}
    textes_passages = [texte_a_indexer(par_id[i]) for i in PASSAGES]   # exactement ce que le RAG indexera
    textes_avis = [texte for _, _, texte in AVIS]
    rep = client.embeddings.create(model="mistral-embed", inputs=textes_avis + textes_passages)
    E = unitaires(rep)
    A, P = E[:len(AVIS)], E[len(AVIS):]
    print(f"{rep.usage.prompt_tokens} tokens facturés ({rep.usage.prompt_tokens * 0.10 / 1e6:.6f} $)")

    for (nom, attendu, _), scores in zip(AVIS, A @ P.T):
        print(f"\nAvis « {nom} » : passage attendu {attendu}")
        for rang, k in enumerate(np.argsort(-scores), 1):
            pid = PASSAGES[k]
            marque = "  <- attendu" if pid == attendu else ""
            print(f"  {rang}. {pid}  {scores[k]:.3f}  {par_id[pid]['titre']}{marque}")
        rang_attendu = 1 + list(np.argsort(-scores)).index(PASSAGES.index(attendu))
        ecart = scores[PASSAGES.index(attendu)] - np.delete(scores, PASSAGES.index(attendu)).max()
        print(f"  -> rang du passage attendu : {rang_attendu} ; écart avec le meilleur autre passage : {ecart:+.3f}")
    print("\nÀ noter : le rang du passage attendu pour chaque avis, et les deux écarts.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--avis", action="store_true", help="avis réalistes contre de vrais passages du corpus")
    args = parser.parse_args()
    client = client_mistral()
    if args.avis:
        test_avis(client)
    else:
        test_phrases(client)


if __name__ == "__main__":
    main()
