"""Jalon 2 : RAG minimal écrit à la main. Retrouver les passages de la politique client utiles pour un avis, puis
rédiger une réponse qui les cite.

Usage :
  python -m src.rag index                                # embeddings du corpus (refaits seulement si le corpus change)
  python -m src.rag jeux                                 # tire rag_dev (50) et rag_eval (150) de la réserve, une fois
  python -m src.rag classer --part rag_dev               # motifs prédits par mistral-small, prompt v4
  python -m src.rag annoter --part rag_dev --n 30        # tes passages pertinents, AVANT toute recherche
  python -m src.rag eval-recherche --part rag_dev        # rappel@k et MRR, avec et sans filtre par motif
  python -m src.rag repondre --part rag_dev --mode rag --limit 5
  python -m src.rag montrer --part rag_dev --mode rag    # relire les réponses
Produit : data/rag/ (index, jeux, pertinents, vecteurs des avis, réponses) et data/labels_<part>_v4.jsonl.
Le jeu rag_eval est gelé comme le test : on ne le regarde pas, on ne règle rien dessus.
"""
import argparse
import hashlib
import json
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.label_llm import MODEL, PAUSE_S, POOL, call_with_retry, label_part, load_prompt
from src.metrics import MOTIFS

CORPUS = Path("corpus/politique_client.jsonl")
DOSSIER = Path("data/rag")
INDEX_NPY = DOSSIER / "index.npy"
INDEX_META = DOSSIER / "index.json"
JEUX = DOSSIER / "jeux.json"
EMBED_MODEL = "mistral-embed"
PRIX_EMBED = 0.10         # $ par million de tokens (fiche du modèle, 22 oct. 2026)
TAILLE_LOT = 32           # textes par requête d'embeddings (aucune limite documentée)
PAUSE_EMBED_S = 1.1
K = 8                     # rappel@8 sans filtre 0,885 sur 26 avis de rag_dev (23 oct. 2026), contre 0,769 à k = 5
N_DEV, N_EVAL = 50, 150
SEED = 42
MARQUE = "Larder Lane"    # marque fictive : à changer si tu préfères un autre nom
PROMPT_RAG = "rag_v4"      # v4 (23 oct. 2026) : cas B restreint (v3 : raison et cas avant la réponse ; v2 : règles)
SANS_PROMESSE = ("escalade", "escalade_securite", "aucune")   # actions qui ne promettent rien : citation facultative
MOTIFS_PASSAGE = set(MOTIFS) - {"autre"} | {"general"}
# Une citation = un ou plusieurs identifiants entre crochets : [P07], ou [P07, P12] que le modèle écrit parfois.
CITATION = re.compile(r"\[(P\d+(?:\s*[,;]\s*P\d+)*)\]")
ACTION_DU_CAS = {"securite": "escalade_securite", "sans_plainte": "aucune"}   # « plainte » : selon les passages

# Phrases à signaler au relecteur humain (cas vus sur rag_dev le 23 oct. 2026)
CONDITION_SUPPOSEE = re.compile(r"\b(since|as|because)\s+(this|it)\s+(is|was)\s+your\s+first\b|first-time buyer",
                                re.IGNORECASE)
ALTERNATIVE = re.compile(r"\b(refund|replacement)\b[^.]*\binstead\b|\bprefer\s+a\s+(refund|replacement)\b",
                         re.IGNORECASE)
AFFIRME_SECURITE = re.compile(r"\b(is|are)\s+(perfectly\s+|completely\s+)?safe\b|\bsafe\s+to\s+(eat|consume)\b"
                              r"|\bnot\s+affect\s+(the\s+)?safety\b", re.IGNORECASE)
# Mots de sécurité dans l'AVIS, indépendants du LLM : 3,1 % des 4 979 avis du train, dont 70 % classés
# fraicheur_securite par v4 (mesuré le 23 oct. 2026) ; les fausses alertes restantes se trient à la relecture.
MOTS_SECURITE = re.compile(
    r"\b(shards?|pits?|stones?|(pieces?|bits?|chunks?) of (glass|plastic|metal|bone|wood)|hair|insects?|bugs?|worms?"
    r"|larvae|maggots?|weevils?|moths?|mou?ldy?|sick|illness|vomit\w*|diarrh\w*|allergic|allergy|hospital\w*"
    r"|chok(ed|ing)|(broke|broken|cracked|chipped)\s+(a\s+|my\s+)?(tooth|teeth)|swollen|bulging)\b", re.IGNORECASE)

Action = Literal["remboursement", "remplacement", "geste_commercial", "escalade_securite", "escalade", "aucune"]
Cas = Literal["securite", "sans_plainte", "plainte"]


class ReponseClient(BaseModel):
    raison: str               # généré en 1er : ce que signale le client, en une phrase
    cas: Cas                  # généré en 2e : la décision, avant toute rédaction
    reponse: str              # le texte destiné au client, avec les [Pxx]
    citations: list[str]      # identifiants des passages utilisés
    action: Action            # ce que la marque fait ensuite


# ---------- Corpus ----------

def charger_corpus(chemin=CORPUS):
    """Liste des passages {id, titre, motifs, texte}, dans l'ordre du fichier. Vérifie le format."""
    with Path(chemin).open(encoding="utf-8") as f:
        passages = [json.loads(line) for line in f if line.strip()]
    ids = [p["id"] for p in passages]
    doublons = sorted({i for i in ids if ids.count(i) > 1})
    if doublons:
        raise ValueError(f"identifiants en double : {doublons}")
    for p in passages:
        if not re.fullmatch(r"P\d+", p["id"]):
            raise ValueError(f"identifiant mal formé : {p['id']}")
        inconnus = set(p["motifs"]) - MOTIFS_PASSAGE
        if not p["motifs"] or inconnus:
            raise ValueError(f"{p['id']} : motifs vides ou inconnus {sorted(inconnus)}")
        if not p["titre"].strip() or not p["texte"].strip():
            raise ValueError(f"{p['id']} : titre ou texte vide")
    return passages


def texte_a_indexer(passage):
    """Ce qu'on transforme en vecteur : le titre résume l'idée et rapproche le vecteur des avis qui en parlent."""
    return f"{passage['titre']}\n{passage['texte']}"


# ---------- Embeddings ----------

def normaliser(M):
    """Matrice (n, d) -> même matrice en float32, chaque ligne divisée par sa norme euclidienne (norme 1).

    Une ligne de norme nulle n'a pas de direction : ValueError.
    """
    M = np.asarray(M, dtype=np.float32)
    normes = np.linalg.norm(M, axis=1, keepdims=True)
    if (normes == 0).any():
        raise ValueError("vecteur nul : impossible de le normaliser")
    return M / normes


def embed(textes, client, taille_lot=TAILLE_LOT, pause_s=PAUSE_EMBED_S, sleep=time.sleep):
    """Embeddings de textes (liste non vide) par lots -> (matrice (n, d) normalisée, tokens facturés).

    Ligne i de la matrice = texte i. Un appel client.embeddings.create(model=EMBED_MODEL, inputs=lot) par lot,
    passé par call_with_retry(..., sleep=sleep) ; pause de pause_s entre deux lots (pas après le dernier).
    Les vecteurs sont dans rep.data[j].embedding, les tokens dans rep.usage.prompt_tokens.
    """
    vecteurs, tokens = [], 0
    for debut in range(0, len(textes), taille_lot):
        lot = textes[debut:debut + taille_lot]
        rep, _ = call_with_retry(lambda: client.embeddings.create(model=EMBED_MODEL, inputs=lot), sleep=sleep)
        vecteurs.extend(d.embedding for d in rep.data)
        tokens += rep.usage.prompt_tokens
        if debut + taille_lot < len(textes):
            sleep(pause_s)
    return normaliser(vecteurs), tokens


def empreinte(textes):
    """Résumé du corpus indexé : change dès qu'un caractère d'un passage (ou le modèle) change."""
    return hashlib.sha256(json.dumps([EMBED_MODEL, textes]).encode("utf-8")).hexdigest()


def construire_index(passages, client, npy=INDEX_NPY, meta=INDEX_META):
    """(matrice des passages, identifiants, tokens facturés). Relit l'index sauvegardé si le corpus n'a pas changé."""
    textes = [texte_a_indexer(p) for p in passages]
    ids = [p["id"] for p in passages]
    signe = empreinte(textes)
    if npy.exists() and meta.exists():
        info = json.loads(meta.read_text(encoding="utf-8"))
        if info["empreinte"] == signe:
            return np.load(npy), info["ids"], 0
    M, tokens = embed(textes, client)
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, M)
    meta.write_text(json.dumps({"empreinte": signe, "modele": EMBED_MODEL, "ids": ids, "tokens": tokens,
                                "date": date.today().isoformat()}, indent=1), encoding="utf-8")
    return M, ids, tokens


# ---------- Recherche ----------

def passages_autorises(motifs_passages, motifs_avis):
    """Masque booléen (numpy, un par passage) : les passages parmi lesquels chercher pour cet avis.

    motifs_passages : liste (un par passage) de listes de motifs. motifs_avis : motifs prédits pour l'avis.
    Pas de filtre (tout True) si motifs_avis est None, vide, ou contient "autre".
    Sinon : True si le passage partage au moins un motif avec l'avis, ou s'il porte "general".
    """
    if not motifs_avis or "autre" in motifs_avis:
        return np.ones(len(motifs_passages), dtype=bool)
    voulus = set(motifs_avis) | {"general"}
    return np.array([bool(voulus & set(m)) for m in motifs_passages], dtype=bool)


def rechercher(q, M, ids, k=K, masque=None):
    """Les k passages les plus proches de q : liste de (id, score) par score décroissant, scores en float Python.

    q : vecteur (d,) normalisé ; M : matrice (n, d) normalisée, ligne i = passage ids[i] ;
    masque : booléens (n,) ou None (tous autorisés). Moins de k passages autorisés : on renvoie ceux-là.
    À score égal, l'ordre du corpus (tri stable).
    """
    scores = M @ q
    candidats = np.flatnonzero(masque) if masque is not None else np.arange(len(ids))
    ordre = candidats[np.argsort(-scores[candidats], kind="stable")][:k]
    return [(ids[i], float(scores[i])) for i in ordre]


# ---------- Génération ----------

def construire_messages(system_prompt, passages, title, text):
    """Messages de l'appel : le prompt système, puis les passages et l'avis entre délimiteurs.

    Message utilisateur, au caractère près (une ligne par passage, dans l'ordre reçu) :
        <passages>
        [P07] Damaged item on arrival: If your order ...
        [P12] Titre: texte
        </passages>
        <review>
        Title: {title}
        Text: {text}
        </review>
    """
    bloc = "\n".join(f"[{p['id']}] {p['titre']}: {p['texte']}" for p in passages)
    user = f"<passages>\n{bloc}\n</passages>\n<review>\nTitle: {title}\nText: {text}\n</review>"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def verifier_citations(reponse, citations, action, ids_fournis):
    """Règles que le schéma ne garantit pas. Renvoie la liste des problèmes (vide = valide).

    Dans cet ordre, chacun au plus une fois :
      "citation_inconnue"         : un identifiant (de citations ou du texte) absent de ids_fournis ;
      "citation_non_declaree"     : un [Pxx] du texte absent de citations ;
      "citation_absente_du_texte" : un identifiant de citations jamais écrit [Pxx] dans le texte ;
      "sans_citation"             : citations vide alors que l'action promet quelque chose (hors SANS_PROMESSE).
    Les [Pxx] du texte se trouvent avec ids_dans_texte(reponse).
    """
    dans_texte = ids_dans_texte(reponse)
    declarees = set(citations)
    problemes = []
    if (dans_texte | declarees) - set(ids_fournis):
        problemes.append("citation_inconnue")
    if dans_texte - declarees:
        problemes.append("citation_non_declaree")
    if declarees - dans_texte:
        problemes.append("citation_absente_du_texte")
    if not citations and action not in SANS_PROMESSE:
        problemes.append("sans_citation")
    return problemes


def ids_dans_texte(reponse):
    """Ensemble des identifiants cités dans le texte : [P07] -> {"P07"}, [P20, P23] -> {"P20", "P23"}."""
    return {i for groupe in CITATION.findall(reponse) for i in re.findall(r"P\d+", groupe)}


def signaux_redaction(reponse, action, cas, avis=""):
    """Phrases et incohérences à signaler au relecteur humain. Le code ne corrige rien : il étiquette.

    Dans cet ordre, chacun au plus une fois :
      "cas_action_incoherent"  : le cas impose une action (securite -> escalade_securite, sans_plainte -> aucune)
                                 et l'action choisie est différente ;
      "condition_supposee"     : une condition invérifiable affirmée (« since this is your first purchase ») ;
      "alternative_non_prevue" : un remboursement ou remplacement proposé « à la place » d'un avoir ;
      "affirme_securite"       : le texte affirme qu'un produit est sûr (interdit par P05 avant enquête) ;
      "securite_a_verifier"    : l'avis (titre + texte) contient un mot de sécurité et l'action n'est pas
                                 escalade_securite. Seul signal qui ne dépend pas de ce que le LLM a écrit.
    """
    signaux = []
    if cas in ACTION_DU_CAS and action != ACTION_DU_CAS[cas]:
        signaux.append("cas_action_incoherent")
    if CONDITION_SUPPOSEE.search(reponse):
        signaux.append("condition_supposee")
    if action == "geste_commercial" and ALTERNATIVE.search(reponse):
        signaux.append("alternative_non_prevue")
    if AFFIRME_SECURITE.search(reponse):
        signaux.append("affirme_securite")
    if action != "escalade_securite" and MOTS_SECURITE.search(avis):
        signaux.append("securite_a_verifier")
    return signaux


# ---------- Mesure de la recherche ----------

def rappel_a_k(classements, pertinents, k):
    """Part des avis dont au moins un passage pertinent figure dans les k premiers de leur classement.

    classements : liste (un par avis) de listes d'identifiants, du plus proche au moins proche ;
    pertinents : liste (un par avis) d'ensembles non vides d'identifiants.
    """
    trouves = [bool(set(c[:k]) & p) for c, p in zip(classements, pertinents)]
    return float(np.mean(trouves))


def mrr(classements, pertinents):
    """Moyenne sur les avis de 1 / rang du premier passage pertinent (rang 1 = premier) ; 0 s'il est absent."""
    inverses = []
    for c, p in zip(classements, pertinents):
        rangs = [r for r, i in enumerate(c, 1) if i in p]
        inverses.append(1 / rangs[0] if rangs else 0.0)
    return float(np.mean(inverses))


def ic_bootstrap(valeurs, n_boot=2000, seed=SEED):
    """(moyenne, borne basse, borne haute) à 95 % : on tire des avis avec remise."""
    valeurs = np.asarray(valeurs, dtype=float)
    rng = np.random.default_rng(seed)
    tirages = [valeurs[rng.integers(0, len(valeurs), len(valeurs))].mean() for _ in range(n_boot)]
    return float(valeurs.mean()), *np.percentile(tirages, [2.5, 97.5]).tolist()


# ---------- Jeux d'avis du jalon 2 ----------

def tirer_jeux(pool, n_dev=N_DEV, n_eval=N_EVAL, seed=SEED):
    """{"rag_dev": [review_id...], "rag_eval": [...]} tirés de la réserve, sans recouvrement.

    Trié par review_id avant le tirage : le résultat ne dépend pas de l'ordre des lignes du fichier.
    """
    reserve = pool[pool["part"] == "reserve"].sort_values("review_id")
    tirage = reserve.sample(n_dev + n_eval, random_state=seed)["review_id"].tolist()
    return {"rag_dev": tirage[:n_dev], "rag_eval": tirage[n_dev:]}


def charger_part(part):
    """Les avis d'un jeu du jalon 2, dans l'ordre du tirage."""
    ids = json.loads(JEUX.read_text(encoding="utf-8"))[part]
    pool = pd.read_parquet(POOL).set_index("review_id")
    return pool.loc[ids].reset_index()


def charger_motifs(part):
    """{review_id: motifs prédits par v4} ; la dernière ligne réussie l'emporte ; motifs invalides -> None."""
    chemin = Path(f"data/labels_{part}_v4.jsonl")
    if not chemin.exists():
        return {}
    motifs = {}
    with chemin.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                motifs[r["review_id"]] = None if r["problems"] else r["motifs"]
    return motifs


def vecteurs_avis(df, client, chemin):
    """{review_id: vecteur normalisé} ; calcule seulement les avis absents du fichier .npz, puis le met à jour."""
    connus = dict(np.load(chemin)) if chemin.exists() else {}
    manquants = df[~df["review_id"].isin(connus)]
    if len(manquants):
        textes = (manquants["title"].astype(str) + "\n" + manquants["text"].astype(str)).tolist()
        M, tokens = embed(textes, client)
        connus.update(zip(manquants["review_id"], M))
        chemin.parent.mkdir(parents=True, exist_ok=True)
        np.savez(chemin, **connus)
        print(f"{len(manquants)} avis vectorisés, {tokens} tokens ({tokens * PRIX_EMBED / 1e6:.5f} $)")
    return connus


# ---------- Commandes ----------

def client_mistral():
    from dotenv import load_dotenv
    from mistralai.client import Mistral
    load_dotenv()
    return Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def cmd_index():
    passages = charger_corpus()
    M, ids, tokens = construire_index(passages, client_mistral())
    if tokens:
        print(f"Index construit : {len(ids)} passages, dimension {M.shape[1]}, {tokens} tokens "
              f"({tokens * PRIX_EMBED / 1e6:.5f} $)")
    else:
        print(f"Corpus inchangé : index relu ({len(ids)} passages, aucun appel)")
    print("Normes min et max :", M.dtype, np.linalg.norm(M, axis=1).min().round(4), np.linalg.norm(M, axis=1).max().round(4))


def cmd_jeux():
    if JEUX.exists():
        print(f"{JEUX} existe déjà : les jeux sont tirés une fois pour toutes. Supprime-le à la main pour refaire.")
        return
    jeux = tirer_jeux(pd.read_parquet(POOL))
    JEUX.parent.mkdir(parents=True, exist_ok=True)
    JEUX.write_text(json.dumps(jeux, indent=1), encoding="utf-8")
    print(f"Écrit : {JEUX} ({len(jeux['rag_dev'])} rag_dev, {len(jeux['rag_eval'])} rag_eval)")


def cmd_classer(part, limit):
    df = charger_part(part)
    if limit:
        df = df.head(limit)
    out = Path(f"data/labels_{part}_v4.jsonl")
    label_part(client_mistral(), df, load_prompt("v4"), "v4", out)
    print("Écrit :", out)


def afficher_corpus(passages):
    for motif in sorted(MOTIFS_PASSAGE):
        lignes = [f"{p['id']} {p['titre']}" for p in passages if motif in p["motifs"]]
        print(f"\n[{motif}]  " + "  |  ".join(lignes))


def cmd_annoter(part, n):
    """Pour chaque avis : les identifiants des passages qu'une bonne réponse doit utiliser."""
    passages = charger_corpus()
    connus = {p["id"] for p in passages}
    out = DOSSIER / f"pertinents_{part}.jsonl"
    faits = set()
    if out.exists():
        faits = {json.loads(l)["review_id"] for l in out.open(encoding="utf-8") if l.strip()}
    df = charger_part(part).head(n)
    a_faire = df[~df["review_id"].isin(faits)]
    print(f"{len(faits)} déjà annotés, {len(a_faire)} à faire. Saisie : identifiants séparés par des espaces "
          f"(ex. P07 P12), 0 si aucun passage ne couvre l'avis, ? pour revoir la liste, q pour quitter.")
    afficher_corpus(passages)
    with out.open("a", encoding="utf-8") as f:
        for i, row in enumerate(a_faire.to_dict("records"), len(faits) + 1):
            print(f"\n----- {i}/{len(df)} · note {int(row['rating'])}\nTitle: {row['title']}\nText: {row['text']}")
            while True:
                saisie = input("Passages pertinents > ").strip()
                if saisie == "q":
                    return
                if saisie == "?":
                    afficher_corpus(passages)
                    continue
                choix = [] if saisie == "0" else saisie.upper().split()
                inconnus = [c for c in choix if c not in connus]
                if saisie and not inconnus:
                    break
                print("Identifiant inconnu :", inconnus or "saisie vide")
            f.write(json.dumps({"review_id": row["review_id"], "pertinents": choix}) + "\n")
            f.flush()


def cmd_eval_recherche(part, k_max=10):
    passages = charger_corpus()
    client = client_mistral()
    M, ids, _ = construire_index(passages, client)
    lignes = [json.loads(l) for l in (DOSSIER / f"pertinents_{part}.jsonl").open(encoding="utf-8") if l.strip()]
    sans = [r for r in lignes if not r["pertinents"]]
    lignes = [r for r in lignes if r["pertinents"]]
    print(f"{len(lignes)} avis annotés avec au moins un passage pertinent ({len(sans)} sans, exclus)")
    df = charger_part(part).set_index("review_id").loc[[r["review_id"] for r in lignes]].reset_index()
    vecteurs = vecteurs_avis(df, client, DOSSIER / f"avis_{part}.npz")
    motifs = charger_motifs(part)
    if not motifs:
        print("Pas de motifs prédits : lance d'abord `classer`. Le filtre sera sans effet.")
    pertinents = [set(r["pertinents"]) for r in lignes]
    motifs_passages = [p["motifs"] for p in passages]
    for filtre in (False, True):
        classements = []
        for r in lignes:
            masque = passages_autorises(motifs_passages, motifs.get(r["review_id"])) if filtre else None
            classements.append([i for i, _ in rechercher(vecteurs[r["review_id"]], M, ids, k_max, masque)])
        print(f"\n== {'avec' if filtre else 'sans'} filtre par motif ==")
        for k in (1, 3, 5, K):
            # l'intervalle a besoin du résultat avis par avis : rappel_a_k sur chaque avis seul (0 ou 1)
            par_avis = [rappel_a_k([c], [p], k) for c, p in zip(classements, pertinents)]
            _, bas, haut = ic_bootstrap(par_avis)
            print(f"rappel@{k} = {rappel_a_k(classements, pertinents, k):.3f} [{bas:.3f} ; {haut:.3f}]")
        print(f"MRR (sur les {k_max} premiers) = {mrr(classements, pertinents):.3f}")
        rates = [(r["review_id"], sorted(p), c[:K]) for r, c, p in zip(lignes, classements, pertinents)
                 if not set(c[:K]) & p]
        for rid, p, top in rates:
            print(f"  raté à k={K} : {rid[-24:]}  pertinents {p}  trouvés {top}")


def cmd_repondre(part, mode, k, limit, filtre):
    client = client_mistral()
    passages = charger_corpus()
    par_id = {p["id"]: p for p in passages}
    system_prompt = load_prompt(PROMPT_RAG).replace("{marque}", MARQUE)
    df = charger_part(part)
    if limit:
        df = df.head(limit)
    out = DOSSIER / f"reponses_{part}_{mode}_{PROMPT_RAG}.jsonl"
    faits = set()
    if out.exists():
        faits = {r["review_id"] for r in map(json.loads, out.open(encoding="utf-8")) if "error" not in r}
    df = df[~df["review_id"].isin(faits)]
    print(f"{len(faits)} déjà faits, {len(df)} à faire")
    if mode == "rag":
        M, ids, _ = construire_index(passages, client)
        vecteurs = vecteurs_avis(df, client, DOSSIER / f"avis_{part}.npz")
        motifs = charger_motifs(part)
        motifs_passages = [p["motifs"] for p in passages]
    with out.open("a", encoding="utf-8") as f:
        for row in df.to_dict("records"):
            if mode == "rag":
                masque = passages_autorises(motifs_passages, motifs.get(row["review_id"])) if filtre else None
                trouves = rechercher(vecteurs[row["review_id"]], M, ids, k, masque)
                fournis = [par_id[i] for i, _ in trouves]
            else:
                fournis = passages
            messages = construire_messages(system_prompt, fournis, row["title"], row["text"])
            ligne = {"review_id": row["review_id"], "mode": mode, "k": k if mode == "rag" else None,
                     "filtre": filtre if mode == "rag" else None, "prompt_version": PROMPT_RAG,
                     "passages": [p["id"] for p in fournis] if mode == "rag" else "tous"}
            try:
                t0 = time.perf_counter()
                rep, essais = call_with_retry(lambda: client.chat.parse(
                    model=MODEL, response_format=ReponseClient, temperature=0, max_tokens=500, random_seed=42,
                    messages=messages))
                parsed = rep.choices[0].message.parsed
                if parsed:
                    problemes = (verifier_citations(parsed.reponse, parsed.citations, parsed.action,
                                                    [p["id"] for p in fournis])
                                 + signaux_redaction(parsed.reponse, parsed.action, parsed.cas,
                                                     f"{row['title']} {row['text']}"))
                else:
                    problemes = ["reponse_vide"]
                ligne.update({
                    "model": rep.model,
                    "raison": parsed.raison if parsed else None,
                    "cas": parsed.cas if parsed else None,
                    "reponse": parsed.reponse if parsed else None,
                    "citations": parsed.citations if parsed else [],
                    "action": parsed.action if parsed else None,
                    "problems": problemes,
                    "prompt_tokens": rep.usage.prompt_tokens, "completion_tokens": rep.usage.completion_tokens,
                    "latency_s": round(time.perf_counter() - t0, 3), "attempts": essais})
            except Exception as e:
                ligne["error"] = repr(e)[:300]
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {row['review_id'][-24:]} : {ligne.get('cas', '')} -> {ligne.get('action', 'ERREUR')} "
                  f"{ligne.get('problems', '')}")
            time.sleep(PAUSE_S)
    print("Écrit :", out)


def cmd_montrer(part, mode):
    df = charger_part(part).set_index("review_id")
    chemin = DOSSIER / f"reponses_{part}_{mode}_{PROMPT_RAG}.jsonl"
    for r in map(json.loads, chemin.open(encoding="utf-8")):
        if "error" in r:
            continue
        avis = df.loc[r["review_id"]]
        print(f"\n===== {r['review_id'][-24:]} · note {int(avis['rating'])}\nTitle: {avis['title']}\nText: {avis['text']}")
        if r.get("cas"):                                  # absents des réponses rag_v1 et rag_v2
            print(f"--- raison : {r['raison']}\n--- cas : {r['cas']}")
        print(f"--- passages fournis : {r['passages']}\n--- action : {r['action']} · problèmes : {r['problems']} "
              f"· {r['prompt_tokens']} tokens d'entrée, {r['completion_tokens']} de sortie\n{r['reponse']}")


def main():
    parser = argparse.ArgumentParser()
    sous = parser.add_subparsers(dest="commande", required=True)
    sous.add_parser("index")
    sous.add_parser("jeux")
    p = sous.add_parser("classer")
    p.add_argument("--part", default="rag_dev")
    p.add_argument("--limit", type=int, default=None)
    p = sous.add_parser("annoter")
    p.add_argument("--part", default="rag_dev")
    p.add_argument("--n", type=int, default=30)
    p = sous.add_parser("eval-recherche")
    p.add_argument("--part", default="rag_dev")
    p = sous.add_parser("repondre")
    p.add_argument("--part", default="rag_dev")
    p.add_argument("--mode", choices=["rag", "plein"], default="rag")
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--filtre", action="store_true",
                   help="filtrer par motif prédit (désactivé par défaut : il excluait des passages de sécurité)")
    p = sous.add_parser("montrer")
    p.add_argument("--part", default="rag_dev")
    p.add_argument("--mode", choices=["rag", "plein"], default="rag")
    args = parser.parse_args()

    if args.commande == "index":
        cmd_index()
    elif args.commande == "jeux":
        cmd_jeux()
    elif args.commande == "classer":
        cmd_classer(args.part, args.limit)
    elif args.commande == "annoter":
        cmd_annoter(args.part, args.n)
    elif args.commande == "eval-recherche":
        cmd_eval_recherche(args.part)
    elif args.commande == "repondre":
        cmd_repondre(args.part, args.mode, args.k, args.limit, args.filtre)
    elif args.commande == "montrer":
        cmd_montrer(args.part, args.mode)


if __name__ == "__main__":
    main()
