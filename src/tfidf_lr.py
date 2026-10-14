"""Étape 4 : TF-IDF + régression logistique, entraînés sur les étiquettes du LLM (cours 3).

Usage :
  python -m src.tfidf_lr grid     # les 20 configurations : entraînement sur 3 983 avis, mesure sur les 996 de validation
  python -m src.tfidf_lr final    # UNE seule fois : la configuration choisie prédit les 300 avis de test
  python -m src.tfidf_lr explique                            # poids de termes, confusions et exemples sur la validation
  python -m src.tfidf_lr explique --texte "Tastes like moldy water"     # ce qui pousse chaque motif pour ce texte
  python -m src.tfidf_lr explique --termes taste stale --motif autre    # autres termes, confusions d'un autre motif
Produit : data/tfidf_lr_choix.json (grille et configuration choisie) ; data/preds_test_tfidf.jsonl (une ligne par
avis de test, format lu par `python -m src.metrics eval`).
Le test n'est lu que par `final`, qui refuse de tourner deux fois (cours 3, section 6 : le biais du gagnant).
"""
import argparse
import json
import pickle
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

from src.label_llm import POOL, check_rules
from src.metrics import MOTIFS, exact_match, macro_f1, scores, to_matrix

LABELS = Path("data/labels_train_v4.jsonl")
CHOIX = Path("data/tfidf_lr_choix.json")
PREDS_TEST = Path("data/preds_test_tfidf.jsonl")
N_VAL = 1000              # validation tirée du train (CONVENTIONS.md, section Modèles entraînés)
SEED = 42
THRESHOLD = 0.5           # fixe : class_weight="balanced" fait déjà le travail du seuil (cours 3, section 4)
MOTIFS_PLAINTE = [m for m in MOTIFS if m != "autre"]   # 5 classifieurs ; autre se déduit (cours 3, section 5)
GRID = [{"C": C, "ngram_range": ng, "min_df": md}      # fixée avant toute mesure : 20 configurations
        for C in (0.1, 0.3, 1, 3, 10) for ng in ((1, 1), (1, 2)) for md in (1, 3)]
REFIT = True   # décision B du 14 oct. 2026 (CONVENTIONS.md) : ré-entraîné sur 4 979 avis avant le test


def load_train_labels(path=LABELS):
    """Étiquettes valides du professeur : {review_id: motifs}.

    Ignore les lignes vides, les lignes `error` et les avis dont `problems` n'est pas vide (les 21 invalides,
    option A de CONVENTIONS.md). Si un avis a plusieurs lignes valides, la dernière l'emporte.
    """
    labels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r or r["problems"]:
                continue
            labels[r["review_id"]] = r["motifs"]
    return labels


def split_train_val(ids, n_val=N_VAL, seed=SEED):
    """(train_ids, val_ids) : deux listes disjointes qui couvrent ids, avec len(val_ids) == n_val.

    Recette de CONVENTIONS.md : trier les ids, les permuter avec np.random.default_rng(seed).permutation,
    les n_val premiers vont en validation. Le résultat ne dépend donc pas de l'ordre de `ids`.
    """
    ids = sorted(ids)
    perm = np.random.default_rng(seed).permutation(len(ids))
    return [ids[i] for i in perm[n_val:]], [ids[i] for i in perm[:n_val]]


def build_text(df):
    """Liste des textes à vectoriser, un par ligne de df, dans l'ordre : titre, une espace, texte.

    Titre absent (None ou NaN) -> chaîne vide. Le titre compte autant que le texte (guide v1).
    """
    return [("" if pd.isna(title) else title) + " " + text for title, text in zip(df["title"], df["text"])]


def build_pipeline(C, ngram_range, min_df):
    """Pipeline non entraîné : TfidfVectorizer, puis un classifieur logistique par motif de plainte.

    TfidfVectorizer(ngram_range, min_df, sublinear_tf=True), mots vides gardés (stop_words par défaut) ;
    OneVsRestClassifier(LogisticRegression(C, class_weight="balanced", max_iter=2000)), sans `penalty`
    (déprécié depuis scikit-learn 1.8 ; L2 par défaut). Cours 3, sections 3, 4 et 6.
    """
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, sublinear_tf=True)
    classifier = OneVsRestClassifier(LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
    return make_pipeline(vectorizer, classifier)


def to_motifs(proba, threshold=THRESHOLD):
    """Probabilités (n_avis, 5), colonnes dans l'ordre de MOTIFS_PLAINTE -> listes de motifs valides.

    Un motif est retenu si sa probabilité est >= threshold ; un avis sans aucun motif retenu reçoit ["autre"].
    Chaque liste suit l'ordre de MOTIFS. Lève ValueError si proba n'a pas exactement 5 colonnes
    (garde-fou contre une matrice à 6 colonnes ou des colonnes décalées).
    """
    proba = np.asarray(proba)
    if proba.ndim != 2 or proba.shape[1] != len(MOTIFS_PLAINTE):
        raise ValueError(f"proba doit avoir la forme (n, {len(MOTIFS_PLAINTE)}), reçu {proba.shape}")
    out = []
    for row in proba:
        motifs = [m for m, p in zip(MOTIFS_PLAINTE, row) if p >= threshold]
        out.append(motifs or ["autre"])
    return out


def top_words(pipeline, motif, k=10):
    """Les k termes (mots ou paires de mots) au poids le plus élevé pour `motif`, du plus lourd au moins lourd.

    pipeline : un pipeline de build_pipeline déjà entraîné. motif : un des MOTIFS_PLAINTE, sinon ValueError
    (autre n'a pas de classifieur). Cours 3, section 7.
    """
    if motif not in MOTIFS_PLAINTE:
        raise ValueError(f"{motif!r} n'a pas de classifieur (autre se déduit des 5 autres motifs)")
    names = pipeline[0].get_feature_names_out()
    weights = pipeline[-1].estimators_[MOTIFS_PLAINTE.index(motif)].coef_[0]
    return [str(names[i]) for i in np.argsort(weights)[::-1][:k]]


# ---------- Expliquer les prédictions (cours 3, section 7) ----------

TERMES_FRAICHEUR = ["taste", "flavor", "stale", "old", "date", "expired", "sick", "rancid", "moldy", "bugs"]


def term_weights(pipeline, terms, texts):
    """Tableau des poids (DataFrame) : une ligne par terme, une colonne par motif de plainte, plus `n_avis`.

    n_avis = nombre de textes de `texts` (ceux de l'entraînement du modèle) qui contiennent le terme. Un terme absent
    du vocabulaire (vu dans moins de min_df avis) a des poids et un n_avis vides (NaN).
    """
    vec, clf = pipeline[0], pipeline[-1]
    W = np.vstack([est.coef_[0] for est in clf.estimators_])        # (5 motifs, n_termes)
    X = vec.transform(texts)
    rows = []
    for t in terms:
        j = vec.vocabulary_.get(t.lower())                         # le vectoriseur met tout en minuscules
        if j is None:
            rows.append({**{m: np.nan for m in MOTIFS_PLAINTE}, "n_avis": np.nan})
        else:
            rows.append({**{m: W[k, j] for k, m in enumerate(MOTIFS_PLAINTE)}, "n_avis": int((X[:, j] > 0).sum())})
    return pd.DataFrame(rows, index=list(terms))


def contributions(pipeline, text):
    """Décomposition des 5 probabilités d'un texte : {motif: {"proba", "intercept", "termes"}}.

    termes : [(terme, contribution), ...] pour chaque terme du texte présent dans le vocabulaire, du plus favorable
    au plus défavorable au motif ; contribution = poids du terme x sa valeur TF-IDF dans le texte.
    Par construction, proba = sigmoïde(somme des contributions + intercept) (cours 3, section 4).
    """
    vec, clf = pipeline[0], pipeline[-1]
    x = vec.transform([text])                                      # une ligne creuse : x.indices, x.data
    names = vec.get_feature_names_out()
    out = {}
    for est, m in zip(clf.estimators_, MOTIFS_PLAINTE):
        w, b = est.coef_[0], float(est.intercept_[0])
        termes = sorted(((str(names[j]), float(v * w[j])) for j, v in zip(x.indices, x.data)),
                        key=lambda tc: -tc[1])
        z = sum(c for _, c in termes) + b
        out[m] = {"proba": float(1 / (1 + np.exp(-z))), "intercept": b, "termes": termes}
    return out


def confusions(ref, pred, motif):
    """Pour un motif, deux listes alignées de listes de motifs (référence, prédictions) ->
    {"positifs", "trouves", "manques", "en_trop", "a_la_place"}.

    a_la_place (Counter) compte ce que le modèle a prédit sur les avis manqués. Vaut pour n'importe quel modèle.
    """
    pos = [p for r, p in zip(ref, pred) if motif in r]
    manques = [tuple(p) for p in pos if motif not in p]
    return {"positifs": len(pos), "trouves": len(pos) - len(manques), "manques": len(manques),
            "en_trop": sum(motif in p and motif not in r for r, p in zip(ref, pred)),
            "a_la_place": Counter(manques)}


# ---------- Données, grille, mesure finale ----------

def plainte_matrix(motifs_lists):
    """Listes de motifs -> matrice 0/1 (n_avis, 5) dans l'ordre de MOTIFS_PLAINTE : la cible des 5 classifieurs."""
    return to_matrix(list(motifs_lists))[:, [MOTIFS.index(m) for m in MOTIFS_PLAINTE]]


def load_data(pool_path=POOL, labels_path=LABELS):
    """(train, val) : avis étiquetés de la partie train du pool, avec une colonne `motifs`.

    Découpage de CONVENTIONS.md : tirage sur les 5 000 avis, puis exclusion des invalides (3 983 / 996).
    """
    pool = pd.read_parquet(pool_path)
    part = pool[pool["part"] == "train"]
    train_ids, val_ids = split_train_val(part["review_id"].tolist())
    labels = load_train_labels(labels_path)
    part = part[part["review_id"].isin(labels)].copy()
    part["motifs"] = part["review_id"].map(labels)
    return part[part["review_id"].isin(set(train_ids))], part[part["review_id"].isin(set(val_ids))]


def evaluate(pipeline, df):
    """Prédictions valides (règle autre) sur df, mesurées sur les 6 motifs contre les étiquettes du professeur."""
    pred = to_motifs(pipeline.predict_proba(build_text(df)))
    Y, P = to_matrix(list(df["motifs"])), to_matrix(pred)
    return {"macro_f1": macro_f1(Y, P), "exact": exact_match(Y, P),
            "invalides": sum(bool(check_rules(m)) for m in pred), "scores": scores(Y, P)}


def run_grid(train, val, out=CHOIX):
    X_tr, Y_tr = build_text(train), plainte_matrix(train["motifs"])
    print(f"Entraînement : {len(train)} avis ; validation : {len(val)} avis ; {len(GRID)} configurations "
          f"(class_weight='balanced', sublinear_tf, seuil {THRESHOLD}, règle autre appliquée)")
    print(f"{'C':>5s} {'n-grammes':>9s} {'min_df':>6s} {'vocab':>7s} {'F1 entr.':>8s} {'F1 val':>7s} "
          f"{'exact val':>9s} {'invalides':>9s} {'durée':>7s}")
    rows = []
    for cfg in GRID:
        t0 = time.perf_counter()
        pipe = build_pipeline(**cfg).fit(X_tr, Y_tr)
        fit_s = time.perf_counter() - t0
        tr, va = evaluate(pipe, train), evaluate(pipe, val)
        rows.append({**cfg, "vocab": len(pipe[0].vocabulary_), "f1_train": round(tr["macro_f1"], 4),
                     "f1_val": round(va["macro_f1"], 4), "exact_val": round(va["exact"], 4),
                     "invalides_val": va["invalides"], "fit_s": round(fit_s, 2)})
        r = rows[-1]
        print(f"{cfg['C']:>5} {str(cfg['ngram_range']):>9s} {cfg['min_df']:>6} {r['vocab']:>7} "
              f"{r['f1_train']:>8.3f} {r['f1_val']:>7.3f} {r['exact_val']:>9.3f} {r['invalides_val']:>9} "
              f"{r['fit_s']:>6.1f}s")

    best = max(range(len(rows)), key=lambda i: rows[i]["f1_val"])   # à égalité : la première, donc le C le plus petit
    cfg = GRID[best]
    pipe = build_pipeline(**cfg).fit(X_tr, Y_tr)
    va = evaluate(pipe, val)
    print(f"\nChoisie sur la validation : {cfg} ; macro-F1 val {rows[best]['f1_val']:.3f}, "
          f"écart entraînement - validation {rows[best]['f1_train'] - rows[best]['f1_val']:.3f}")
    print(f"{'motif':20s} {'P':>6s} {'R':>6s} {'F1':>6s} {'support':>8s}")
    for m in MOTIFS:
        d = va["scores"]["par_motif"][m]
        print(f"{m:20s} {d['p']:6.3f} {d['r']:6.3f} {d['f1']:6.3f} {d['support']:8d}")
    texts = build_text(val)
    t0 = time.perf_counter()
    pipe.predict_proba(texts)
    batch_ms = (time.perf_counter() - t0) / len(texts) * 1000
    print(f"Prédiction par lot : {batch_ms:.3f} ms par avis ; taille du modèle : {len(pickle.dumps(pipe)) / 1e6:.1f} Mo")
    print("\nTermes les plus lourds (cours 3, section 7) :")
    for m in MOTIFS_PLAINTE:
        print(f"  {m:20s} " + ", ".join(top_words(pipe, m, 10)))

    out.write_text(json.dumps({"date": date.today().isoformat(), "config": cfg, "f1_val": rows[best]["f1_val"],
                               "n_train": len(train), "n_val": len(val), "grille": rows},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nChoix enregistré dans {out}. Avant `final` : trancher REFIT (cours 3, section 6).")


def run_final(pool_path=POOL, labels_path=LABELS, choix=CHOIX, out=PREDS_TEST, refit=REFIT):
    if out.exists():
        raise SystemExit(f"{out} existe déjà : le test se mesure une seule fois par modèle figé (cours 3, section 6).")
    if refit is None:
        raise SystemExit("Tranche d'abord le ré-entraînement (cours 3, section 6) : REFIT = False ou True en tête "
                         "de src/tfidf_lr.py, et consigne la décision dans CONVENTIONS.md.")
    if not choix.exists():
        raise SystemExit("Lance d'abord `python -m src.tfidf_lr grid`.")
    cfg = json.loads(choix.read_text(encoding="utf-8"))["config"]
    cfg["ngram_range"] = tuple(cfg["ngram_range"])          # JSON ne connaît que les listes
    train, val = load_data(pool_path, labels_path)
    fit_df = pd.concat([train, val]) if refit else train
    t0 = time.perf_counter()
    pipe = build_pipeline(**cfg).fit(build_text(fit_df), plainte_matrix(fit_df["motifs"]))
    fit_s = time.perf_counter() - t0

    test = pd.read_parquet(pool_path).query("part == 'test'")
    texts = build_text(test)
    t0 = time.perf_counter()
    pred = to_motifs(pipe.predict_proba(texts))
    batch_ms = (time.perf_counter() - t0) / len(texts) * 1000
    t0 = time.perf_counter()
    for x in texts[:200]:
        pipe.predict_proba([x])
    one_ms = (time.perf_counter() - t0) / min(200, len(texts)) * 1000

    with open(out, "w", encoding="utf-8") as f:
        for rid, motifs in zip(test["review_id"], pred):
            f.write(json.dumps({"review_id": rid, "model": "tfidf_lr", "config": cfg, "refit": refit,
                                "n_train": len(fit_df), "motifs": motifs, "problems": check_rules(motifs)},
                               ensure_ascii=False) + "\n")
    print(f"Configuration {cfg}, entraînée sur {len(fit_df)} avis en {fit_s:.1f} s.")
    print(f"{len(pred)} avis de test prédits, {sum(bool(check_rules(m)) for m in pred)} invalides -> {out}")
    print(f"Latence : {batch_ms:.3f} ms par avis par lot ; {one_ms:.2f} ms un avis à la fois ; "
          f"modèle {len(pickle.dumps(pipe)) / 1e6:.1f} Mo")
    print(f"Mesure : python -m src.metrics eval {out}")
    print(f"Comparaison appariée avec le LLM : python -m src.metrics eval data/labels_test_v4.jsonl --contre {out}")


def load_config(choix=CHOIX):
    """Configuration choisie par `grid`, prête pour build_pipeline."""
    if not choix.exists():
        raise SystemExit("Lance d'abord `python -m src.tfidf_lr grid`.")
    cfg = json.loads(choix.read_text(encoding="utf-8"))["config"]
    cfg["ngram_range"] = tuple(cfg["ngram_range"])          # JSON ne connaît que les listes
    return cfg


def print_contributions(contrib, k=3):
    for m, d in contrib.items():
        pour = ", ".join(f"{t} {c:+.2f}" for t, c in d["termes"][:k] if c > 0) or "-"
        contre = ", ".join(f"{t} {c:+.2f}" for t, c in d["termes"][::-1][:k] if c < 0) or "-"
        print(f"    {m:20s} p = {d['proba']:.2f}   pour : {pour}   |   contre : {contre}")


def run_explique(terms=TERMES_FRAICHEUR, texte=None, motif="fraicheur_securite", n_exemples=3, choix=CHOIX):
    """Ne lit jamais le test. Deux modèles, même configuration :
    - le modèle livré (entraînement + validation si REFIT) : poids des termes et contributions d'un texte libre ;
    - le modèle entraîné sur l'entraînement seul : confusions et exemples sur la validation, qu'il n'a pas vue.
    """
    cfg = load_config(choix)
    train, val = load_data()
    livre_df = pd.concat([train, val]) if REFIT else train
    livre = build_pipeline(**cfg).fit(build_text(livre_df), plainte_matrix(livre_df["motifs"]))
    if texte is not None:
        print(f"Modèle livré ({cfg}, {len(livre_df)} avis). Contributions (poids x TF-IDF) pour {texte!r} :")
        if any(texte.lower() in x.lower() for x in build_text(livre_df)):
            print("  Attention : ce texte figure dans les avis d'entraînement du modèle livré. Sa prédiction mesure "
                  "ce dont le modèle se souvient, pas ce qu'il sait généraliser.")
        print_contributions(contributions(livre, texte))
        return

    print(f"1. Poids des termes, modèle livré ({cfg}, {len(livre_df)} avis)")
    print(f"   {'terme':16s} " + " ".join(f"{m[:10]:>10s}" for m in MOTIFS_PLAINTE) + f" {'n_avis':>7s}")
    for t, row in term_weights(livre, terms, build_text(livre_df)).iterrows():
        if np.isnan(row["n_avis"]):
            print(f"   {t:16s} absent du vocabulaire (vu dans moins de {cfg['min_df']} avis)")
        else:
            print(f"   {t:16s} " + " ".join(f"{row[m]:>+10.2f}" for m in MOTIFS_PLAINTE) + f" {int(row['n_avis']):>7d}")

    modele = build_pipeline(**cfg).fit(build_text(train), plainte_matrix(train["motifs"]))
    X_val = build_text(val)
    P = modele.predict_proba(X_val)
    pred = to_motifs(P)
    ref = list(val["motifs"])
    c = confusions(ref, pred, motif)
    print(f"\n2. Confusions sur la validation ({len(val)} avis, étiquettes du professeur), modèle entraîné sur "
          f"{len(train)} avis : {motif}")
    print(f"   {c['positifs']} positifs, {c['trouves']} trouvés, {c['manques']} manqués, {c['en_trop']} en trop")
    print("   prédit à la place sur les manqués : "
          + (", ".join(f"{'+'.join(k)} {v}" for k, v in c["a_la_place"].most_common()) or "-"))
    manques = [i for i, (r, p) in enumerate(zip(ref, pred)) if motif in r and motif not in p]
    if manques and motif in MOTIFS_PLAINTE:
        print(f"   probabilité de {motif} sur les manqués : médiane "
              f"{np.median(P[manques, MOTIFS_PLAINTE.index(motif)]):.2f} (seuil {THRESHOLD})")

    print(f"\n3. Exemples de manqués sur la validation ({min(n_exemples, len(manques))} sur {len(manques)})")
    for i in manques[:n_exemples]:
        print(f"\n   {X_val[i][:150]!r}")
        print(f"   professeur : {ref[i]} ; modèle : {pred[i]}")
        print_contributions(contributions(modele, X_val[i]))


def main():
    parser = argparse.ArgumentParser(description="TF-IDF + régression logistique (cours 3)")
    parser.add_argument("cmd", choices=["grid", "final", "explique"])
    parser.add_argument("--texte", default=None, help="explique : contributions des termes pour ce texte")
    parser.add_argument("--termes", nargs="+", default=TERMES_FRAICHEUR, help="explique : termes du tableau des poids")
    parser.add_argument("--motif", default="fraicheur_securite", choices=MOTIFS, help="explique : motif des confusions")
    parser.add_argument("--exemples", type=int, default=3, help="explique : nombre d'avis manqués affichés")
    args = parser.parse_args()
    if args.cmd == "grid":
        run_grid(*load_data())
    elif args.cmd == "final":
        run_final()
    else:
        run_explique(args.termes, args.texte, args.motif, args.exemples)


if __name__ == "__main__":
    main()
