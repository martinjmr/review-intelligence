"""Tests de l'étape 4 : TF-IDF + régression logistique. Lance-les avec :  python -m pytest -q
Données jouets, sauf le dernier test, qui vérifie le découpage réel contre CONVENTIONS.md
(sauté si data/pool.parquet ou data/labels_train_v4.jsonl est absent).
"""
import json
import warnings
from collections import Counter

import numpy as np
import pytest
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.label_llm import POOL, check_rules
from src.metrics import MOTIFS
from src.tfidf_lr import (LABELS, MOTIFS_PLAINTE, build_pipeline, build_text, confusions, contributions, load_data,
                          load_train_labels, plainte_matrix, run_final, split_train_val, term_weights, to_motifs,
                          top_words)

TOY = [
    ("box arrived broken and melted", ["reception"]),
    ("arrived crushed box damaged", ["reception"]),
    ("tastes awful bland texture", ["gout_texture"]),
    ("awful taste weird texture", ["gout_texture"]),
    ("expired stale moldy", ["fraicheur_securite"]),
    ("stale expired rancid", ["fraicheur_securite"]),
    ("wrong item not as described", ["non_conforme"]),
    ("wrong size not as described", ["non_conforme"]),
    ("too expensive overpriced", ["prix"]),
    ("overpriced price too high", ["prix"]),
    ("arrived melted and too expensive", ["reception", "prix"]),
    ("fine product no complaint", ["autre"]),
]
TEXTS = [t for t, _ in TOY]
Y5 = plainte_matrix([m for _, m in TOY])


# ---------- 1. load_train_labels ----------

def test_load_train_labels_ne_garde_que_les_etiquettes_valides(tmp_path):
    lines = [
        {"review_id": "a", "motifs": ["prix"], "problems": []},
        {"review_id": "b", "error": "429"},
        {"review_id": "c", "motifs": ["autre", "gout_texture"], "problems": ["autre_combine"]},
        {"review_id": "d", "motifs": ["reception", "prix"], "problems": []},
        {"review_id": "e", "error": "timeout"},
        {"review_id": "e", "motifs": ["gout_texture"], "problems": []},
        {"review_id": "f", "motifs": [], "problems": ["liste_vide"]},
    ]
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n\n", encoding="utf-8")
    assert load_train_labels(path) == {"a": ["prix"], "d": ["reception", "prix"], "e": ["gout_texture"]}


# ---------- 2. split_train_val ----------

IDS = [f"r{i:03d}" for i in range(50)]


def test_split_train_val_partition_et_tailles():
    train, val = split_train_val(IDS, n_val=10, seed=42)
    assert len(val) == 10 and len(train) == 40
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(IDS)


def test_split_train_val_ne_depend_pas_de_l_ordre_des_ids():
    a = split_train_val(IDS, n_val=10, seed=42)
    b = split_train_val(IDS[::-1], n_val=10, seed=42)
    assert set(a[1]) == set(b[1])


def test_split_train_val_suit_la_recette_de_conventions():
    s = sorted(IDS)
    attendu = {s[i] for i in np.random.default_rng(42).permutation(len(s))[:10]}
    assert set(split_train_val(IDS, n_val=10, seed=42)[1]) == attendu
    assert set(split_train_val(IDS, n_val=10, seed=7)[1]) != attendu        # la graine compte


# ---------- 3. build_text ----------

def test_build_text_titre_puis_texte_et_titre_absent():
    df = pd.DataFrame({"title": ["Arrived melted", None, np.nan], "text": ["all stuck together", "too sweet", "stale"]})
    out = build_text(df)
    assert len(out) == 3
    assert [t.split() for t in out] == [["Arrived", "melted", "all", "stuck", "together"], ["too", "sweet"], ["stale"]]
    assert "None" not in out[1] and "nan" not in out[2]


# ---------- 4. build_pipeline ----------

def test_build_pipeline_assemble_les_bons_reglages():
    pipe = build_pipeline(C=3, ngram_range=(1, 2), min_df=2)
    assert isinstance(pipe, Pipeline) and len(pipe.steps) == 2
    vec, clf = pipe[0], pipe[-1]
    assert isinstance(vec, TfidfVectorizer)
    assert (vec.ngram_range, vec.min_df, vec.sublinear_tf, vec.stop_words) == ((1, 2), 2, True, None)
    assert isinstance(clf, OneVsRestClassifier) and isinstance(clf.estimator, LogisticRegression)
    assert (clf.estimator.C, clf.estimator.class_weight) == (3, "balanced")
    assert clf.estimator.max_iter >= 1000


def test_build_pipeline_s_entraine_sans_avertissement_et_sort_5_probabilites():
    pipe = build_pipeline(C=10, ngram_range=(1, 2), min_df=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)         # `penalty` passé alors qu'il est déprécié
        warnings.simplefilter("error", ConvergenceWarning)    # max_iter trop petit
        pipe.fit(TEXTS, Y5)
    P = pipe.predict_proba(["arrived melted", "fine product"])
    assert P.shape == (2, 5)
    assert np.all((P >= 0) & (P <= 1))


def test_le_vocabulaire_vient_du_seul_entrainement():
    pipe = build_pipeline(C=1, ngram_range=(1, 1), min_df=1).fit(TEXTS, Y5)
    vocab = dict(pipe[0].vocabulary_)
    pipe.predict_proba(["zzzinconnu arrived"])                # prédire n'apprend rien
    assert "zzzinconnu" not in pipe[0].vocabulary_
    assert pipe[0].vocabulary_ == vocab


# ---------- 5. to_motifs ----------

def test_to_motifs_seuil_et_regle_autre():
    proba = np.array([[0.9, 0.1, 0.2, 0.1, 0.6],
                      [0.1, 0.2, 0.3, 0.4, 0.49],
                      [0.5, 0.0, 0.0, 0.0, 0.0]])
    assert to_motifs(proba) == [["reception", "prix"], ["autre"], ["reception"]]     # 0,5 pile : retenu
    assert to_motifs(proba, threshold=0.3)[1] == ["fraicheur_securite", "non_conforme", "prix"]


def test_to_motifs_sorties_toujours_valides_et_dans_l_ordre():
    proba = np.random.default_rng(0).random((200, 5))
    out = to_motifs(proba)
    assert len(out) == 200
    assert all(check_rules(m) == [] for m in out)
    assert all(m == sorted(m, key=MOTIFS.index) for m in out)
    json.dumps(out)                                           # sérialisable tel quel dans un JSONL


def test_to_motifs_refuse_une_matrice_a_6_colonnes():
    with pytest.raises(ValueError):
        to_motifs(np.full((3, 6), 0.5))


# ---------- 6. top_words ----------

def test_top_words_trouve_les_marqueurs_d_un_motif():
    pipe = build_pipeline(C=10, ngram_range=(1, 1), min_df=1).fit(TEXTS, Y5)
    top = top_words(pipe, "fraicheur_securite", 3)
    assert len(top) == 3 and all(isinstance(w, str) for w in top)
    assert {"expired", "stale"} <= set(top)


def test_top_words_trie_du_plus_lourd_au_moins_lourd():
    pipe = build_pipeline(C=10, ngram_range=(1, 2), min_df=1).fit(TEXTS, Y5)
    vocab = list(pipe[0].get_feature_names_out())
    w = pipe[-1].estimators_[MOTIFS_PLAINTE.index("prix")].coef_[0]
    top = top_words(pipe, "prix", 5)
    poids = [w[vocab.index(t)] for t in top]
    assert poids == sorted(poids, reverse=True)
    assert min(poids) >= np.sort(w)[-5]                       # ce sont bien les 5 plus lourds


def test_top_words_refuse_autre():
    pipe = build_pipeline(C=1, ngram_range=(1, 1), min_df=1).fit(TEXTS, Y5)
    with pytest.raises(ValueError):
        top_words(pipe, "autre")


# ---------- Expliquer les prédictions ----------

def test_term_weights_poids_et_nombre_d_avis():
    pipe = build_pipeline(C=10, ngram_range=(1, 1), min_df=1).fit(TEXTS, Y5)
    tw = term_weights(pipe, ["melted", "Arrived", "zzzinconnu"], TEXTS)
    assert list(tw.columns) == MOTIFS_PLAINTE + ["n_avis"]
    assert tw.loc["melted", MOTIFS_PLAINTE].idxmax() == "reception"     # « melted » pousse surtout vers reception
    assert tw.loc["Arrived", "n_avis"] == 3                               # majuscule ignorée ; 3 textes jouets
    assert tw.loc["zzzinconnu"].isna().all()


def test_contributions_redonnent_la_probabilite_du_modele():
    pipe = build_pipeline(C=10, ngram_range=(1, 2), min_df=1).fit(TEXTS, Y5)
    texte = "arrived melted, tastes awful and zzzinconnu"
    contrib = contributions(pipe, texte)
    P = pipe.predict_proba([texte])[0]
    for j, m in enumerate(MOTIFS_PLAINTE):
        d = contrib[m]
        z = sum(c for _, c in d["termes"]) + d["intercept"]
        assert 1 / (1 + np.exp(-z)) == pytest.approx(P[j])                 # sigmoïde(somme des votes + biais)
        assert d["proba"] == pytest.approx(P[j])
        cs = [c for _, c in d["termes"]]
        assert cs == sorted(cs, reverse=True)
    termes = [t for t, _ in contrib["reception"]["termes"]]
    assert "melted" in termes and "zzzinconnu" not in termes


def test_confusions_compte_manques_et_remplacements():
    ref = [["fraicheur_securite"], ["fraicheur_securite", "gout_texture"], ["fraicheur_securite"], ["prix"], ["autre"]]
    pred = [["gout_texture"], ["fraicheur_securite"], ["autre"], ["fraicheur_securite", "prix"], ["autre"]]
    c = confusions(ref, pred, "fraicheur_securite")
    assert (c["positifs"], c["trouves"], c["manques"], c["en_trop"]) == (3, 1, 2, 1)
    assert c["a_la_place"] == Counter({("gout_texture",): 1, ("autre",): 1})


# ---------- Garde-fous de la mesure finale ----------

def test_final_refuse_si_les_predictions_du_test_existent(tmp_path):
    out = tmp_path / "preds.jsonl"
    out.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="une seule fois"):
        run_final(out=out, refit=False)


def test_final_refuse_tant_que_le_reentrainement_n_est_pas_tranche(tmp_path):
    with pytest.raises(SystemExit, match="REFIT"):
        run_final(out=tmp_path / "preds.jsonl", refit=None)


# ---------- Le vrai découpage, contre CONVENTIONS.md ----------

@pytest.mark.skipif(not (POOL.exists() and LABELS.exists()), reason="données locales absentes")
def test_decoupage_reel_conforme_a_conventions():
    train, val = load_data()
    assert (len(train), len(val)) == (3983, 996)
    positifs = {m: sum(m in x for x in val["motifs"]) for m in MOTIFS}
    assert positifs == {"reception": 104, "gout_texture": 571, "fraicheur_securite": 159,
                        "non_conforme": 84, "prix": 150, "autre": 89}
