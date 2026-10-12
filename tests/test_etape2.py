"""Tests de l'étape 2 (outil d'annotation) et des métriques (cours 2). Lance-les avec :  python -m pytest -q
Les métriques sont comparées à scikit-learn sur des données tirées au hasard : si tes chiffres
diffèrent de ceux de la bibliothèque, le test le dit.
"""
import json
import math

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support

from src.annotate import load_annotations, parse_answer, pick_reannotation
from src.metrics import (MOTIFS, bootstrap_ci, cohen_kappa, exact_match, kappa_par_motif, macro_f1,
                         paired_bootstrap, scores, to_matrix)


# ---------- annotate.py ----------

def test_parse_answer_traduit_et_range_dans_l_ordre_des_motifs():
    assert parse_answer("rg") == (["reception", "gout_texture"], False)
    assert parse_answer("gr") == (["reception", "gout_texture"], False)
    assert parse_answer(" P f ") == (["fraicheur_securite", "prix"], False)
    assert parse_answer("a") == (["autre"], False)


def test_parse_answer_doute():
    assert parse_answer("g!") == (["gout_texture"], True)
    assert parse_answer("!n p") == (["non_conforme", "prix"], True)


@pytest.mark.parametrize("raw", ["", "   ", "!", "x", "gx", "ag", "gg"])
def test_parse_answer_refuse_les_saisies_invalides(raw):
    with pytest.raises(ValueError):
        parse_answer(raw)


def test_load_annotations_la_derniere_ligne_l_emporte(tmp_path):
    lines = [
        {"review_id": "a", "round": 1, "motifs": ["prix"], "doute": True},
        {"review_id": "b", "round": 1, "motifs": ["autre"], "doute": False},
        {"review_id": "a", "round": 2, "motifs": ["reception"], "doute": False},
        {"review_id": "a", "round": 1, "motifs": ["gout_texture", "prix"], "doute": False},
    ]
    path = tmp_path / "ann.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n\n", encoding="utf-8")
    ann = load_annotations(path)
    assert set(ann) == {("a", 1), ("b", 1), ("a", 2)}
    assert ann[("a", 1)]["motifs"] == ["gout_texture", "prix"]
    assert ann[("a", 1)]["doute"] is False
    assert ann[("a", 2)]["motifs"] == ["reception"]


def test_load_annotations_fichier_absent(tmp_path):
    assert load_annotations(tmp_path / "absent.jsonl") == {}


def test_pick_reannotation_reproductible_et_melange():
    ids = [f"r{i}" for i in range(300)]
    a = pick_reannotation(ids, 100, seed=42)
    assert a == pick_reannotation(ids, 100, seed=42)              # même graine, même tirage
    assert len(a) == 100 and len(set(a)) == 100 and set(a) <= set(ids)
    assert a != ids[:100]                                          # pas les 100 premiers
    assert a != sorted(a, key=ids.index)                           # et pas dans l'ordre de la passe 1
    assert pick_reannotation(ids, 100, seed=1) != a


# ---------- metrics.py ----------

def _random_pair(seed, n=200):
    """Une fausse référence Y et une fausse prédiction P (environ 15 % de cases fausses)."""
    rng = np.random.default_rng(seed)
    Y = (rng.random((n, 6)) < [0.15, 0.55, 0.15, 0.12, 0.12, 0.07]).astype(int)
    P = np.where(rng.random((n, 6)) < 0.85, Y, 1 - Y)
    return Y, P


def test_to_matrix():
    Y = to_matrix([["prix"], ["autre"], ["reception", "gout_texture"]])
    assert MOTIFS == ["reception", "gout_texture", "fraicheur_securite", "non_conforme", "prix", "autre"]
    assert Y.shape == (3, 6)
    assert Y.tolist() == [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0]]


def test_cohen_kappa_exemple_du_cours():
    a = [1] * 8 + [0] * 42                       # passe 1 : 8 avis « prix » sur 50
    b = [1] * 6 + [0] * 2 + [1] + [0] * 41       # passe 2 : 6 en commun, 2 perdus, 1 ajouté
    assert cohen_kappa(a, b) == pytest.approx(0.7649, abs=1e-4)


def test_cohen_kappa_comme_scikit_learn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.integers(0, 3, 60)
        b = np.where(rng.random(60) < 0.7, a, rng.integers(0, 3, 60))
        assert cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))
    y1 = ["negative", "positive", "negative", "neutral", "positive"]
    y2 = ["negative", "positive", "negative", "neutral", "negative"]
    assert cohen_kappa(y1, y2) == pytest.approx(0.6875)          # exemple de la doc scikit-learn


def test_cohen_kappa_cas_limites():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)
    assert math.isnan(cohen_kappa([0, 0, 0], [0, 0, 0]))          # une seule étiquette : non défini


def test_kappa_par_motif():
    Y, P = _random_pair(3)
    k = kappa_par_motif(Y, P)
    assert list(k) == MOTIFS
    for j, m in enumerate(MOTIFS):
        assert k[m] == pytest.approx(cohen_kappa_score(Y[:, j], P[:, j]))


def test_scores_comme_scikit_learn():
    for seed in range(5):
        Y, P = _random_pair(seed)
        s = scores(Y, P)
        p, r, f, sup = precision_recall_fscore_support(Y, P, average=None, zero_division=0)
        for j, m in enumerate(MOTIFS):
            d = s["par_motif"][m]
            assert (d["p"], d["r"], d["f1"]) == pytest.approx((p[j], r[j], f[j]))
            assert d["support"] == sup[j]
        assert s["micro_f1"] == pytest.approx(f1_score(Y, P, average="micro", zero_division=0))
        assert s["macro_f1"] == pytest.approx(f1_score(Y, P, average="macro", zero_division=0))


def test_scores_division_par_zero():
    Y = to_matrix([["prix"], ["autre"]])
    P = to_matrix([["autre"], ["autre"]])
    s = scores(Y, P)
    assert (s["par_motif"]["prix"]["p"], s["par_motif"]["prix"]["r"], s["par_motif"]["prix"]["f1"]) == (0, 0, 0)
    assert s["par_motif"]["autre"]["p"] == pytest.approx(0.5)
    assert s["par_motif"]["autre"]["r"] == pytest.approx(1.0)
    assert s["par_motif"]["autre"]["f1"] == pytest.approx(2 / 3)
    assert s["par_motif"]["reception"]["f1"] == 0                  # absent des deux côtés : 0 par convention


def test_bootstrap_ci_retrouve_la_theorie_sur_une_proportion():
    # La part d'avis positifs, p = 0,5 sur 400 avis : la théorie donne un IC de largeur
    # 2 x 1,96 x racine(0,5 x 0,5 / 400) = 0,098.
    rng = np.random.default_rng(0)
    Y = (rng.random((400, 6)) < 0.5).astype(int)
    part = lambda Yt, Yp: float(Yt[:, 0].mean())
    est, low, high = bootstrap_ci(Y, Y, part, n_boot=2000, seed=42)
    assert est == pytest.approx(Y[:, 0].mean())
    assert low < est < high
    assert 0.08 < high - low < 0.12


def test_bootstrap_ci_tire_des_avis_entiers_et_les_memes_des_deux_cotes():
    Y, _ = _random_pair(4, n=100)
    # Si Y_true et Y_pred étaient tirés avec des indices différents, des avis seraient mélangés
    # et l'accord exact d'une prédiction parfaite tomberait sous 1.
    assert bootstrap_ci(Y, Y.copy(), exact_match, n_boot=200) == pytest.approx((1.0, 1.0, 1.0))


def test_bootstrap_ci_reproductible():
    Y, P = _random_pair(1)
    assert bootstrap_ci(Y, P, macro_f1, n_boot=200, seed=7) == bootstrap_ci(Y, P, macro_f1, n_boot=200, seed=7)
    assert bootstrap_ci(Y, P, macro_f1, n_boot=200, seed=7) != bootstrap_ci(Y, P, macro_f1, n_boot=200, seed=8)


def test_paired_bootstrap():
    Y, A = _random_pair(2)
    assert paired_bootstrap(Y, A, A.copy(), macro_f1, n_boot=200) == pytest.approx((0.0, 0.0, 0.0))
    diff, low, high = paired_bootstrap(Y, A, Y.copy(), macro_f1, n_boot=500)   # B parfait
    assert diff == pytest.approx(1 - macro_f1(Y, A))
    assert 0 < low <= diff <= high
