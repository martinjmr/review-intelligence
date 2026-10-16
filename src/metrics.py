"""Étapes 2 et 3 : mesurer un accord entre annotations et la qualité d'un classifieur.

Usage :
  python -m src.metrics accord                                    # ta cohérence : passe 1 contre passe 2
  python -m src.metrics eval data/labels_test_v3.jsonl            # un modèle contre ta référence (étape 3)
  python -m src.metrics eval data/labels_test_v3.jsonl --contre data/labels_test_autre.jsonl
                                                                  # deux modèles : bootstrap apparié
"""
import argparse
import json
from pathlib import Path
from typing import get_args

import numpy as np

from src.annotate import OUT, load_annotations
from src.label_llm import Motif

MOTIFS = list(get_args(Motif))
N_BOOT = 2000


def to_matrix(labels):
    """Liste de listes de motifs -> matrice 0/1 (numpy, entiers) de forme (n_avis, 6).

    Colonnes dans l'ordre de MOTIFS : la colonne j vaut 1 si l'avis porte le motif MOTIFS[j].
    """
    Y = np.zeros((len(labels), len(MOTIFS)), dtype=int)
    for i, motifs in enumerate(labels):
        for m in motifs:
            Y[i, MOTIFS.index(m)] = 1
    return Y


def cohen_kappa(a, b):
    """Kappa de Cohen entre deux séquences d'étiquettes de même longueur (2 classes ou plus).

    kappa = (p_o - p_e) / (1 - p_e). Renvoie float("nan") si p_e = 1 (kappa non défini :
    les deux annotateurs n'ont utilisé qu'une seule et même étiquette).
    """
    a, b = np.asarray(a), np.asarray(b)
    p_o = np.mean(a == b)
    p_e = sum(np.mean(a == k) * np.mean(b == k) for k in set(a.tolist()) | set(b.tolist()))
    if p_e == 1:
        return float("nan")
    return float((p_o - p_e) / (1 - p_e))


def kappa_par_motif(Y1, Y2):
    """{motif: kappa} entre deux matrices 0/1 : chaque colonne est un problème oui/non à part."""
    return {m: cohen_kappa(Y1[:, j], Y2[:, j]) for j, m in enumerate(MOTIFS)}


def scores(Y_true, Y_pred):
    """Précision, rappel, F1 et support par motif, plus micro-F1 et macro-F1.

    Renvoie {"par_motif": {motif: {"p", "r", "f1", "support"}}, "micro_f1": float, "macro_f1": float}.
    Convention : toute division par zéro donne 0 (comme zero_division=0 dans scikit-learn).
    """
    def div(a, b):
        return float(a / b) if b else 0.0

    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    tp = ((Y_true == 1) & (Y_pred == 1)).sum(axis=0)
    fp = ((Y_true == 0) & (Y_pred == 1)).sum(axis=0)
    fn = ((Y_true == 1) & (Y_pred == 0)).sum(axis=0)
    par_motif = {}
    for j, m in enumerate(MOTIFS):
        par_motif[m] = {
            "p": div(tp[j], tp[j] + fp[j]),
            "r": div(tp[j], tp[j] + fn[j]),
            "f1": div(2 * tp[j], 2 * tp[j] + fp[j] + fn[j]),
            "support": int(tp[j] + fn[j]),
        }
    micro = div(2 * tp.sum(), 2 * tp.sum() + fp.sum() + fn.sum())
    macro = float(np.mean([par_motif[m]["f1"] for m in MOTIFS]))
    return {"par_motif": par_motif, "micro_f1": micro, "macro_f1": macro}


def bootstrap_ci(Y_true, Y_pred, metric, n_boot=N_BOOT, seed=42, alpha=0.05):
    """(valeur sur le jeu complet, borne basse, borne haute) : intervalle percentile à 1 - alpha.

    metric(Y_true, Y_pred) -> float. On tire des AVIS (des lignes) avec remise.
    Les tirages où la métrique vaut nan sont ignorés (np.nanpercentile).
    """
    rng = np.random.default_rng(seed)
    n = len(Y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        values.append(metric(Y_true[idx], Y_pred[idx]))
    low, high = np.nanpercentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return metric(Y_true, Y_pred), float(low), float(high)


def paired_bootstrap(Y_true, P_a, P_b, metric, n_boot=N_BOOT, seed=42, alpha=0.05):
    """Écart metric(B) - metric(A) sur le jeu complet, avec son intervalle percentile.

    Apparié : à chaque tirage, A et B sont évalués sur les MÊMES avis.
    Renvoie (écart, borne basse, borne haute).
    """
    rng = np.random.default_rng(seed)
    n = len(Y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(metric(Y_true[idx], P_b[idx]) - metric(Y_true[idx], P_a[idx]))
    low, high = np.nanpercentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return metric(Y_true, P_b) - metric(Y_true, P_a), float(low), float(high)


# ---------- Chargement, rapports et ligne de commande ----------

def macro_f1(Y_true, Y_pred):
    return scores(Y_true, Y_pred)["macro_f1"]


def exact_match(Y_true, Y_pred):
    """Part des avis dont TOUS les motifs sont justes (ni oubli, ni ajout)."""
    return float(np.mean(np.all(Y_true == Y_pred, axis=1)))


def reference(ann):
    """Ta référence : l'arbitrage (passe 3) s'il existe, sinon la passe 1. {review_id: motifs}."""
    ref = {rid: a["motifs"] for (rid, rnd), a in ann.items() if rnd == 1}
    ref.update({rid: a["motifs"] for (rid, rnd), a in ann.items() if rnd == 3})
    return ref


def load_predictions(path):
    """Dernière réponse réussie du modèle pour chaque avis : {review_id: motifs}."""
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    preds[r["review_id"]] = r["motifs"] or []
    return preds


def _fmt_ci(est, low, high):
    return f"{est:.3f} [{low:.3f}, {high:.3f}]"


def report_accord(ann):
    ids = [rid for (rid, rnd) in ann if rnd == 2]
    if not ids:
        print("Aucune ré-annotation (passe 2) : lance d'abord `python -m src.annotate --round 2`.")
        return
    Y1 = to_matrix([ann[(rid, 1)]["motifs"] for rid in ids])
    Y2 = to_matrix([ann[(rid, 2)]["motifs"] for rid in ids])
    print(f"Passe 1 contre passe 2 : {len(ids)} avis")
    print("Accord exact (tous les motifs identiques) :", _fmt_ci(*bootstrap_ci(Y1, Y2, exact_match)))
    print("Kappa par motif, IC 95 % (positifs passe 1 / passe 2) :")
    for j, m in enumerate(MOTIFS):
        k = lambda A, B, j=j: cohen_kappa(A[:, j], B[:, j])
        print(f"  {m:20s} {_fmt_ci(*bootstrap_ci(Y1, Y2, k))}   ({Y1[:, j].sum()} / {Y2[:, j].sum()})")
    print("Avis en désaccord :", int((~np.all(Y1 == Y2, axis=1)).sum()))


def report_eval(ann, path, other=None):
    ref = reference(ann)
    ids = sorted(ref)
    preds = load_predictions(path)
    missing = [rid for rid in ids if rid not in preds]
    print(f"Référence : {len(ids)} avis annotés ; modèle : {path} ({len(missing)} avis sans réponse, "
          "comptés comme liste vide)")
    Y = to_matrix([ref[rid] for rid in ids])
    P = to_matrix([preds.get(rid, []) for rid in ids])
    s = scores(Y, P)
    print(f"{'motif':20s} {'P':>6s} {'R':>6s} {'F1':>6s} {'support':>8s}   F1 IC 95 %")
    for j, m in enumerate(MOTIFS):
        d = s["par_motif"][m]
        f1_j = lambda A, B, m=m: scores(A, B)["par_motif"][m]["f1"]
        _, lo, hi = bootstrap_ci(Y, P, f1_j)
        print(f"{m:20s} {d['p']:6.3f} {d['r']:6.3f} {d['f1']:6.3f} {d['support']:8d}   [{lo:.3f}, {hi:.3f}]")
    print("micro-F1    :", _fmt_ci(*bootstrap_ci(Y, P, lambda A, B: scores(A, B)["micro_f1"])))
    print("macro-F1    :", _fmt_ci(*bootstrap_ci(Y, P, macro_f1)))
    print("accord exact:", _fmt_ci(*bootstrap_ci(Y, P, exact_match)))
    print("Kappa toi / modèle par motif (à comparer à ton accord avec toi-même) :")
    print("  " + ", ".join(f"{m} {k:.2f}" for m, k in kappa_par_motif(Y, P).items()))
    if other:
        preds_b = load_predictions(other)
        B = to_matrix([preds_b.get(rid, []) for rid in ids])
        print(f"\nComparaison appariée : B = {other} contre A = {path}")
        for name, metric in [("macro-F1", macro_f1), ("accord exact", exact_match)]:
            print(f"  écart {name} (B - A) :", _fmt_ci(*paired_bootstrap(Y, P, B, metric)))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("accord")
    ev = sub.add_parser("eval")
    ev.add_argument("path")
    ev.add_argument("--contre", default=None)
    parser.add_argument("--annotations", default=str(OUT))
    args = parser.parse_args()
    ann = load_annotations(Path(args.annotations))
    if args.cmd == "accord":
        report_accord(ann)
    else:
        report_eval(ann, args.path, args.contre)


if __name__ == "__main__":
    main()
