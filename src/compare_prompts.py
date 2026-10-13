"""Étape 3 : comparer deux prompts sur explo (critères d'acceptation de v3, CONVENTIONS.md, section Prompt v3).

Usage :  python -m src.compare_prompts v2 v3
  1. critère 1 : score de chaque prompt sur les 29 avis d'explo tranchés dans le guide (hors exemples du prompt) ;
  2. avis où les deux prompts diffèrent, hors guide et hors exemples : seul leur nombre est affiché (pas leurs
     étiquettes, pour que l'annotation reste à l'aveugle) ; leurs identifiants sont écrits dans
     data/divergences_explo_<a>_<b>.txt, à annoter avec  python -m src.annotate --part explo --ids <ce fichier> ;
  3. critère 3, une fois ces avis annotés (relancer la même commande) : qui a raison le plus souvent, test du signe.
Le critère 2 (invalides < 2 %) est donné par  python -m src.report data/labels_explo_<b>.jsonl.
"""
import argparse
import json
from math import comb
from pathlib import Path

from src.annotate import KEYS, load_annotations, load_test

# Avis d'explo (n° 1 à 200, numérotation du journal) tranchés dans le guide v1, hors exemples du prompt v3.
GUIDE_EXACT = {24: "a", 44: "g", 50: "rf", 58: "a", 66: "rf", 86: "a", 87: "rg", 112: "a",
               146: "a", 147: "rgp", 152: "fg", 160: "f", 183: "a"}            # étiquette complète
GUIDE_REQUIRED = {7: "f", 11: "f", 19: "g", 22: "f", 26: "n", 38: "r", 65: "g", 75: "r",
                  92: "f", 96: "f", 101: "r", 129: "f", 134: "f", 157: "f", 180: "r", 181: "r"}  # motif présent
EXAMPLES_V3 = {61, 64, 149, 163, 169, 178}     # dans les prompts v3 et v4 : exclus de toute mesure
MIN_CRITERE_1 = 22                             # sur 29, fixé le 13 oct. 2026 avant tout appel v3
ANNOTATIONS = Path("data/annotations_explo.jsonl")
ORDER = "rfngpa"                               # ordre d'affichage des lettres (celui de l'arbre du guide)


def to_set(letters):
    return {KEYS[c] for c in letters}


def letters(motifs):
    return "".join(c for c in ORDER if KEYS[c] in motifs) or "-"


def load_motifs(path):
    """{review_id: ensemble de motifs} ; lignes en erreur ignorées, la dernière réussite l'emporte."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    out[r["review_id"]] = set(r["motifs"])
    return out


def guide_ok(n, motifs):
    """L'étiquette respecte-t-elle la décision du guide pour l'avis n ?"""
    if n in GUIDE_EXACT:
        return motifs == to_set(GUIDE_EXACT[n])
    return to_set(GUIDE_REQUIRED[n]) <= motifs


def sign_test(wins_a, wins_b):
    """p-valeur bilatérale exacte du test du signe : sous H0, chaque avis gagné l'est par A ou B à pile ou face."""
    n = wins_a + wins_b
    k = min(wins_a, wins_b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a", help="prompt de référence, ex. v2")
    parser.add_argument("b", help="prompt candidat, ex. v3")
    args = parser.parse_args()
    a, b = args.a, args.b

    explo = load_test(part="explo")                       # même ordre que le journal : n° = position + 1
    ids = dict(enumerate(explo["review_id"], 1))
    lab = {v: load_motifs(f"data/labels_explo_{v}.jsonl") for v in (a, b)}
    for v in (a, b):
        print(f"{v} : {len(lab[v])}/{len(ids)} avis étiquetés sans erreur")
    if any(len(lab[v]) < len(ids) for v in (a, b)):
        print("  attention : étiquetage incomplet, les résultats ci-dessous sont partiels")

    # Critère 1 : les cas tranchés dans le guide. Avis manquant = liste vide = échec.
    cases = sorted(set(GUIDE_EXACT) | set(GUIDE_REQUIRED))
    print(f"\nCritère 1 : {len(cases)} avis tranchés dans le guide (hors exemples) ; "
          f"exact = étiquette identique, doit = motif présent")
    score = {a: 0, b: 0}
    for n in cases:
        cells = []
        for v in (a, b):
            m = lab[v].get(ids[n], set())
            ok = guide_ok(n, m)
            score[v] += ok
            cells.append(f"{v} {letters(m):4s} {'ok' if ok else 'XX'}")
        rule = f"exact {GUIDE_EXACT[n]}" if n in GUIDE_EXACT else f"doit {GUIDE_REQUIRED[n]}"
        print(f"  n° {n:3d}  {rule:9s}  " + "   ".join(cells))
    print(f"Score : {a} {score[a]}/{len(cases)}, {b} {score[b]}/{len(cases)} -> critère 1 "
          + ("rempli" if score[b] >= MIN_CRITERE_1 else f"NON rempli (seuil {MIN_CRITERE_1})"))

    # Avis divergents, hors guide et hors exemples : ceux que Martin annote à l'aveugle.
    excluded = set(cases) | EXAMPLES_V3
    div = [ids[n] for n in ids if n not in excluded
           and ids[n] in lab[a] and ids[n] in lab[b] and lab[a][ids[n]] != lab[b][ids[n]]]
    div_path = Path(f"data/divergences_explo_{a}_{b}.txt")
    div_path.write_text("\n".join(div) + "\n", encoding="utf-8")
    print(f"\n{len(div)} avis où {a} et {b} diffèrent (hors guide et exemples) -> {div_path}")

    # Critère 3 : sur ces avis, qui a raison selon l'annotation de Martin (étiquette identique) ?
    ann = load_annotations(ANNOTATIONS)
    gold = {rid: set(r["motifs"]) for (rid, rnd), r in ann.items() if rnd == 1}
    done = [rid for rid in div if rid in gold]
    if not done:
        print(f"Critère 3 : aucun de ces avis n'est annoté. Annote-les d'abord :\n"
              f"  python -m src.annotate --part explo --ids {div_path}")
        return
    num = {rid: n for n, rid in ids.items()}
    wins = {a: 0, b: 0}
    print(f"\nCritère 3 : {len(done)}/{len(div)} avis annotés")
    for rid in done:
        winner = next((v for v in (a, b) if lab[v][rid] == gold[rid]), None)   # au plus un : a et b diffèrent
        if winner:
            wins[winner] += 1
        print(f"  n° {num[rid]:3d}  toi {letters(gold[rid]):4s}  {a} {letters(lab[a][rid]):4s}  "
              f"{b} {letters(lab[b][rid]):4s}  -> {winner or 'les deux faux'}")
    both_wrong = len(done) - wins[a] - wins[b]
    print(f"{b} juste {wins[b]}, {a} juste {wins[a]}, les deux faux {both_wrong} ; "
          f"test du signe p = {sign_test(wins[a], wins[b]):.3f} -> critère 3 "
          + ("rempli" if wins[b] >= wins[a] else "NON rempli"))
    if len(done) < len(div):
        print("  (partiel : relance après avoir annoté les avis restants)")


if __name__ == "__main__":
    main()
