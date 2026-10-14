"""Étape 2 : annoter à la main les 300 avis de test, à l'aveugle (cours 2, sections 3 et 4).

Usage :
  python -m src.annotate --part explo            # échauffement sur explo (q pour arrêter quand tu veux)
  python -m src.annotate                         # passe 1 : les 300 avis de test, reprise automatique
  python -m src.annotate --revoir doutes         # fin de passe 1 : trancher les avis marqués « ! »
  python -m src.annotate --corriger "mot"        # revoir les avis déjà annotés dont le titre contient « mot »
  python -m src.annotate --round 2               # passe 2, au moins 3 jours plus tard : 50 avis tirés au hasard
  python -m src.annotate --revoir desaccords     # après la mesure d'accord : trancher les écarts entre passes
  python -m src.annotate --part explo --ids F    # n'annoter que les avis listés dans F (un review_id par ligne)
Produit : data/annotations_<part>.jsonl (une ligne par réponse ; pour un même avis et une même passe,
la dernière ligne l'emporte). Passes : 1 = première annotation, 2 = ré-annotation, 3 = arbitrage des désaccords.
"""
import argparse
import json
import random
import statistics
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import get_args

import pandas as pd

from src.label_llm import Motif, check_rules

POOL = Path("data/pool.parquet")
OUT = Path("data/annotations_test.jsonl")
GUIDE_VERSION = "v1"      # à changer à chaque modification du guide (docs/guide_annotation.md)
N_REANNOT = 50            # avis ré-annotés en passe 2 (décision du 13 oct. 2026 ; cours 2, section 5)
MIN_DAYS = 3              # délai minimal entre passe 1 et passe 2, pour oublier ses réponses

MOTIFS = get_args(Motif)
KEYS = {"r": "reception", "g": "gout_texture", "f": "fraicheur_securite",
        "n": "non_conforme", "p": "prix", "a": "autre"}
DOUTE = "!"
RETOUR = "<"
AIDE = ("  r=reception  g=gout_texture  f=fraicheur_securite  n=non_conforme  p=prix  a=autre\n"
        "  plusieurs lettres = plusieurs motifs (ex. rg) ; ajoute ! si tu hésites (ex. g!) ;\n"
        "  < = revenir à l'avis précédent (la nouvelle réponse remplace l'ancienne) ;\n"
        "  ? = cette aide ; q = quitter (tout est déjà enregistré)")


def load_test(pool_path=POOL, part="test"):
    """Les avis d'une part (test par défaut), dans l'ordre du réservoir (déjà mélangé, graine 42).

    On n'affiche que le titre et le texte : exactement ce que voient les modèles (pas la note).
    """
    df = pd.read_parquet(pool_path)
    return df[df["part"] == part][["review_id", "title", "text"]].reset_index(drop=True)


def parse_answer(raw):
    """Traduit la saisie de l'annotateur en (motifs, doute).

    Exemples : "rg" -> (["reception", "gout_texture"], False) ; "f!" -> (["fraicheur_securite"], True).
    Lève ValueError si la saisie ne donne aucun motif, contient une touche inconnue
    ou viole une règle (les mêmes que pour le LLM : check_rules).
    """
    s = raw.lower().replace(" ", "")
    doute = DOUTE in s
    s = s.replace(DOUTE, "")
    for c in s:
        if c not in KEYS:
            raise ValueError(f"touche inconnue : {c!r}")
    motifs = [KEYS[c] for c in s]
    problems = check_rules(motifs)
    if problems:
        raise ValueError(", ".join(problems))
    return sorted(motifs, key=MOTIFS.index), doute


def load_annotations(path=OUT):
    """Dernière réponse pour chaque couple (review_id, round) : {(review_id, round): ligne (dict)}.

    Le fichier ne fait que grandir : corriger un avis, c'est écrire une nouvelle ligne,
    qui remplace la précédente à la lecture. Rien n'est jamais effacé.
    """
    path = Path(path)
    if not path.exists():
        return {}
    ann = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ann[(r["review_id"], r["round"])] = r
    return ann


def pick_reannotation(ids, n, seed=42):
    """n identifiants tirés au hasard parmi ids, sans remise, dans un ordre mélangé.

    Même graine -> même tirage : relancer la passe 2 reprend exactement les mêmes avis.
    """
    return random.Random(seed).sample(list(ids), n)


# ---------- Affichage et boucle de saisie ----------

def show(row, i, total, previous=None):
    print("\n" + "=" * 100)
    print(f"[{i}/{total}]  {row['title']}")
    print()
    print(textwrap.fill(str(row["text"]), width=100))
    if previous:                                   # seulement en arbitrage (--revoir)
        print()
        for label, ans in previous:
            note = f"  (note : {ans['note']})" if ans.get("note") else ""
            print(f"  {label} : {', '.join(ans['motifs'])}{note}")
    print("-" * 100)


def ask():
    """Redemande jusqu'à une saisie valide. Renvoie (motifs, doute), RETOUR, ou None pour quitter."""
    while True:
        raw = input("motifs> ").strip()
        if raw.lower() == "q":
            return None
        if raw == RETOUR:
            return RETOUR
        if raw == "?":
            print(AIDE)
            continue
        try:
            return parse_answer(raw)
        except ValueError as e:
            print(f"  refusé : {e}  (tape ? pour l'aide)")


def annotate(rows, round_, out_path=OUT, previous=None):
    """Affiche chaque avis, enregistre chaque réponse dès qu'elle est donnée (reprise possible)."""
    durations, n_doute = [], 0
    i = 0
    with out_path.open("a", encoding="utf-8") as f:
        while i < len(rows):
            row = rows[i]
            show(row, i + 1, len(rows), previous.get(row["review_id"]) if previous else None)
            t0 = time.perf_counter()
            answer = ask()
            if answer is None:
                print("Arrêt. Relance la même commande pour reprendre.")
                break
            if answer == RETOUR:
                if i == 0:
                    print("  pas d'avis précédent dans cette séance")
                i = max(i - 1, 0)
                continue
            motifs, doute = answer
            duree = time.perf_counter() - t0
            note = input("  note sur ton doute (Entrée pour passer)> ").strip() if doute else ""
            line = {
                "review_id": row["review_id"],
                "round": round_,
                "motifs": motifs,
                "doute": doute,
                "note": note,
                "duree_s": round(duree, 1),
                "guide_version": GUIDE_VERSION,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()                                    # écrit sur disque tout de suite
            durations.append(duree)
            n_doute += doute
            i += 1
    if durations:
        print(f"\n{len(durations)} réponses enregistrées, {n_doute} doutes, "
              f"durée médiane {statistics.median(durations):.0f} s par avis")


def print_shares(ann, round_=1):
    """Part des avis qui portent chaque motif, sur la passe donnée (même calcul que report.motif_shares)."""
    rows = [a["motifs"] for (rid, rnd), a in ann.items() if rnd == round_]
    if rows:
        counts = {m: sum(m in motifs for motifs in rows) for m in MOTIFS}
        print(f"Passe {round_}, {len(rows)} avis : "
              + ", ".join(f"{m} {c} ({c / len(rows):.0%})" for m, c in counts.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["test", "explo"], default="test")
    parser.add_argument("--round", type=int, choices=[1, 2], default=1)
    parser.add_argument("--revoir", choices=["doutes", "desaccords"], default=None)
    parser.add_argument("--corriger", default=None, help="revoir les avis de passe 1 dont le titre contient ce texte")
    parser.add_argument("--n", type=int, default=N_REANNOT, help="avis ré-annotés en passe 2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--voir", type=int, nargs="+", default=None,
                        help="afficher les avis d'explo n° ... (numérotation du journal : 1 à 200), puis quitter")
    parser.add_argument("--ids", default=None,
                        help="fichier de review_id (un par ligne) : n'annoter que ces avis (ex. divergences entre prompts)")
    args = parser.parse_args()

    if args.voir:
        explo = load_test(part="explo")
        for n in args.voir:
            show(explo.iloc[n - 1], n, len(explo))
        return

    out = OUT if args.part == "test" else Path(f"data/annotations_{args.part}.jsonl")
    test = load_test(part=args.part)
    by_id = {r["review_id"]: r for r in test.to_dict("records")}
    ann = load_annotations(out)
    done1 = {rid for (rid, rnd) in ann if rnd == 1}
    print(f"Part {args.part}, guide {GUIDE_VERSION} ; passe 1 : {len(done1)}/{len(test)} avis annotés")

    if args.revoir == "doutes":
        ids = [rid for (rid, rnd), a in ann.items() if rnd == 1 and a["doute"]]
        previous = {rid: [("ta réponse", ann[(rid, 1)])] for rid in ids}
        print(f"{len(ids)} avis marqués « ! » à trancher (la nouvelle réponse remplace l'ancienne)")
        annotate([by_id[r] for r in ids], 1, out, previous=previous)
    elif args.corriger:
        mot = args.corriger.lower()
        ids = [r["review_id"] for r in test.to_dict("records")
               if r["review_id"] in done1 and mot in str(r["title"]).lower()]
        previous = {rid: [("ta réponse", ann[(rid, 1)])] for rid in ids}
        print(f"{len(ids)} avis déjà annotés dont le titre contient « {args.corriger} »")
        annotate([by_id[r] for r in ids], 1, out, previous=previous)
    elif args.revoir == "desaccords":
        ids = [rid for (rid, rnd), a in ann.items()
               if rnd == 2 and set(a["motifs"]) != set(ann[(rid, 1)]["motifs"]) and (rid, 3) not in ann]
        previous = {rid: [("passe 1", ann[(rid, 1)]), ("passe 2", ann[(rid, 2)])] for rid in ids}
        print(f"{len(ids)} désaccords entre passes à arbitrer (passe 3).")
        print("Lance d'abord `python -m src.metrics accord` : l'arbitrage ne change pas cette mesure,")
        print("mais c'est elle qu'on veut garder.")
        annotate([by_id[r] for r in ids], 3, out, previous=previous)
    elif args.ids:
        wanted = [r for r in Path(args.ids).read_text(encoding="utf-8").split() if r in by_id]
        todo = [by_id[r] for r in wanted if r not in done1]
        print(f"{len(todo)} avis à faire sur {len(wanted)} listés dans {args.ids}. Tape ? pour l'aide.")
        annotate(todo, 1, out)
    elif args.round == 1:
        todo = [r for r in test.to_dict("records") if r["review_id"] not in done1]
        print(f"{len(todo)} avis à faire. Tape ? pour l'aide.")
        annotate(todo, 1, out)
        print_shares(load_annotations(out), 1)
    else:
        if len(done1) < len(test):
            print("La passe 1 n'est pas terminée : termine-la avant la passe 2.")
            return
        last = max(datetime.fromisoformat(a["ts"]) for (rid, rnd), a in ann.items() if rnd == 1)
        days = (datetime.now() - last).days
        if days < MIN_DAYS:
            print(f"Attention : passe 1 terminée il y a {days} jour(s). Attends {MIN_DAYS} jours "
                  "pour ne pas te souvenir de tes réponses.")
            if input("Continuer quand même ? (o/n) ").strip().lower() != "o":
                return
        ids = pick_reannotation(test["review_id"].tolist(), args.n, seed=args.seed)
        todo = [by_id[r] for r in ids if (r, 2) not in ann]
        print(f"Passe 2 : {len(ids) - len(todo)}/{len(ids)} faits, {len(todo)} à faire. "
              "Tes réponses de la passe 1 ne sont pas affichées.")
        annotate(todo, 2, out)


if __name__ == "__main__":
    main()
