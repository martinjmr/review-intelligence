"""Jalon 2 : LLM-juge. Noter les réponses du RAG avec une grille de 3 critères oui / non, calibrer le juge sur les
notes de Martin, puis comparer deux modes de génération (A = rag, B = plein) sur les mêmes avis.

Usage :
  python -m src.juge noter --part rag_dev --n 20               # tes notes, AVANT de lancer le juge (terminal)
  python -m src.juge page --part rag_dev --n 20                # ou : page HTML de notation, hors du terminal
  python -m src.juge juger --part rag_dev --mode rag --limit 20
  python -m src.juge calibrer --part rag_dev                   # matrice, métriques, désaccords
  python -m src.juge juger --part rag_dev --modele mistral-large-4-0 --limit 20   # autre juge, autre fichier
  python -m src.juge juger --part rag_eval --mode rag --limit 50
  python -m src.juge juger --part rag_eval --mode plein --limit 50
  python -m src.juge bilan --part rag_dev --contre rag_v4      # action conforme : prompt actuel contre v4, apparié
  python -m src.juge comparer --part rag_eval                  # A contre B, sur les avis jugés dans les deux modes
Lit : data/rag/reponses_<part>_<mode>_<prompt RAG>.jsonl (écrits par python -m src.rag repondre).
Produit : data/rag/notes_martin_<part>.jsonl, data/rag/jugements_<part>_<mode>_<prompt juge>.jsonl.
"""
import argparse
import json
import math
import time
from math import comb
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from src.label_llm import call_with_retry, load_prompt
from src.rag import (DOSSIER, MARQUE, PROMPT_RAG, charger_corpus, charger_part, client_mistral, ic_bootstrap)

JUGE_MODEL = "mistral-large-2512"   # Large 3, version datée : différent du générateur (mistral-small-2603)
PROMPT_JUGE = "juge_v2"   # v2 (23 oct. 2026) : ton réécrit, citation hors sujet = non fondée ; v1 : ton refusé
PAUSE_JUGE_S = 25.0   # ~8 000 tokens par jugement ; à 20 000 tokens/min -> 24 s. À ajuster aux limites du compte
# Large 4 réfléchit par défaut (un ThinkChunk avant la réponse, qui épuisait les 400 tokens) : on le coupe pour juger
# dans les mêmes conditions que Large 3, la `raison` tenant lieu de raisonnement. Large 3 ne connaît pas ce paramètre.
REGLAGES_MODELE = {"mistral-large-4-0": {"reasoning_effort": "none"}}
CRITERES = ("action_conforme", "fondee", "ton")
SEUIL_RAPPEL = 0.80     # règle d'acceptation du juge, appliquée critère par critère (23 oct. 2026, avant le juge) :
SEUIL_PRECISION = 0.67  # sur le verdict global, 18 fautes sur 20 laissaient passer un juge qui dit « fautive » à tout
SEUIL_KAPPA = 0.40      # sur « ton » (14 non sur 20), dire « non » à tout donne déjà une précision de 0,70 ; kappa 0
MIN_PAR_CLASSE = 5      # au moins 5 « non » et 5 « oui » de ta part par critère, sinon noter 10 réponses de plus
GABARIT_PAGE = Path(__file__).with_name("gabarit_notation.html")


class Jugement(BaseModel):
    raison: str               # généré en 1er : ce que l'avis demande, la règle qui s'applique, ce que fait la réponse
    action_conforme: bool
    fondee: bool
    ton: bool


# ---------- Messages du juge ----------

def construire_messages_juge(grille, passages, avis, reponse):
    """Messages de l'appel au juge.

    Message système : la grille, puis toute la politique, identiques à chaque appel (préfixe mis en cache).
        {grille}

        <policy>
        [P01] Titre: texte
        ...
        </policy>
    Message utilisateur, au caractère près :
        <review>
        Rating: {rating}
        Title: {title}
        Text: {text}
        </review>
        <reply>
        {reponse}
        </reply>
        Declared citations: P17, P23      (« none » si la liste est vide)
        Declared action: geste_commercial
    avis : dict avec rating, title, text. reponse : dict avec reponse, citations, action ; ses autres champs (raison,
    cas, problems) ne sont jamais envoyés, pour que le juge se fasse sa propre idée.
    """
    bloc = "\n".join(f"[{p['id']}] {p['titre']}: {p['texte']}" for p in passages)
    system = f"{grille}\n\n<policy>\n{bloc}\n</policy>"
    citations = ", ".join(reponse["citations"]) or "none"
    user = (f"<review>\nRating: {int(avis['rating'])}\nTitle: {avis['title']}\nText: {avis['text']}\n</review>\n"
            f"<reply>\n{reponse['reponse']}\n</reply>\n"
            f"Declared citations: {citations}\nDeclared action: {reponse['action']}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------- Verdict et calibration ----------

def est_fautive(note):
    """Vrai si au moins un des 3 critères est à non. Même définition pour tes notes et pour celles du juge."""
    return not all(note[c] for c in CRITERES)


def matrice_fautes(notes_ref, notes_juge, positif=est_fautive):
    """{"VP", "FP", "FN", "VN"} sur les review_id notés des deux côtés ; « positif » = fautive par défaut.

    notes_ref (tes notes) et notes_juge : {review_id: note}. VP = faute attrapée, FP = fausse alerte,
    FN = faute ratée, VN = bonne réponse reconnue. Les review_id présents d'un seul côté sont ignorés.
    positif : la fonction qui dit si une note est un « positif » ; pour un critère c, lambda n: not n[c].
    """
    m = {"VP": 0, "FP": 0, "FN": 0, "VN": 0}
    for rid in notes_ref.keys() & notes_juge.keys():
        ref, juge = positif(notes_ref[rid]), positif(notes_juge[rid])
        m[("V" if ref == juge else "F") + ("P" if juge else "N")] += 1
    return m


def matrices_par_critere(notes_ref, notes_juge):
    """{critère: matrice} où « positif » = ce critère à non."""
    return {c: matrice_fautes(notes_ref, notes_juge, positif=lambda n, c=c: not n[c]) for c in CRITERES}


def regle_acceptation(m):
    """La règle du 23 oct. 2026 sur la matrice d'un critère : "trop peu" si tu as mis moins de MIN_PAR_CLASSE « non »
    ou « oui » ; sinon "accepté" si rappel >= SEUIL_RAPPEL, précision >= SEUIL_PRECISION et kappa >= SEUIL_KAPPA,
    "refusé" sinon.

    Les « oui » comptent aussi : sans eux, un juge qui répond « non » à tout aurait une précision flatteuse. Le kappa
    ferme le dernier trou : quand tu as mis beaucoup de « non », la précision d'un tel juge reste au-dessus du seuil.
    """
    if min(m["VP"] + m["FN"], m["FP"] + m["VN"]) < MIN_PAR_CLASSE:
        return "trop peu"
    met = metriques_calibration(m)
    ok = (met["rappel_fautes"] >= SEUIL_RAPPEL and met["precision_alertes"] is not None
          and met["precision_alertes"] >= SEUIL_PRECISION and met["kappa"] is not None and met["kappa"] >= SEUIL_KAPPA)
    return "accepté" if ok else "refusé"


def metriques_calibration(m):
    """Accord, rappel des fautes, précision des alertes et kappa de Cohen à partir de la matrice.

    Une métrique vaut None quand son dénominateur est nul (aucune faute, aucune alerte, ou kappa indéfini quand
    les deux correcteurs mettent tous la même note).
    """
    n = sum(m.values())
    vp, fp, fn, vn = m["VP"], m["FP"], m["FN"], m["VN"]
    accord = (vp + vn) / n if n else None
    rappel = vp / (vp + fn) if vp + fn else None
    precision = vp / (vp + fp) if vp + fp else None
    kappa = None
    if n:
        p_juge, p_ref = (vp + fp) / n, (vp + fn) / n          # part de « fautive » de chaque côté
        p_e = p_juge * p_ref + (1 - p_juge) * (1 - p_ref)     # accord attendu par hasard
        if p_e < 1:
            kappa = (accord - p_e) / (1 - p_e)
    return {"n": n, "accord": accord, "rappel_fautes": rappel, "precision_alertes": precision, "kappa": kappa}


# ---------- Comparaison appariée ----------

def p_test_du_signe(gagnes_a, gagnes_b):
    """p bilatéral du test du signe : sous l'hypothèse « A et B se valent », chaque avis discordant est gagné par
    l'un ou l'autre avec une probabilité 1/2. Aucun avis discordant -> 1.0."""
    n = gagnes_a + gagnes_b
    if n == 0:
        return 1.0
    queue = sum(comb(n, i) for i in range(max(gagnes_a, gagnes_b), n + 1)) / 2 ** n
    return min(1.0, 2 * queue)


def comparer_apparie(acceptables_a, acceptables_b, seed=42):
    """Compare A et B sur les review_id présents des deux côtés. acceptables_x : {review_id: bool}.

    Renvoie n, taux_a, taux_b, ecart (taux_b - taux_a), gagnes_a (A acceptable et B fautive), gagnes_b, p (test du
    signe) et ic (intervalle bootstrap à 95 % de l'écart, en tirant des avis avec remise : les paires restent liées).
    """
    ids = sorted(acceptables_a.keys() & acceptables_b.keys())
    a = np.array([acceptables_a[i] for i in ids], dtype=float)
    b = np.array([acceptables_b[i] for i in ids], dtype=float)
    gagnes_a, gagnes_b = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    _, bas, haut = ic_bootstrap(b - a, seed=seed)
    return {"n": len(ids), "taux_a": a.mean(), "taux_b": b.mean(), "ecart": (b - a).mean(),
            "gagnes_a": gagnes_a, "gagnes_b": gagnes_b, "p": p_test_du_signe(gagnes_a, gagnes_b), "ic": (bas, haut)}


# ---------- Fichiers ----------

def lire_par_avis(chemin):
    """{review_id: ligne} d'un fichier JSONL ; lignes en erreur ignorées ; la dernière ligne réussie l'emporte."""
    if not chemin.exists():
        return {}
    lignes = {}
    for r in map(json.loads, chemin.open(encoding="utf-8")):
        if "error" not in r:
            lignes[r["review_id"]] = r
    return lignes


def chemin_reponses(part, mode, prompt=PROMPT_RAG):
    return DOSSIER / f"reponses_{part}_{mode}_{prompt}.jsonl"


def chemin_jugements(part, mode, modele=JUGE_MODEL, prompt_rag=PROMPT_RAG):
    """Un fichier par jeu, mode, prompt de génération, grille et juge : rien ne se mélange. Le juge par défaut garde
    le nom court ; un autre modèle ajoute son nom."""
    suffixe = "" if modele == JUGE_MODEL else f"_{modele}"
    return DOSSIER / f"jugements_{part}_{mode}_{prompt_rag}_{PROMPT_JUGE}{suffixe}.jsonl"


def chemin_notes(part):
    return DOSSIER / f"notes_martin_{part}.jsonl"


def reponses_dans_l_ordre(part, mode, n=None, prompt=PROMPT_RAG):
    """Les réponses réussies, dans l'ordre du tirage du jeu, limitées aux n premiers avis du jeu."""
    reponses = lire_par_avis(chemin_reponses(part, mode, prompt))
    ids = charger_part(part)["review_id"].tolist()[:n]
    return [reponses[i] for i in ids if i in reponses]


def wilson(k, n, z=1.959964):
    """Intervalle de Wilson à 95 % d'une proportion k / n ; plus juste que p ± 1,96 σ quand n est petit."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    demi = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - demi, centre + demi


def fmt(x):
    return "  –  " if x is None else f"{x:.3f}"


# ---------- Commandes ----------

def afficher_reponse(avis, r, par_id):
    print(f"\n===== {r['review_id'][-24:]} · note {int(avis['rating'])}\nTitle: {avis['title']}\nText: {avis['text']}")
    print(f"--- réponse (action : {r['action']}, citations : {', '.join(r['citations']) or 'aucune'})\n{r['reponse']}")
    for i in r["citations"]:
        if i in par_id:
            print(f"--- [{i}] {par_id[i]['titre']}: {par_id[i]['texte']}")


def cmd_noter(part, n, mode):
    """Tes 3 oui / non pour chaque réponse. Ni la raison, ni le cas, ni les signaux ne sont affichés : ta note doit
    venir de l'avis, de la réponse et de la politique seulement."""
    passages = charger_corpus()
    par_id = {p["id"]: p for p in passages}
    avis = charger_part(part).set_index("review_id")
    out = chemin_notes(part)
    faits = set(lire_par_avis(out))
    a_faire = [r for r in reponses_dans_l_ordre(part, mode, n) if r["review_id"] not in faits]
    print(f"{len(faits)} déjà notées, {len(a_faire)} à faire. Saisie : 3 lettres o/n dans l'ordre action_conforme, "
          f"fondee, ton, puis un commentaire facultatif (ex. « onn condition supposée »). P17 affiche un passage, "
          f"q quitte.")
    with out.open("a", encoding="utf-8") as f:
        for r in a_faire:
            afficher_reponse(avis.loc[r["review_id"]], r, par_id)
            while True:
                saisie = input("action fondee ton > ").strip()
                if saisie == "q":
                    return
                if saisie.upper() in par_id:
                    p = par_id[saisie.upper()]
                    print(f"[{p['id']}] {p['titre']}: {p['texte']}")
                    continue
                lettres, _, commentaire = saisie.partition(" ")
                if len(lettres) == 3 and set(lettres.lower()) <= {"o", "n"}:
                    break
                print("Il faut 3 lettres o ou n, par exemple « ono ».")
            note = dict(zip(CRITERES, (c == "o" for c in lettres.lower())))
            f.write(json.dumps({"review_id": r["review_id"], "mode": mode, "prompt_version": r["prompt_version"],
                                **note, "commentaire": commentaire.strip()}, ensure_ascii=False) + "\n")
            f.flush()
    print("Écrit :", out)


def cmd_page(part, n, mode):
    """Même contenu que noter, dans une page HTML autonome (data/rag/notation_<part>.html) : guide, politique
    filtrable, sauvegarde dans le navigateur, bouton qui télécharge notes_martin_<part>.jsonl au format de noter."""
    passages = charger_corpus()
    avis = charger_part(part).set_index("review_id")
    reponses = reponses_dans_l_ordre(part, mode, n)
    donnees = {"part": part, "mode": mode, "prompt_version": PROMPT_RAG, "passages": passages,
               "reponses": [{"review_id": r["review_id"], "rating": int(avis.loc[r["review_id"], "rating"]),
                             "title": avis.loc[r["review_id"], "title"], "text": avis.loc[r["review_id"], "text"],
                             "reponse": r["reponse"], "citations": r["citations"], "action": r["action"]}
                            for r in reponses]}   # ni raison, ni cas, ni problems : comme dans noter
    # « </ » dans un avis fermerait la balise <script> qui contient les données
    page = (GABARIT_PAGE.read_text(encoding="utf-8").replace("__PART__", part)
            .replace("__NB_PASSAGES__", str(len(passages)))
            .replace("__DONNEES__", json.dumps(donnees, ensure_ascii=False).replace("</", "<\\/")))
    out = DOSSIER / f"notation_{part}.html"
    out.write_text(page, encoding="utf-8")
    print(f"{len(reponses)} réponses ({mode}, {PROMPT_RAG}) sur les {n} premiers avis de {part}. Écrit : {out}")


def cmd_juger(part, mode, limit, pause_s, modele, prompt):
    client = client_mistral()
    passages = charger_corpus()
    grille = load_prompt(PROMPT_JUGE).replace("{marque}", MARQUE)
    avis = charger_part(part).set_index("review_id")
    out = chemin_jugements(part, mode, modele, prompt)
    faits = set(lire_par_avis(out))
    a_faire = [r for r in reponses_dans_l_ordre(part, mode, limit, prompt) if r["review_id"] not in faits]
    print(f"{len(faits)} déjà jugées, {len(a_faire)} à faire (réponses {prompt}, juge {modele}, {PROMPT_JUGE}, "
          f"pause {pause_s} s)")
    with out.open("a", encoding="utf-8") as f:
        for r in a_faire:
            messages = construire_messages_juge(grille, passages, avis.loc[r["review_id"]], r)
            ligne = {"review_id": r["review_id"], "mode": mode, "prompt_rag": r["prompt_version"],
                     "prompt_juge": PROMPT_JUGE}
            try:
                t0 = time.perf_counter()
                rep, essais = call_with_retry(lambda: client.chat.parse(
                    model=modele, response_format=Jugement, temperature=0, top_p=1, max_tokens=400, random_seed=42,
                    messages=messages, **REGLAGES_MODELE.get(modele, {})))
                parsed = rep.choices[0].message.parsed
                if parsed is None:   # la cause est dans la fin de génération et le début du texte brut
                    choix = rep.choices[0]
                    raise ValueError(f"jugement vide : finish_reason={choix.finish_reason}, "
                                     f"{rep.usage.completion_tokens} tokens de sortie, "
                                     f"contenu={str(choix.message.content)[:400]!r}")
                ligne.update({"model": rep.model, **parsed.model_dump(),
                              "prompt_tokens": rep.usage.prompt_tokens,
                              "completion_tokens": rep.usage.completion_tokens,
                              "latency_s": round(time.perf_counter() - t0, 3), "attempts": essais})
            except Exception as e:
                ligne["error"] = repr(e)[:300]
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
            f.flush()
            verdict = "ERREUR" if "error" in ligne else ("fautive" if est_fautive(ligne) else "acceptable")
            print(f"  {r['review_id'][-24:]} : {verdict} " + " ".join(f"{c}={ligne.get(c)}" for c in CRITERES))
            time.sleep(pause_s)
    print("Écrit :", out)


def cmd_calibrer(part, mode, modele, prompt):
    notes = {rid: r for rid, r in lire_par_avis(chemin_notes(part)).items()
             if r["mode"] == mode and r["prompt_version"] == prompt}
    jugements = lire_par_avis(chemin_jugements(part, mode, modele, prompt))
    communs = sorted(notes.keys() & jugements.keys())
    print(f"{len(notes)} notes, {len(jugements)} jugements, {len(communs)} en commun "
          f"({part}, {mode}, réponses {prompt}, {PROMPT_JUGE}, {modele})")
    if not communs:
        return
    m = matrice_fautes(notes, jugements)
    met = metriques_calibration(m)
    print(f"\nVerdict global (pour information, il ne décide rien) :\n"
          f"                 toi : fautive   toi : acceptable\n"
          f"juge : fautive        {m['VP']:3d}             {m['FP']:3d}\n"
          f"juge : acceptable     {m['FN']:3d}             {m['VN']:3d}\n"
          f"accord {fmt(met['accord'])} · kappa {fmt(met['kappa'])}")
    print(f"\nPar critère (positif = « non ») ; règle : rappel >= {SEUIL_RAPPEL}, précision >= {SEUIL_PRECISION} et "
          f"kappa >= {SEUIL_KAPPA}, avec au moins {MIN_PAR_CLASSE} « non » et {MIN_PAR_CLASSE} « oui » de ta part")
    print(f"{'critère':<16} {'non toi/juge':>12} {'rappel':>7} {'IC 95 %':>13} {'précision':>9} {'kappa':>6} "
          f"{'accord':>6}  décision")
    decisions = {}
    for c, mc in matrices_par_critere(notes, jugements).items():
        mt = metriques_calibration(mc)
        ic = wilson(mc["VP"], mc["VP"] + mc["FN"])
        decisions[c] = regle_acceptation(mc)
        print(f"{c:<16} {mc['VP'] + mc['FN']:>5} / {mc['VP'] + mc['FP']:<4} {fmt(mt['rappel_fautes']):>7} "
              f"{f'[{ic[0]:.2f} ; {ic[1]:.2f}]' if ic else '–':>13} {fmt(mt['precision_alertes']):>9} "
              f"{fmt(mt['kappa']):>6} {fmt(mt['accord']):>6}  {decisions[c]}")
    if "trop peu" in decisions.values():
        print("\nRègle : trop peu de « non » ou de « oui » sur un critère -> noter 10 réponses de plus avant de "
              "décider.")
    else:
        ok = all(d == "accepté" for d in decisions.values())
        print(f"\nRègle : {'JUGE ACCEPTÉ' if ok else 'juge refusé'} "
              f"({', '.join(c for c, d in decisions.items() if d == 'refusé') or 'les 3 critères passent'})")
    desaccords = [i for i in communs if any(notes[i][c] != jugements[i][c] for c in CRITERES)]
    print(f"\n{len(desaccords)} réponses avec au moins un critère en désaccord :")
    for i in desaccords:
        n_, j_ = notes[i], jugements[i]
        print(f"\n--- {i[-24:]}  toi {''.join('o' if n_[c] else 'n' for c in CRITERES)} · juge "
              f"{''.join('o' if j_[c] else 'n' for c in CRITERES)}  (action, fondée, ton)"
              f"\n    ton commentaire : {n_['commentaire'] or '–'}\n    raison du juge : {j_['raison']}")


def cmd_bilan(part, mode, prompt, contre, modele):
    """Action conforme (le seul critère où le juge est validé), cas choisis et signaux en code pour un prompt de
    génération ; avec contre, comparaison appariée des deux prompts sur les avis jugés pour les deux."""
    versions = [contre, prompt] if contre else [prompt]
    jug = {v: lire_par_avis(chemin_jugements(part, mode, modele, v)) for v in versions}
    rep = {v: lire_par_avis(chemin_reponses(part, mode, v)) for v in versions}
    ids = sorted(set.intersection(*(set(jug[v]) for v in versions)))
    print(f"{len(ids)} avis jugés pour {' et '.join(versions)} ({part}, {mode}, juge {PROMPT_JUGE}, {modele})")
    if not ids:
        return
    lignes = [("action conforme (juge)", lambda j, r: j["action_conforme"])]
    lignes += [(f"  cas {c}", lambda j, r, c=c: r.get("cas") == c) for c in ("securite", "sans_plainte", "plainte")]
    lignes += [("signal en code (problems)", lambda j, r: bool(r["problems"]))]
    print(f"\n{'':30}" + "".join(f"{v:>12}" for v in versions))
    for nom, f in lignes:
        print(f"{nom:30}" + "".join(f"{np.mean([f(jug[v][i], rep[v][i]) for i in ids]):>12.3f}" for v in versions))
    if contre:
        res = comparer_apparie({i: jug[contre][i]["action_conforme"] for i in ids},
                               {i: jug[prompt][i]["action_conforme"] for i in ids})
        print(f"\naction conforme, {prompt} − {contre} = {res['ecart']:+.3f} [{res['ic'][0]:+.3f} ; "
              f"{res['ic'][1]:+.3f}] · avis discordants : {prompt} en gagne {res['gagnes_b']}, {contre} en gagne "
              f"{res['gagnes_a']} · test du signe p = {res['p']:.3f}")
    rates = [i for i in ids if not jug[prompt][i]["action_conforme"]]
    print(f"\n{len(rates)} réponses {prompt} à action non conforme selon le juge :")
    for i in rates:
        r = rep[prompt][i]
        print(f"  {i[-24:]}  cas {r.get('cas')} -> {r['action']} · {jug[prompt][i]['raison'][:220]}")


def cmd_comparer(part, modele, prompt):
    """A (rag) contre B (plein) pour un même prompt. Seule l'action conforme décide : c'est le seul critère où le
    juge est validé ; fondée et ton sont affichés pour information."""
    jug = {mode: lire_par_avis(chemin_jugements(part, mode, modele, prompt)) for mode in ("rag", "plein")}
    rep = {mode: lire_par_avis(chemin_reponses(part, mode, prompt)) for mode in ("rag", "plein")}
    ids = sorted(jug["rag"].keys() & jug["plein"].keys())
    print(f"{len(ids)} avis jugés dans les deux modes ({part}, générateur {prompt}, juge {PROMPT_JUGE}, {modele})")
    if not ids:
        return
    res = comparer_apparie({i: jug["rag"][i]["action_conforme"] for i in ids},
                           {i: jug["plein"][i]["action_conforme"] for i in ids})
    lignes = [("action conforme (juge validé)", lambda j, r: j["action_conforme"]),
              ("fondée (juge non validé)", lambda j, r: j["fondee"]),
              ("ton (juge non validé)", lambda j, r: j["ton"]),
              ("signal en code (problems)", lambda j, r: bool(r["problems"]))]
    print(f"\n{'':32}{'A (rag)':>12}{'B (plein)':>12}")
    for nom, f in lignes:
        print(f"{nom:32}" + "".join(f"{np.mean([f(jug[m][i], rep[m][i]) for i in ids]):>12.3f}"
                                    for m in ("rag", "plein")))
    for nom, f in [("longueur moyenne (mots)", lambda r: len(r["reponse"].split())),
                   ("tokens d'entrée (moyenne)", lambda r: r["prompt_tokens"]),
                   ("tokens de sortie (moyenne)", lambda r: r["completion_tokens"])]:
        print(f"{nom:32}" + "".join(f"{np.mean([f(rep[m][i]) for i in ids]):>12.0f}" for m in ("rag", "plein")))
    print(f"{'latence médiane (s)':32}" + "".join(f"{np.median([rep[m][i]['latency_s'] for i in ids]):>12.2f}"
                                                for m in ("rag", "plein")))
    print(f"\naction conforme, B − A = {res['ecart']:+.3f} [{res['ic'][0]:+.3f} ; {res['ic'][1]:+.3f}] · avis "
          f"discordants : B en gagne {res['gagnes_b']}, A en gagne {res['gagnes_a']} · test du signe "
          f"p = {res['p']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    sous = parser.add_subparsers(dest="commande", required=True)
    p = sous.add_parser("noter")
    p.add_argument("--part", default="rag_dev")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--mode", choices=["rag", "plein"], default="rag")
    p = sous.add_parser("page")
    p.add_argument("--part", default="rag_dev")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--mode", choices=["rag", "plein"], default="rag")
    p = sous.add_parser("juger")
    p.add_argument("--part", default="rag_dev")
    p.add_argument("--mode", choices=["rag", "plein"], default="rag")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--pause", type=float, default=PAUSE_JUGE_S)
    p.add_argument("--modele", default=JUGE_MODEL)
    p.add_argument("--prompt", default=PROMPT_RAG, help="version du prompt des réponses à juger")
    p = sous.add_parser("calibrer")
    p.add_argument("--part", default="rag_dev")
    p.add_argument("--mode", choices=["rag", "plein"], default="rag")
    p.add_argument("--modele", default=JUGE_MODEL)
    p.add_argument("--prompt", default="rag_v4", help="tes 20 notes portent sur les réponses rag_v4")
    p = sous.add_parser("bilan")
    p.add_argument("--part", default="rag_dev")
    p.add_argument("--mode", choices=["rag", "plein"], default="rag")
    p.add_argument("--prompt", default=PROMPT_RAG)
    p.add_argument("--contre", default=None, help="autre version du prompt, pour une comparaison appariée")
    p.add_argument("--modele", default=JUGE_MODEL)
    p = sous.add_parser("comparer")
    p.add_argument("--part", default="rag_eval")
    p.add_argument("--modele", default=JUGE_MODEL)
    p.add_argument("--prompt", default=PROMPT_RAG)
    args = parser.parse_args()

    if args.commande == "noter":
        cmd_noter(args.part, args.n, args.mode)
    elif args.commande == "page":
        cmd_page(args.part, args.n, args.mode)
    elif args.commande == "juger":
        cmd_juger(args.part, args.mode, args.limit, args.pause, args.modele, args.prompt)
    elif args.commande == "calibrer":
        cmd_calibrer(args.part, args.mode, args.modele, args.prompt)
    elif args.commande == "bilan":
        cmd_bilan(args.part, args.mode, args.prompt, args.contre, args.modele)
    elif args.commande == "comparer":
        cmd_comparer(args.part, args.modele, args.prompt)


if __name__ == "__main__":
    main()
