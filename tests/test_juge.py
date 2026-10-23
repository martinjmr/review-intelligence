"""Tests du LLM-juge (src/juge.py). Lance-les avec :  python -m pytest -q
Sans Internet ni clé : ils vérifient les messages, le verdict, la calibration et la comparaison appariée sur les
exemples chiffrés du cours 6 (sections 5 et 7).
"""
import pytest
from pydantic import ValidationError

from src.juge import (CRITERES, JUGE_MODEL, PROMPT_JUGE, Jugement, chemin_jugements, comparer_apparie,
                      construire_messages_juge, est_fautive, matrice_fautes, matrices_par_critere,
                      metriques_calibration, p_test_du_signe, regle_acceptation, wilson)


def note(action=True, fondee=True, ton=True):
    return {"action_conforme": action, "fondee": fondee, "ton": ton}


def notes_exemple():
    """Les 20 réponses de la section 5 : 7 VP, 2 FP, 1 FN, 10 VN (fautive = critère fondee à non)."""
    ref, juge = {}, {}
    for i, (r, j) in enumerate([(False, False)] * 7 + [(True, False)] * 2 + [(False, True)] + [(True, True)] * 10):
        ref[f"r{i:02d}"], juge[f"r{i:02d}"] = note(fondee=r), note(fondee=j)
    return ref, juge


AVIS = {"rating": 2.0, "title": "Stale", "text": "Tasted old."}
REPONSE = {"reponse": "Sorry [P17].", "citations": ["P17"], "action": "geste_commercial",
           "raison": "NE PAS ENVOYER", "cas": "sans_plainte", "problems": ["condition_supposee"]}
PASSAGES = [{"id": "P01", "titre": "A", "motifs": ["general"], "texte": "x"},
            {"id": "P17", "titre": "Taste guarantee", "motifs": ["gout_texture"], "texte": "y"}]


# ---------- construire_messages_juge ----------

def test_construire_messages_juge_format_exact():
    m = construire_messages_juge("GRILLE", PASSAGES, AVIS, REPONSE)
    assert m[0] == {"role": "system", "content": "GRILLE\n\n<policy>\n[P01] A: x\n[P17] Taste guarantee: y\n</policy>"}
    assert m[1] == {"role": "user", "content": "<review>\nRating: 2\nTitle: Stale\nText: Tasted old.\n</review>\n"
                                               "<reply>\nSorry [P17].\n</reply>\n"
                                               "Declared citations: P17\nDeclared action: geste_commercial"}


def test_construire_messages_juge_n_envoie_ni_raison_ni_cas_ni_signaux():
    contenu = " ".join(x["content"] for x in construire_messages_juge("G", PASSAGES, AVIS, REPONSE))
    for interdit in ("NE PAS ENVOYER", "sans_plainte", "condition_supposee"):
        assert interdit not in contenu


def test_construire_messages_juge_systeme_identique_pour_le_cache_et_citations_vides():
    autre = {**REPONSE, "reponse": "Thanks!", "citations": [], "action": "aucune"}
    m1 = construire_messages_juge("G", PASSAGES, AVIS, REPONSE)
    m2 = construire_messages_juge("G", PASSAGES, {**AVIS, "text": "Great."}, autre)
    assert m1[0] == m2[0]
    assert m2[1]["content"].endswith("Declared citations: none\nDeclared action: aucune")


# ---------- est_fautive ----------

def test_est_fautive_un_seul_non_suffit():
    assert not est_fautive(note())
    for c in CRITERES:
        assert est_fautive({**note(), c: False})


# ---------- matrice_fautes et metriques_calibration ----------

def test_matrice_fautes_exemple_du_cours_et_ids_communs_seulement():
    ref, juge = notes_exemple()
    juge["absent_de_ref"] = note(ton=False)
    assert matrice_fautes(ref, juge) == {"VP": 7, "FP": 2, "FN": 1, "VN": 10}


def test_metriques_calibration_exemple_du_cours():
    met = metriques_calibration({"VP": 7, "FP": 2, "FN": 1, "VN": 10})
    assert met["n"] == 20
    assert met["accord"] == pytest.approx(0.85)
    assert met["rappel_fautes"] == pytest.approx(0.875)
    assert met["precision_alertes"] == pytest.approx(7 / 9)
    assert met["kappa"] == pytest.approx(0.6939, abs=1e-4)


def test_metriques_calibration_juge_indulgent():
    met = metriques_calibration({"VP": 0, "FP": 0, "FN": 2, "VN": 18})
    assert met["accord"] == pytest.approx(0.90)
    assert met["rappel_fautes"] == 0
    assert met["precision_alertes"] is None      # le juge n'a lancé aucune alerte
    assert met["kappa"] == pytest.approx(0.0)


def test_metriques_calibration_sans_faute_ni_note():
    met = metriques_calibration({"VP": 0, "FP": 0, "FN": 0, "VN": 5})
    assert met["rappel_fautes"] is None and met["kappa"] is None    # tout le monde d'accord sur « acceptable »
    assert metriques_calibration({"VP": 0, "FP": 0, "FN": 0, "VN": 0})["accord"] is None


def test_matrices_par_critere_positif_egal_non():
    ref = {"a": note(fondee=False), "b": note(), "c": note(ton=False)}
    juge = {"a": note(fondee=False, ton=False), "b": note(fondee=False), "c": note(ton=False)}
    m = matrices_par_critere(ref, juge)
    assert m["fondee"] == {"VP": 1, "FP": 1, "FN": 0, "VN": 1}
    assert m["ton"] == {"VP": 1, "FP": 1, "FN": 0, "VN": 1}
    assert m["action_conforme"] == {"VP": 0, "FP": 0, "FN": 0, "VN": 3}


def test_regle_acceptation_refuse_le_juge_qui_dit_non_a_tout():
    """Tes 20 notes sur « fondée » : 9 non, 11 oui. Juge qui dit non à tout : rappel 1, précision 9/20 = 0,45."""
    assert regle_acceptation({"VP": 9, "FP": 11, "FN": 0, "VN": 0}) == "refusé"
    assert regle_acceptation({"VP": 8, "FP": 2, "FN": 1, "VN": 9}) == "accepté"      # rappel 0,89, précision 0,80
    assert regle_acceptation({"VP": 7, "FP": 1, "FN": 2, "VN": 10}) == "refusé"      # rappel 0,78 < 0,80
    # « ton » : 14 non, 6 oui ; non à tout -> précision 0,70 >= 0,67 mais kappa 0 < 0,40
    assert regle_acceptation({"VP": 14, "FP": 6, "FN": 0, "VN": 0}) == "refusé"


def test_regle_acceptation_trop_peu_de_oui_ou_de_non():
    """Le verdict global de tes notes : 18 fautives, 2 acceptables -> la règle ne peut pas trancher."""
    assert regle_acceptation({"VP": 18, "FP": 2, "FN": 0, "VN": 0}) == "trop peu"
    assert regle_acceptation({"VP": 2, "FP": 0, "FN": 2, "VN": 16}) == "trop peu"


def test_wilson_exemple_du_cours():
    bas, haut = wilson(7, 8)
    assert (round(bas, 2), round(haut, 2)) == (0.53, 0.98)
    assert wilson(0, 0) is None


# ---------- p_test_du_signe et comparer_apparie ----------

def test_p_test_du_signe_exemples_du_cours():
    assert p_test_du_signe(3, 8) == pytest.approx(0.2266, abs=1e-4)
    assert p_test_du_signe(1, 10) == pytest.approx(0.0117, abs=1e-4)
    assert p_test_du_signe(5, 15) == pytest.approx(0.0414, abs=1e-4)
    assert p_test_du_signe(0, 0) == 1.0
    assert p_test_du_signe(4, 4) == 1.0          # égalité : p plafonné à 1


def test_comparer_apparie_exemple_du_cours():
    paires = [(True, True)] * 28 + [(True, False)] * 3 + [(False, True)] * 8 + [(False, False)] * 11
    a = {f"r{i:02d}": x for i, (x, _) in enumerate(paires)}
    b = {f"r{i:02d}": y for i, (_, y) in enumerate(paires)}
    b["seulement_b"] = False
    res = comparer_apparie(a, b)
    assert res["n"] == 50
    assert (res["taux_a"], res["taux_b"]) == (pytest.approx(0.62), pytest.approx(0.72))
    assert res["ecart"] == pytest.approx(0.10)
    assert (res["gagnes_a"], res["gagnes_b"]) == (3, 8)
    assert res["p"] == pytest.approx(0.2266, abs=1e-4)
    bas, haut = res["ic"]
    assert bas < 0 < haut and bas < res["ecart"] < haut


# ---------- Fichiers ----------

def test_chemin_jugements_un_fichier_par_prompt_et_par_juge():
    defaut = chemin_jugements("rag_dev", "rag", prompt_rag="rag_v4")
    autre = chemin_jugements("rag_dev", "rag", "mistral-large-4-0", "rag_v4")
    assert defaut.name == f"jugements_rag_dev_rag_rag_v4_{PROMPT_JUGE}.jsonl"    # nom court pour le juge par défaut
    assert autre.name == f"jugements_rag_dev_rag_rag_v4_{PROMPT_JUGE}_mistral-large-4-0.jsonl"
    assert chemin_jugements("rag_dev", "rag", JUGE_MODEL, "rag_v5") != defaut    # v4 et v5 ne se mélangent pas


# ---------- Jugement ----------

def test_jugement_exige_les_quatre_champs():
    Jugement(raison="ok", action_conforme=True, fondee=False, ton=True)
    with pytest.raises(ValidationError):
        Jugement(raison="ok", action_conforme=True, fondee=False)
