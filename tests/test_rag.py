"""Tests du jalon 2 : RAG minimal (src/rag.py). Lance-les avec :  python -m pytest -q
Sans Internet ni clé : un faux client renvoie des vecteurs calculés à partir du texte.
Les deux derniers blocs vérifient les vrais fichiers (corpus, tirage des jeux) et sont sautés s'ils sont absents.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.label_llm import POOL
from src.rag import (CORPUS, MOTIFS_PASSAGE, ReponseClient, cas_b_contredit, charger_corpus, construire_index,
                     construire_messages, embed, ids_dans_texte, message_correction, messages_de_relance, mrr,
                     normaliser, passages_autorises, rappel_a_k, rechercher, signaux_redaction, texte_a_indexer,
                     texte_de_relance, tirer_jeux, verifier_citations)


def vecteur_jouet(texte):
    """Vecteur non normalisé, différent pour chaque texte, toujours le même pour un texte donné."""
    return [len(texte), texte.count("a") + 1.0, texte.count("e") + 1.0]


class Erreur429(Exception):
    status_code = 429      # ce que call_with_retry regarde pour décider de réessayer


class FauxEmbeddings:
    def __init__(self, erreurs=0):
        self.appels = []          # un élément par requête réussie : la liste des textes envoyés
        self.erreurs = erreurs    # nombre de 429 à lever avant de répondre

    def create(self, model, inputs):
        if self.erreurs:
            self.erreurs -= 1
            raise Erreur429()
        self.appels.append(list(inputs))
        data = [SimpleNamespace(index=i, embedding=vecteur_jouet(t)) for i, t in enumerate(inputs)]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=10 * len(inputs)))


def faux_client(erreurs=0):
    return SimpleNamespace(embeddings=FauxEmbeddings(erreurs))


def unitaire(M):
    """Lignes de norme 1, calculées ici sans passer par normaliser (que les tests vérifient à part)."""
    M = np.asarray(M, dtype=np.float32)
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def passage(i, motifs, titre="T", texte="x"):
    return {"id": f"P{i:02d}", "titre": titre, "motifs": motifs, "texte": texte}


# ---------- normaliser ----------

def test_normaliser_norme_un_et_float32():
    M = normaliser([[3.0, 4.0], [0.0, 2.0]])
    assert M.dtype == np.float32
    assert np.allclose(np.linalg.norm(M, axis=1), 1)
    assert np.allclose(M[0], [0.6, 0.8])           # la direction ne change pas


def test_normaliser_refuse_un_vecteur_nul():
    with pytest.raises(ValueError):
        normaliser([[1.0, 0.0], [0.0, 0.0]])


# ---------- embed ----------

def test_embed_par_lots_dans_l_ordre():
    client, pauses = faux_client(), []
    textes = [f"texte numero {i} " + "a" * (i % 7) for i in range(70)]
    M, tokens = embed(textes, client, taille_lot=32, pause_s=1.5, sleep=pauses.append)
    assert [len(lot) for lot in client.embeddings.appels] == [32, 32, 6]
    assert M.shape == (70, 3) and M.dtype == np.float32
    assert np.allclose(M, unitaire([vecteur_jouet(t) for t in textes]))   # ligne i = texte i
    assert tokens == 700
    assert pauses == [1.5, 1.5]                    # entre deux lots, pas après le dernier


def test_embed_reessaie_apres_un_429():
    client, pauses = faux_client(erreurs=1), []
    M, _ = embed(["un", "deux"], client, sleep=pauses.append)
    assert M.shape == (2, 3)
    assert len(client.embeddings.appels) == 1
    assert pauses == [1]                           # le backoff de call_with_retry passe par le même sleep


# ---------- recherche ----------

MOTIFS_P = [["reception"], ["prix"], ["general"], ["fraicheur_securite", "reception"]]


def test_passages_autorises_sans_filtre():
    for motifs_avis in (None, [], ["autre"]):
        assert passages_autorises(MOTIFS_P, motifs_avis).tolist() == [True] * 4


def test_passages_autorises_motif_plus_general():
    assert passages_autorises(MOTIFS_P, ["reception"]).tolist() == [True, False, True, True]
    assert passages_autorises(MOTIFS_P, ["prix"]).tolist() == [False, True, True, False]
    assert passages_autorises(MOTIFS_P, ["prix", "fraicheur_securite"]).tolist() == [False, True, True, True]
    assert passages_autorises(MOTIFS_P, ["gout_texture"]).tolist() == [False, False, True, False]


M4 = unitaire([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.2]])
IDS4 = ["P01", "P02", "P03", "P04"]


def test_rechercher_les_k_plus_proches_dans_l_ordre():
    q = unitaire([[1.0, 0.1]])[0]
    res = rechercher(q, M4, IDS4, k=2)
    assert [i for i, _ in res] == ["P04", "P01"]
    assert all(type(s) is float for _, s in res)
    assert res[0][1] >= res[1][1]
    assert np.isclose(res[0][1], float(M4[3] @ q))


def test_rechercher_respecte_le_masque():
    q = unitaire([[1.0, 0.1]])[0]
    masque = np.array([False, True, True, False])
    assert [i for i, _ in rechercher(q, M4, IDS4, k=3, masque=masque)] == ["P03", "P02"]   # 2 autorisés < k


def test_rechercher_egalite_garde_l_ordre_du_corpus():
    M = unitaire([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    res = rechercher(np.array([1.0, 0.0], dtype=np.float32), M, ["P01", "P02", "P03"], k=2)
    assert [i for i, _ in res] == ["P02", "P03"]


# ---------- génération ----------

def test_construire_messages_format_exact():
    ps = [passage(7, ["reception"], "Damaged item", "We replace it."), passage(12, ["general"], "Contact", "Write.")]
    msgs = construire_messages("SYSTEME", ps, "Broken jar", "Arrived broken.")
    assert msgs[0] == {"role": "system", "content": "SYSTEME"}
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == ("<passages>\n[P07] Damaged item: We replace it.\n[P12] Contact: Write.\n</passages>\n"
                                  "<review>\nTitle: Broken jar\nText: Arrived broken.\n</review>")
    assert len(msgs) == 2


def test_verifier_citations_valide():
    assert verifier_citations("Sorry [P07]. We refund you [P12].", ["P07", "P12"], "remboursement",
                              ["P07", "P12", "P20"]) == []


def test_verifier_citations_inconnue():
    assert verifier_citations("Sorry [P99].", ["P99"], "remboursement", ["P07"]) == ["citation_inconnue"]


def test_verifier_citations_non_declaree_et_absente():
    assert verifier_citations("Sorry [P07].", [], "escalade", ["P07"]) == ["citation_non_declaree"]
    assert verifier_citations("Sorry.", ["P07"], "remboursement", ["P07"]) == ["citation_absente_du_texte"]


def test_verifier_citations_sans_citation():
    assert verifier_citations("Sorry.", [], "remboursement", ["P07"]) == ["sans_citation"]
    assert verifier_citations("A colleague will contact you.", [], "escalade", ["P07"]) == []
    assert verifier_citations("Thanks.", [], "aucune", ["P07"]) == []
    assert verifier_citations("Our safety team will contact you.", [], "escalade_securite", ["P07"]) == []


def test_verifier_citations_plusieurs_problemes_dans_l_ordre_une_fois():
    probs = verifier_citations("See [P99] and [P98] and [P07].", ["P07", "P50", "P51"], "remboursement", ["P07"])
    assert probs == ["citation_inconnue", "citation_non_declaree", "citation_absente_du_texte"]


def test_reponse_client_refuse_une_action_ou_un_cas_inconnu():
    ReponseClient(raison="r", cas="plainte", reponse="ok", citations=[], action="escalade")
    with pytest.raises(Exception):
        ReponseClient(raison="r", cas="plainte", reponse="ok", citations=[], action="bon_de_reduction")
    with pytest.raises(Exception):
        ReponseClient(raison="r", cas="urgent", reponse="ok", citations=[], action="escalade")
    assert list(ReponseClient.model_fields)[:2] == ["raison", "cas"]   # décidés avant la réponse


def test_ids_dans_texte_lit_les_deux_formats():
    assert ids_dans_texte("A [P07][P12]. B [P20, P23]. C [note] P30 sans crochets.") == {"P07", "P12", "P20", "P23"}
    assert verifier_citations("Credit [P20, P23].", ["P20", "P23"], "geste_commercial", ["P20", "P23"]) == []


# Phrases tirées des réponses réelles de rag_dev (23 oct. 2026)
def test_signaux_condition_supposee():
    assert signaux_redaction("Since this is your first purchase of this product, we can offer a credit [P17].",
                             "geste_commercial", "plainte") == ["condition_supposee"]
    assert signaux_redaction("If this was your first purchase of this product, we can offer a credit [P17].",
                             "geste_commercial", "plainte") == []


def test_signaux_alternative_non_prevue():
    texte = "We can offer a store credit [P17]. If you’d prefer a refund or replacement instead, let us know."
    assert signaux_redaction(texte, "geste_commercial", "plainte") == ["alternative_non_prevue"]
    assert signaux_redaction(texte, "remboursement", "plainte") == []          # seulement face à un avoir
    assert signaux_redaction("The taste guarantee is not a cash refund [P17].", "geste_commercial", "plainte") == []


def test_signaux_securite_et_coherence():
    texte = "Natural variations can occur, and while they do not affect safety, we understand."
    assert signaux_redaction(texte, "geste_commercial", "securite") == ["cas_action_incoherent", "affirme_securite"]
    assert signaux_redaction("Our safety team will contact you.", "escalade_securite", "securite") == []
    assert signaux_redaction("Thank you!", "geste_commercial", "sans_plainte") == ["cas_action_incoherent"]
    assert signaux_redaction("Thank you!", "aucune", "sans_plainte") == []


def test_signal_securite_a_verifier_lit_l_avis():
    # Avis inventés : aucun texte du jeu de données dans le dépôt.
    avis = "Crunchy dates\nThe second box had lots of pits and shards in it."
    assert signaux_redaction("Credit [P17].", "geste_commercial", "plainte", avis) == ["securite_a_verifier"]
    assert signaux_redaction("Our team will contact you.", "escalade_securite", "securite", avis) == []
    assert signaux_redaction("Credit [P17].", "geste_commercial", "plainte", "Too sweet for me.") == []
    assert signaux_redaction("Thanks.", "aucune", "plainte", "I chipped a tooth on it") == ["securite_a_verifier"]


# ---------- relance corrective (rag_v5) ----------

def test_cas_b_contredit_lit_le_debut_de_la_raison():
    assert not cas_b_contredit("no negative phrase. The customer loves it.", "sans_plainte")
    assert not cas_b_contredit('"No negative phrase" The customer loves it.', "sans_plainte")   # guillemets, majuscules
    assert cas_b_contredit('"I do not like this paste" The customer dislikes the smell.', "sans_plainte")
    assert cas_b_contredit("", "sans_plainte")                      # raison vide : rien ne prouve l'absence de plainte
    assert not cas_b_contredit('"horrible" The customer hates it.', "plainte")    # seul le cas B est contrôlé


def test_message_correction_cite_la_raison_et_interdit_le_cas_b():
    m = message_correction('"I do not like this paste" The customer dislikes it.\nsecond line')
    assert '"I do not like this paste"' in m and "second line" not in m
    assert "case B is not allowed" in m and '"plainte"' in m


def test_messages_de_relance_etendue_securite_et_condition():
    avis = "Prunes. Lots of shards of pits in the second bag."
    rep = "Since this was your first purchase of this product, we can offer a credit [P17]."
    motifs = [m for m, _ in messages_de_relance('"shards of pits" ...', "plainte", rep, "geste_commercial", avis, True)]
    assert motifs == ["securite", "condition"]
    texte = texte_de_relance(messages_de_relance('"shards of pits"', "plainte", rep, "geste_commercial", avis, True))
    assert '"shards"' in texte and "if this was your first purchase" in texte and "Fill all the fields again" in texte
    # sans etendue (rag_v5c) : seul le cas B relance ; déjà escaladé : pas de relance sécurité
    assert messages_de_relance('"shards"', "plainte", rep, "geste_commercial", avis, False) == []
    assert messages_de_relance('"shards"', "securite", "Please do not eat it.", "escalade_securite", avis, True) == []


def test_messages_de_relance_cas_b_en_premier_et_un_seul_appel_a_l_action():
    relances = messages_de_relance('"I got sick"', "sans_plainte", "Thanks!", "aucune", "It made me sick.", True)
    assert [m for m, _ in relances] == ["cas_b", "securite"]
    assert texte_de_relance(relances).count("Fill all the fields again") == 1


# ---------- mesure de la recherche ----------

# Exemple chiffré : rang du premier passage pertinent 1, 3, absent, 2.
CLASSEMENTS = [["P01", "P02", "P03"], ["P05", "P06", "P07"], ["P01", "P02", "P03"], ["P04", "P09", "P01"]]
PERTINENTS = [{"P01"}, {"P07"}, {"P30"}, {"P09", "P01"}]


def test_rappel_a_k_exemple_chiffre():
    assert rappel_a_k(CLASSEMENTS, PERTINENTS, 1) == 0.25
    assert rappel_a_k(CLASSEMENTS, PERTINENTS, 3) == 0.75
    assert rappel_a_k(CLASSEMENTS, PERTINENTS, 5) == 0.75
    assert type(rappel_a_k(CLASSEMENTS, PERTINENTS, 1)) is float


def test_mrr_exemple_chiffre():
    assert np.isclose(mrr(CLASSEMENTS, PERTINENTS), (1 + 1 / 3 + 0 + 1 / 2) / 4)
    assert mrr([["P01", "P02"]], [{"P02", "P01"}]) == 1.0      # seul le premier pertinent compte


# ---------- index (sauvegarde et reprise) ----------

def test_construire_index_relit_sans_appel_puis_refait_si_le_corpus_change(tmp_path):
    ps = [passage(1, ["reception"], "Damaged", "We replace."), passage(2, ["prix"], "Prices", "No matching.")]
    npy, meta = tmp_path / "index.npy", tmp_path / "index.json"
    client = faux_client()
    M1, ids1, tok1 = construire_index(ps, client, npy, meta)
    M2, ids2, tok2 = construire_index(ps, client, npy, meta)
    assert len(client.embeddings.appels) == 1 and tok1 > 0 and tok2 == 0
    assert ids1 == ids2 == ["P01", "P02"] and np.allclose(M1, M2)
    ps[1]["texte"] = "No price matching."
    construire_index(ps, client, npy, meta)
    assert len(client.embeddings.appels) == 2
    assert texte_a_indexer(ps[0]) == "Damaged\nWe replace."


def test_charger_corpus_refuse_doublon_et_motif_inconnu(tmp_path):
    chemin = tmp_path / "c.jsonl"
    chemin.write_text("\n".join(json.dumps(p) for p in [passage(1, ["prix"]), passage(1, ["prix"])]), encoding="utf-8")
    with pytest.raises(ValueError):
        charger_corpus(chemin)
    chemin.write_text(json.dumps(passage(1, ["livraison"])), encoding="utf-8")
    with pytest.raises(ValueError):
        charger_corpus(chemin)


# ---------- vrais fichiers ----------

@pytest.mark.skipif(not CORPUS.exists(), reason="corpus absent")
def test_le_corpus_respecte_les_criteres():
    passages = charger_corpus()
    assert 40 <= len(passages) <= 80
    for motif in MOTIFS_PASSAGE:
        assert sum(motif in p["motifs"] for p in passages) >= 5, motif
    for p in passages:
        assert 100 <= len(p["texte"].split()) <= 250, (p["id"], len(p["texte"].split()))


@pytest.mark.skipif(not POOL.exists(), reason="data/pool.parquet absent")
def test_tirer_jeux_reserve_disjoints_reproductibles():
    pool = pd.read_parquet(POOL)
    jeux = tirer_jeux(pool)
    assert len(jeux["rag_dev"]) == 50 and len(jeux["rag_eval"]) == 150
    assert not set(jeux["rag_dev"]) & set(jeux["rag_eval"])
    parts = pool.set_index("review_id").loc[jeux["rag_dev"] + jeux["rag_eval"], "part"]
    assert (parts == "reserve").all()
    assert tirer_jeux(pool.sample(frac=1, random_state=0)) == jeux   # ne dépend pas de l'ordre des lignes
