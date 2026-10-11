"""Tests de l'étape 1. Lance-les avec :  python -m pytest -q
Tous doivent passer AVANT d'appeler la vraie API : ils n'utilisent ni Internet ni ta clé.
"""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data import build_pool, clean_text, make_review_id, split_pool
from src.label_llm import Etiquetage, call_with_retry, check_rules, label_part
from src.report import motif_shares, verdict


# ---------- data.py ----------

def test_clean_text_enleve_html_et_espaces():
    assert clean_text("Great<br />taste &amp; smell  ") == "Great taste & smell"
    assert clean_text(None) == ""


def test_make_review_id_stable():
    row = {"user_id": "U1", "parent_asin": "B0", "timestamp": 123}
    assert make_review_id(row) == make_review_id(dict(row))
    assert "U1" in make_review_id(row) and "B0" in make_review_id(row)


def _fake_raw(n=400):
    return pd.DataFrame({
        "rating": [1.0, 2.0, 3.0, 5.0] * (n // 4),
        "title": ["t"] * n,
        "text": [f"this product is not good number {i}" for i in range(n)],
        "user_id": [f"u{i}" for i in range(n)],
        "parent_asin": [f"p{i % 50}" for i in range(n)],
        "timestamp": list(range(n)),
    })


def test_build_pool_filtre_les_notes_et_les_textes_courts():
    raw = _fake_raw()
    raw.loc[0, "text"] = "bad"                      # trop court
    pool = build_pool(raw, pool_size=200, seed=0)
    assert len(pool) == 200
    assert pool["rating"].max() <= 3
    assert (pool["text"].str.split().str.len() >= 5).all()
    assert pool["review_id"].is_unique


def test_build_pool_meme_graine_meme_resultat():
    raw = _fake_raw()
    a = build_pool(raw, pool_size=100, seed=1)["review_id"].tolist()
    b = build_pool(raw, pool_size=100, seed=1)["review_id"].tolist()
    assert a == b


def test_split_pool_tailles_et_parts_disjointes():
    pool = pd.DataFrame({"review_id": [f"r{i}" for i in range(6000)]})
    out = split_pool(pool, seed=0)
    sizes = out["part"].value_counts().to_dict()
    assert sizes == {"train": 5000, "reserve": 500, "test": 300, "explo": 200}
    assert out["review_id"].is_unique          # chaque avis dans une seule part


# ---------- label_llm.py ----------

def test_check_rules():
    assert check_rules(["qualite", "prix"]) == []
    assert "liste_vide" in check_rules([])
    assert "autre_combine" in check_rules(["autre", "prix"])


class _HttpError(Exception):
    def __init__(self, code):
        self.status_code = code


def test_retry_reessaie_sur_429_puis_reussit():
    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise _HttpError(429)
        return "ok"

    assert call_with_retry(flaky, sleep=waits.append) == "ok"
    assert waits == [1, 2]                      # backoff exponentiel


def test_retry_abandonne_tout_de_suite_sur_401():
    waits = []
    with pytest.raises(_HttpError):
        call_with_retry(lambda: (_ for _ in ()).throw(_HttpError(401)), sleep=waits.append)
    assert waits == []


class _FakeClient:
    """Imite client.chat.parse sans appeler Internet."""

    def __init__(self):
        self.n_calls = 0
        self.chat = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.n_calls += 1
        parsed = Etiquetage(evidence="arrived late", motif_libre="late delivery", motifs=["livraison"])
        message = SimpleNamespace(parsed=parsed)
        usage = SimpleNamespace(prompt_tokens=600, completion_tokens=40)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def test_label_part_ecrit_puis_reprend_sans_refaire(tmp_path):
    df = pd.DataFrame({"review_id": ["a", "b", "c"], "title": ["t"] * 3, "text": ["x"] * 3})
    out = tmp_path / "labels.jsonl"
    client = _FakeClient()
    label_part(client, df.head(2), "prompt", "v1", out, pause_s=0)
    label_part(client, df, "prompt", "v1", out, pause_s=0)     # reprise
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert [l["review_id"] for l in lines] == ["a", "b", "c"]
    assert client.n_calls == 3                                 # rien n'a été refait
    assert lines[0]["motifs"] == ["livraison"] and lines[0]["prompt_version"] == "v1"
    assert lines[0]["prompt_tokens"] == 600


# ---------- report.py ----------

def test_motif_shares_multi_etiquette():
    df = pd.DataFrame({"motifs": [["qualite"], ["qualite", "prix"], ["autre"], ["livraison"]]})
    s = motif_shares(df)
    assert s["qualite"] == 0.5 and s["prix"] == 0.25


def test_verdict():
    ok = {"livraison": .1, "emballage": .1, "qualite": .5, "prix": .1, "service": .06, "autre": .1}
    assert verdict(ok) == []
    bad = dict(ok, service=0.01, autre=0.4)
    assert len(verdict(bad)) == 2
