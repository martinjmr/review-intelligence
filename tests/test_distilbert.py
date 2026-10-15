"""Tests de DistilBERT (cours 4) : seulement les fonctions sans torch, qui tournent en local.
L'entraînement lui-même se vérifie sur Colab avec `python -m src.distilbert grid --rapide`.
"""
import json
import zipfile

import numpy as np
import pytest

from src.distilbert import (batches, choose_config, lr_factor, make_bundle, n_epochs_of, pos_weights, proba_path,
                            read_runs, run_final)


def test_pos_weights_negatifs_sur_positifs():
    Y = np.array([[1, 0], [0, 0], [0, 1], [0, 1]])
    assert pos_weights(Y) == pytest.approx([3.0, 1.0])          # 3 négatifs pour 1 positif ; 2 pour 2


def test_pos_weights_refuse_un_motif_sans_positif():
    with pytest.raises(ValueError):
        pos_weights(np.array([[1, 0], [1, 0]]))


def test_lr_factor_monte_puis_descend():
    assert lr_factor(0, 100, 10) == 0.0
    assert lr_factor(5, 100, 10) == pytest.approx(0.5)
    assert lr_factor(10, 100, 10) == pytest.approx(1.0)
    assert lr_factor(55, 100, 10) == pytest.approx(0.5)
    assert lr_factor(100, 100, 10) == 0.0


def test_batches_couvrent_tout_une_fois():
    lots = batches(10, 4, seed=1)
    assert [len(b) for b in lots] == [4, 4, 2]
    assert sorted(np.concatenate(lots).tolist()) == list(range(10))
    assert [b.tolist() for b in batches(10, 4, seed=1)] == [b.tolist() for b in lots]   # même graine, même ordre
    assert np.concatenate(batches(10, 4)).tolist() == list(range(10))                     # sans graine : ordre naturel


def _run(lr, seed, f1s, n_epochs=None):
    r = {"lr": lr, "seed": seed, "epochs": [{"epoch": i + 1, "macro_f1": f} for i, f in enumerate(f1s)]}
    if n_epochs is not None:
        r["n_epochs"] = n_epochs
    return r


def test_choose_config_moyenne_sur_les_graines():
    runs = [_run(2e-5, 42, [0.60, 0.66, 0.67]), _run(2e-5, 43, [0.61, 0.68, 0.65]),
            _run(5e-5, 42, [0.70, 0.62, 0.60]), _run(5e-5, 43, [0.58, 0.63, 0.61])]
    best = choose_config(runs)
    assert (best["lr"], best["n_epochs"], best["epochs"], best["n_graines"]) == (2e-5, 3, 2, 2)
    assert best["macro_f1"] == pytest.approx(0.67)            # (0,66 + 0,68) / 2 bat 0,64 malgré le 0,70 isolé
    assert best["ecart_type"] == pytest.approx(np.std([0.66, 0.68], ddof=1), abs=1e-4)


def test_choose_config_distingue_les_calendriers():
    # même taux : l'époque 4 d'un calendrier de 6 n'est pas l'époque 4 d'un calendrier de 4 (taux pas encore à 0)
    runs = ([_run(5e-5, s, [0.55, 0.64, 0.69, 0.70]) for s in (42, 43)]
            + [_run(5e-5, s, [0.54, 0.63, 0.67, 0.69, 0.72, 0.71], n_epochs=6) for s in (42, 43)])
    best = choose_config(runs)
    assert (best["lr"], best["n_epochs"], best["epochs"]) == (5e-5, 6, 5)


def test_choose_config_ex_aequo_et_graines_incompletes():
    runs = [_run(2e-5, s, [0.65, 0.65]) for s in (42, 43)] + [_run(5e-5, s, [0.65, 0.65]) for s in (42, 43)]
    best = choose_config(runs)                                # ex aequo : plus petit taux, puis moins d'époques
    assert (best["lr"], best["epochs"]) == (2e-5, 1)
    runs.append(_run(5e-5, 44, [0.65, 0.90]))                 # 5e-5 a maintenant 3 graines, 2e-5 seulement 2
    best = choose_config(runs)
    assert (best["lr"], best["epochs"], best["n_graines"]) == (5e-5, 2, 3)


def test_n_epochs_of_et_proba_path(tmp_path):
    assert n_epochs_of(_run(5e-5, 42, [0.6, 0.7])) == 2         # lignes du 14 oct. : pas de champ n_epochs
    assert n_epochs_of(_run(5e-5, 42, [0.6], n_epochs=6)) == 6
    assert proba_path(tmp_path, 5e-5, 6, 42).name == "val_proba_lr5e-05_ep6_graine42.npy"
    assert proba_path(tmp_path, 5e-5, 4, 42).name == "val_proba_lr5e-05_ep4_graine42.npy"
    (tmp_path / "val_proba_lr5e-05_graine42.npy").write_bytes(b"")
    assert proba_path(tmp_path, 5e-5, 4, 42).name == "val_proba_lr5e-05_graine42.npy"    # ancien nom, 4 époques


def test_read_runs(tmp_path):
    assert read_runs(tmp_path / "absent.jsonl") == []
    p = tmp_path / "grille.jsonl"
    p.write_text(json.dumps(_run(2e-5, 42, [0.6])) + "\n\n", encoding="utf-8")
    assert read_runs(p) == [_run(2e-5, 42, [0.6])]


def test_make_bundle_garde_les_chemins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "b.jsonl").write_text("{}", encoding="utf-8")
    out = make_bundle("paquet/p.zip", ["src/a.py", "data/b.jsonl"])
    assert sorted(zipfile.ZipFile(out).namelist()) == ["data/b.jsonl", "src/a.py"]


def test_final_refuse_si_les_predictions_existent(tmp_path):
    (tmp_path / "preds_test_distilbert.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="une seule fois"):
        run_final(tmp_path, validee=True)


def test_final_refuse_tant_que_la_grille_n_est_pas_validee(tmp_path):
    with pytest.raises(SystemExit, match="GRILLE_VALIDEE"):
        run_final(tmp_path, validee=False)
