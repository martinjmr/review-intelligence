"""Étape 4 : DistilBERT fine-tuné sur les étiquettes du LLM, sur GPU (Colab) (cours 4).

Usage dans Colab, depuis le dossier du paquet (voir data/colab/distilbert_colab.ipynb) :
  python -m src.distilbert grid --rapide --out-dir DOSSIER   # test de fumée : 64 avis, 1 époque, ~1 min
  python -m src.distilbert grid --out-dir DOSSIER            # GRID x 3 graines, mesure sur la validation à chaque époque
  python -m src.distilbert final --out-dir DOSSIER           # UNE fois, après validation de la grille (GRILLE_VALIDEE)
En local :
  python -m src.distilbert paquet                            # data/colab/review_intelligence_colab.zip (code + données)
  python -m src.distilbert compare                           # TF-IDF contre DistilBERT sur la validation (apparié)
                                                             # (résultats de Colab copiés dans data/distilbert/)
La grille ne lit jamais le test. Relancer `grid` reprend où il s'est arrêté (une ligne par entraînement terminé dans
DOSSIER/grille.jsonl) ; quand tout est fait, il affiche seulement le bilan.
torch et transformers ne sont importés que dans les fonctions qui s'en servent : les tests tournent sans eux.
"""
import argparse
import json
import math
import time
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.label_llm import POOL, check_rules
from src.metrics import MOTIFS, exact_match, macro_f1, paired_bootstrap, scores, to_matrix
from src.tfidf_lr import MOTIFS_PLAINTE, build_text, confusions, load_data, plainte_matrix, to_motifs

MODEL_NAME = "distilbert-base-uncased"   # avis en anglais ; 66 M de paramètres
MAX_LENGTH = 512          # tokens ; mesuré sur l'entraînement : médiane 47, 1,9 % au-delà de 256, 0,2 % (8 avis) de 512
BATCH_SIZE = 16
GRID = [(2e-5, 4), (5e-5, 4), (5e-5, 6)]   # (taux d'apprentissage, époques du calendrier), chacun x SEEDS ;
                                            # (5e-5, 6) ajouté le 15 oct. 2026 : le meilleur réglage était au bord
GRID_RAPIDE = [(2e-5, 1)]
SEEDS = (42, 43, 44)      # 3 graines par réglage : un fine-tuning varie d'un essai à l'autre
WARMUP = 0.1              # part des pas en montée linéaire du taux d'apprentissage
WEIGHT_DECAY = 0.01
SEED_FINAL = 42
OUT_DIR = Path("data/distilbert")
GRILLE_VALIDEE = True
BUNDLE = Path("data/colab/review_intelligence_colab.zip")
BUNDLE_FILES = ["src/__init__.py", "src/label_llm.py", "src/annotate.py", "src/metrics.py", "src/tfidf_lr.py",
                "src/distilbert.py", "data/pool.parquet", "data/labels_train_v4.jsonl"]   # pas la référence du test


# ---------- Fonctions sans torch (testées en local) ----------

def pos_weights(Y):
    """Poids des positifs pour BCEWithLogitsLoss, un par motif : n_négatifs / n_positifs.

    Même rapport positifs / négatifs que class_weight="balanced" de TF-IDF (cours 3, section 4).
    Lève ValueError si un motif n'a aucun positif.
    """
    Y = np.asarray(Y)
    pos = Y.sum(axis=0)
    if (pos == 0).any():
        raise ValueError("un motif n'a aucun positif dans l'entraînement")
    return (len(Y) - pos) / pos


def lr_factor(step, total_steps, warmup_steps):
    """Multiplicateur du taux d'apprentissage au pas `step` : montée linéaire de 0 à 1 sur warmup_steps, puis
    descente linéaire jusqu'à 0 au pas total_steps (même forme que get_linear_schedule_with_warmup)."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def batches(n, batch_size, seed=None):
    """Lots d'indices (tableaux numpy) de taille batch_size au plus, qui couvrent 0..n-1 une fois chacun.

    Ordre mélangé avec np.random.default_rng(seed) si seed n'est pas None, sinon ordre naturel.
    """
    idx = np.arange(n) if seed is None else np.random.default_rng(seed).permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def n_epochs_of(run):
    """Longueur du calendrier d'un entraînement. Les lignes du 14 oct. 2026 n'ont pas le champ `n_epochs` :
    leur calendrier est leur nombre d'époques mesurées."""
    return run.get("n_epochs", len(run["epochs"]))


def choose_config(runs):
    """Choix sur la validation : pour chaque triplet (taux, calendrier, époque d'arrêt), moyenne du macro-F1 sur les
    graines, puis le meilleur triplet. À égalité : le plus petit taux, puis le calendrier le plus court, puis le moins
    d'époques. Seuls les triplets mesurés sur autant de graines que le plus complet comptent (pas de choix sur une
    graine contre une moyenne de trois).

    runs : [{"lr", "seed", "n_epochs" (facultatif), "epochs": [{"epoch", "macro_f1", ...}, ...]}, ...]
    Renvoie {"lr", "n_epochs", "epochs", "macro_f1", "ecart_type", "n_graines"} ; epochs = époque d'arrêt.
    """
    par_triplet = defaultdict(list)
    for r in runs:
        for e in r["epochs"]:
            par_triplet[(r["lr"], n_epochs_of(r), e["epoch"])].append(e["macro_f1"])
    n = max(len(v) for v in par_triplet.values())
    complets = sorted((k, v) for k, v in par_triplet.items() if len(v) == n)
    (lr, n_epochs, epochs), v = max(complets, key=lambda kv: np.mean(kv[1]))   # max garde le premier ex aequo
    return {"lr": lr, "n_epochs": n_epochs, "epochs": epochs, "macro_f1": round(float(np.mean(v)), 4),
            "ecart_type": round(float(np.std(v, ddof=1)) if n > 1 else 0.0, 4), "n_graines": n}


def read_runs(path):
    """Entraînements terminés de la grille (une ligne JSON chacun) ; [] si le fichier n'existe pas."""
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def proba_path(out_dir, lr, n_epochs, seed):
    """Fichier des probabilités de validation d'un entraînement, de forme (époques, avis, 5). Les fichiers du
    14 oct. 2026 (calendrier de 4 époques) n'ont pas « _ep4 » dans leur nom."""
    path = Path(out_dir) / f"val_proba_lr{lr:g}_ep{n_epochs}_graine{seed}.npy"
    ancien = Path(out_dir) / f"val_proba_lr{lr:g}_graine{seed}.npy"
    return ancien if n_epochs == 4 and not path.exists() and ancien.exists() else path


def make_bundle(out=BUNDLE, files=BUNDLE_FILES):
    """Zip du code et des données nécessaires à Colab, chemins relatifs conservés."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f)
    return out


# ---------- Fonctions avec torch (GPU) ----------

def seed_everything(seed):
    import random

    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(tokenizer, texts, max_length=MAX_LENGTH):
    """Liste des identifiants de tokens de chaque texte, tronqués à max_length, sans remplissage."""
    return tokenizer(list(texts), truncation=True, max_length=max_length)["input_ids"]


def collate(ids_list, pad_id):
    """Un lot : remplissage jusqu'au plus long texte du lot (pas jusqu'à max_length) -> (input_ids, attention_mask)."""
    import torch
    longest = max(len(x) for x in ids_list)
    ids = torch.full((len(ids_list), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids_list), longest), dtype=torch.long)
    for i, x in enumerate(ids_list):
        ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        mask[i, :len(x)] = 1
    return ids, mask


def predict_proba(model, ids, pad_id, device, batch_size=64):
    """Probabilités (n_avis, 5) dans l'ordre de MOTIFS_PLAINTE : sigmoïde des 5 sorties du modèle."""
    import torch
    model.eval()
    order = np.argsort([len(x) for x in ids])          # lots de longueurs voisines : moins de remplissage
    P = np.zeros((len(ids), len(MOTIFS_PLAINTE)))
    with torch.no_grad():
        for b in batches(len(ids), batch_size):
            sel = order[b]
            x, m = collate([ids[i] for i in sel], pad_id)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=x.to(device), attention_mask=m.to(device)).logits
            P[sel] = torch.sigmoid(logits.float()).cpu().numpy()
    return P


def train(ids_tr, Y_tr, lr, n_epochs, seed, device, pad_id, on_epoch=None, stop_after=None, model_name=MODEL_NAME):
    """Fine-tuning complet (encodeur + tête à 5 sorties) ; perte BCE pondérée par pos_weights.

    AdamW, montée puis descente linéaire du taux sur un calendrier de n_epochs époques, gradients bornés à 1,
    demi-précision sur GPU. stop_after (facultatif) arrête après cette époque sans changer le calendrier : c'est la
    trajectoire exacte mesurée par la grille. Appelle on_epoch(epoch, model, perte_moyenne) à la fin de chaque
    époque. Renvoie le modèle.
    """
    import torch
    from transformers import AutoModelForSequenceClassification
    seed_everything(seed)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(MOTIFS_PLAINTE)).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights(Y_tr), dtype=torch.float32,
                                                                 device=device))
    no_decay = ("bias", "LayerNorm", "layer_norm")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": WEIGHT_DECAY},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
               "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr)
    total = math.ceil(len(ids_tr) / BATCH_SIZE) * n_epochs
    warmup = int(WARMUP * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total, warmup))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    Y = torch.tensor(np.asarray(Y_tr), dtype=torch.float32)
    for epoch in range(1, (stop_after or n_epochs) + 1):
        model.train()
        loss_sum = 0.0
        for b in batches(len(ids_tr), BATCH_SIZE, seed=seed * 1000 + epoch):
            x, m = collate([ids_tr[i] for i in b], pad_id)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=x.to(device), attention_mask=m.to(device)).logits
            loss = loss_fn(logits.float(), Y[torch.as_tensor(b)].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(b)
        if on_epoch is not None:
            on_epoch(epoch, model, loss_sum / len(ids_tr))
    return model


# ---------- Commandes ----------

def print_bilan(runs):
    par_triplet = defaultdict(list)
    for r in runs:
        for e in r["epochs"]:
            par_triplet[(r["lr"], n_epochs_of(r), e["epoch"])].append(e["macro_f1"])
    print("\nBilan : macro-F1 de validation, moyenne [min ; max] sur les graines")
    for (lr, n, ep), v in sorted(par_triplet.items()):
        print(f"  taux {lr:g}  calendrier {n}  époque {ep}  :  {np.mean(v):.3f} [{min(v):.3f} ; {max(v):.3f}]"
              f"  ({len(v)} graine(s))")


def run_grid(out_dir=OUT_DIR, rapide=False):
    import torch
    from transformers import AutoTokenizer
    out_dir = Path(out_dir) / "rapide" if rapide else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df = load_data()
    grid, seeds = (GRID_RAPIDE, SEEDS[:1]) if rapide else (GRID, SEEDS)
    if rapide:
        train_df, val_df = train_df.head(64), val_df.head(64)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ids_tr, ids_va = encode(tok, build_text(train_df)), encode(tok, build_text(val_df))
    n_trunc = sum(len(x) >= MAX_LENGTH for x in ids_tr)
    print(f"Matériel : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'pas de GPU'}) ; "
          f"entraînement {len(train_df)} avis, validation {len(val_df)} avis ; {n_trunc} avis tronqués à {MAX_LENGTH} "
          f"tokens ; réglages (taux, époques) {grid} x {len(seeds)} graines", flush=True)
    Y_tr = plainte_matrix(train_df["motifs"])
    Y_va = to_matrix(list(val_df["motifs"]))
    path = out_dir / "grille.jsonl"
    done = {(r["lr"], n_epochs_of(r), r["seed"]) for r in read_runs(path)}
    for lr, n_epochs in grid:
        for seed in seeds:
            if (lr, n_epochs, seed) in done:
                print(f"  taux {lr:g} calendrier {n_epochs} graine {seed} : déjà fait, sauté", flush=True)
                continue
            epochs, probas, t0 = [], [], time.perf_counter()

            def on_epoch(epoch, model, loss):
                P = predict_proba(model, ids_va, tok.pad_token_id, device)
                pred = to_motifs(P)
                Pm = to_matrix(pred)
                s = scores(Y_va, Pm)
                epochs.append({"epoch": epoch, "train_loss": round(loss, 4), "macro_f1": round(s["macro_f1"], 4),
                               "exact": round(exact_match(Y_va, Pm), 4),
                               "invalides": sum(bool(check_rules(p)) for p in pred),
                               "f1": {m: round(s["par_motif"][m]["f1"], 3) for m in MOTIFS},
                               "duree_s": round(time.perf_counter() - t0, 1)})
                probas.append(P)
                print(f"  taux {lr:g} calendrier {n_epochs} graine {seed} époque {epoch} : perte {loss:.4f}, "
                      f"macro-F1 val {s['macro_f1']:.3f}, accord exact {epochs[-1]['exact']:.3f}, "
                      f"{epochs[-1]['duree_s']:.0f} s", flush=True)

            train(ids_tr, Y_tr, lr, n_epochs, seed, device, tok.pad_token_id, on_epoch)
            np.save(out_dir / f"val_proba_lr{lr:g}_ep{n_epochs}_graine{seed}.npy", np.stack(probas))
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"lr": lr, "n_epochs": n_epochs, "seed": seed, "model": MODEL_NAME,
                                    "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE, "n_train": len(train_df),
                                    "n_val": len(val_df), "device": str(device), "date": date.today().isoformat(),
                                    "epochs": epochs}, ensure_ascii=False) + "\n")
    runs = read_runs(path)
    print_bilan(runs)
    best = choose_config(runs)
    (out_dir / "choix.json").write_text(json.dumps(best, indent=1), encoding="utf-8")
    print(f"\nChoix sur la validation : {best}")
    print(f"Résultats dans {out_dir}. Ne pas lancer `final` avant d'avoir validé la grille (GRILLE_VALIDEE).")


def run_compare(out_dir=OUT_DIR):
    """TF-IDF contre DistilBERT sur les 996 avis de validation, bootstrap apparié (cours 2, section 7).
    Les deux modèles sont entraînés sur les mêmes 3 983 avis ; DistilBERT : le réglage choisi par la grille."""
    from src.tfidf_lr import build_pipeline, load_config
    out_dir = Path(out_dir)
    runs = read_runs(out_dir / "grille.jsonl")
    if not runs:
        raise SystemExit(f"Aucun résultat de grille dans {out_dir} : copie-y grille.jsonl et les val_proba_*.npy.")
    best = choose_config(runs)
    train_df, val_df = load_data()
    ref = list(val_df["motifs"])
    Y = to_matrix(ref)
    tfidf = build_pipeline(**load_config()).fit(build_text(train_df), plainte_matrix(train_df["motifs"]))
    pred_tf = to_motifs(tfidf.predict_proba(build_text(val_df)))
    P_tf = to_matrix(pred_tf)
    print(f"Validation : {len(val_df)} avis, étiquettes du professeur. DistilBERT : {best}")
    print(f"TF-IDF : macro-F1 {macro_f1(Y, P_tf):.3f}, accord exact {exact_match(Y, P_tf):.3f}")
    preds_b = {}
    for seed in SEEDS:
        path = proba_path(out_dir, best["lr"], best["n_epochs"], seed)
        if not path.exists():
            print(f"  graine {seed} : {path.name} absent, sauté")
            continue
        proba = np.load(path)
        if proba.shape[1] != len(val_df):
            raise SystemExit(f"{path.name} : {proba.shape[1]} avis au lieu de {len(val_df)}")
        preds_b[seed] = to_motifs(proba[best["epochs"] - 1])
        P_b = to_matrix(preds_b[seed])
        d, lo, hi = paired_bootstrap(Y, P_tf, P_b, macro_f1)
        de, loe, hie = paired_bootstrap(Y, P_tf, P_b, exact_match)
        print(f"DistilBERT graine {seed} : macro-F1 {macro_f1(Y, P_b):.3f} ; écart DistilBERT - TF-IDF : macro-F1 "
              f"{d:+.3f} [{lo:+.3f} ; {hi:+.3f}], accord exact {de:+.3f} [{loe:+.3f} ; {hie:+.3f}]")
    if not preds_b:
        return
    seed = SEED_FINAL if SEED_FINAL in preds_b else min(preds_b)
    s_tf, s_b = scores(Y, P_tf), scores(Y, to_matrix(preds_b[seed]))
    print(f"\nF1 par motif (DistilBERT graine {seed}) :")
    print(f"  {'motif':20s} {'TF-IDF':>7s} {'DistilBERT':>10s} {'support':>8s}")
    for m in MOTIFS:
        print(f"  {m:20s} {s_tf['par_motif'][m]['f1']:7.3f} {s_b['par_motif'][m]['f1']:10.3f} "
              f"{s_tf['par_motif'][m]['support']:8d}")
    for name, pred in [("TF-IDF", pred_tf), (f"DistilBERT graine {seed}", preds_b[seed])]:
        c = confusions(ref, pred, "fraicheur_securite")
        print(f"fraicheur_securite, {name} : {c['manques']} manqués sur {c['positifs']} ({c['en_trop']} en trop) ; "
              "à la place : " + (", ".join(f"{'+'.join(k)} {v}" for k, v in c["a_la_place"].most_common(3)) or "-"))


def run_final(out_dir=OUT_DIR, validee=GRILLE_VALIDEE, pool_path=POOL):
    out_dir = Path(out_dir)
    out = out_dir / "preds_test_distilbert.jsonl"
    if out.exists():
        raise SystemExit(f"{out} existe déjà : le test se mesure une seule fois par modèle figé (cours 3, section 6).")
    if not validee:
        raise SystemExit("La grille doit d'abord être expliquée et validée (cours 4) : GRILLE_VALIDEE = True en tête de "
                         "src/distilbert.py, et décision consignée dans CONVENTIONS.md.")
    runs = read_runs(out_dir / "grille.jsonl")
    if not runs:
        raise SystemExit(f"Aucun résultat de grille dans {out_dir} : lance d'abord `grid`.")
    import torch                                        # après les garde-fous : ils se testent sans torch
    from transformers import AutoTokenizer
    best = choose_config(runs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df = load_data()
    fit_df = pd.concat([train_df, val_df])              # décision B : ré-entraînement sur les 4 979 avis
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    t0 = time.perf_counter()
    model = train(encode(tok, build_text(fit_df)), plainte_matrix(fit_df["motifs"]), best["lr"], best["n_epochs"],
                  SEED_FINAL, device, tok.pad_token_id, stop_after=best["epochs"])
    fit_s = time.perf_counter() - t0

    test = pd.read_parquet(pool_path).query("part == 'test'")
    ids_te = encode(tok, build_text(test))
    t0 = time.perf_counter()
    pred = to_motifs(predict_proba(model, ids_te, tok.pad_token_id, device))
    batch_ms = (time.perf_counter() - t0) / len(ids_te) * 1000
    t0 = time.perf_counter()
    for x in ids_te[:100]:
        predict_proba(model, [x], tok.pad_token_id, device, batch_size=1)
    one_ms = (time.perf_counter() - t0) / min(100, len(ids_te)) * 1000

    cfg = {"model": MODEL_NAME, "lr": best["lr"], "n_epochs": best["n_epochs"], "epochs": best["epochs"],
           "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE, "seed": SEED_FINAL}
    with open(out, "w", encoding="utf-8") as f:
        for rid, motifs in zip(test["review_id"], pred):
            f.write(json.dumps({"review_id": rid, "model": "distilbert", "config": cfg, "refit": True,
                                "n_train": len(fit_df), "motifs": motifs, "problems": check_rules(motifs)},
                               ensure_ascii=False) + "\n")
    model.save_pretrained(out_dir / "modele")
    tok.save_pretrained(out_dir / "modele")
    print(f"Configuration {cfg}, entraînée sur {len(fit_df)} avis en {fit_s / 60:.1f} min ({device}).")
    print(f"{len(pred)} avis de test prédits, {sum(bool(check_rules(m)) for m in pred)} invalides -> {out}")
    print(f"Latence ({device}) : {batch_ms:.2f} ms par avis par lot ; {one_ms:.1f} ms un avis à la fois")
    print(f"Modèle enregistré dans {out_dir / 'modele'}. Copier {out.name} dans data/ en local, puis :")
    print(f"  python -m src.metrics eval data/{out.name}")


def main():
    parser = argparse.ArgumentParser(description="DistilBERT fine-tuné (cours 4)")
    parser.add_argument("cmd", choices=["grid", "final", "paquet", "compare"])
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="dossier des résultats (sur Drive dans Colab)")
    parser.add_argument("--rapide", action="store_true", help="grid : test de fumée sur 64 avis, 1 époque")
    args = parser.parse_args()
    if args.cmd == "grid":
        run_grid(Path(args.out_dir), args.rapide)
    elif args.cmd == "final":
        run_final(Path(args.out_dir))
    elif args.cmd == "compare":
        run_compare(Path(args.out_dir))
    else:
        out = make_bundle()
        print(f"{out} : {len(BUNDLE_FILES)} fichiers, {out.stat().st_size / 1e6:.1f} Mo")


if __name__ == "__main__":
    main()
