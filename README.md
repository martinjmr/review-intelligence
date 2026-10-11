# Review Intelligence

Classer des avis clients par motif d'insatisfaction, comparer trois approches
(TF-IDF + régression logistique, DistilBERT fine-tuné, LLM via API) sur la qualité,
le coût et la latence, puis rédiger des réponses aux avis négatifs.

Données : Amazon Reviews 2023 (McAuley Lab), catégorie Grocery_and_Gourmet_Food, avis notés 1 à 3.

## État

- [ ] Étape 1 : vérifier la liste de motifs sur 200 avis (en cours)
- [ ] Étape 2 : annoter 300 avis de test à la main
- [ ] Étape 3 : étiqueter 5 000 avis par LLM, mesurer l'exactitude sur les 300
- [ ] Étape 4 : TF-IDF + régression logistique, DistilBERT, tableau qualité / coût / latence

## Installation (Windows)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      (puis colle ta clé Mistral dans .env)
python -m pytest -q         (tous les tests doivent passer)
```

## Étape 1

```
python -m src.data --n-read 500000                      # -> data/pool.parquet
python -m src.label_llm --part explo --prompt v1 --limit 30
python -m src.report data/labels_explo_v1.jsonl
python -m src.label_llm --part explo --prompt v1        # les 200, reprend où il s'est arrêté
python -m src.report data/labels_explo_v1.jsonl
```
