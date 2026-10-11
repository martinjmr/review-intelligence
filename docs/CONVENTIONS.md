# Conventions du projet (résumé des décisions du cours 1)

## Motifs (version v1, à confirmer à l'étape 1)
| Motif | Définition | Frontière |
| --- | --- | --- |
| livraison | délai, colis perdu ou non reçu, produit abîmé par le transport (fondu, chaud) | boîte écrasée ou qui fuit -> emballage |
| emballage | emballage abîmé, ouvert, qui fuit, mal conçu | produit lui-même mauvais -> qualite |
| qualite | goût, texture, fraîcheur, périmé, corps étranger, non conforme à la description | |
| prix | trop cher, mauvais rapport quantité/prix, hausse de prix | |
| service | service client, vendeur, retour ou remboursement | |
| autre | aucun des motifs ci-dessus ; jamais combiné avec un autre | |

Multi-étiquette : un avis peut porter plusieurs motifs.
Seuils de l'étape 1 : chaque motif >= 5 % des avis, « autre » <= 25 %. Sinon : prompt v2 et nouvelle liste.

## Sortie du LLM
- Schéma Pydantic `Etiquetage` : `evidence` (str), puis `motif_libre` (str, 3-6 mots), puis `motifs` (liste).
  L'ordre compte : la justification est générée avant l'étiquette.
- Règles vérifiées après coup (`check_rules`) : liste non vide, « autre » seul, pas de doublon. Invalides < 2 %.
- Appels : temperature=0, random_seed=42, max_tokens=200. Prompts versionnés dans `prompts/vN.txt`.
- Erreurs : 401 -> arrêt ; 429 / 5xx / réseau -> backoff 1, 2, 4, 8 s, 5 essais max.
- Sauvegarde : une ligne JSONL par avis, écrite immédiatement ; reprise en sautant les review_id déjà présents.
  Champs : review_id, prompt_version, model, evidence, motif_libre, motifs, problems,
  prompt_tokens, completion_tokens, latency_s.

## Données
- Amazon Reviews 2023, Grocery_and_Gourmet_Food (14,3 M avis, 6 Go), lu en streaming.
- Filtre : notes 1 à 3, >= 5 mots, HTML nettoyé, dédoublonnage texte et review_id.
- review_id = f"{user_id}_{parent_asin}_{timestamp}" (le jeu n'a pas d'identifiant d'avis).
- Réservoir 20 000 avis, graine 42, découpé une fois : explo 200 / test 300 / train 5 000 / reserve.
- Le jeu `test` est GELÉ : annoté à la main, jamais dans un prompt (ni en exemple), jamais pour entraîner.
  Les exemples few-shot viennent de `explo`.
- Vérifier la diversité des 500 000 premiers avis (produits, années) : le streaming lit dans l'ordre du fichier.

## Coûts (hypothèses à remplacer par les mesures réelles)
- ~600 tokens de prompt système + ~80 d'avis en entrée, ~50 en sortie.
- Mistral Small 4 : 0,15 $ / 0,60 $ par million de tokens (source tierce, 6 oct. 2026).
- 200 avis ~0,03 $ ; 5 000 avis ~0,66 $ ; 100 000 avis ~13 $.
