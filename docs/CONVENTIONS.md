# Conventions du projet (résumé des décisions du cours 1)

## Motifs (version v2, validée le 12 oct. 2026 sur les 200 avis d'explo)
| Motif | Définition | Frontière |
| --- | --- | --- |
| reception | problème lié au transport ou au colis : retard, colis perdu ou incomplet, produit cassé, écrasé, fondu ou qui fuit à l'arrivée, emballage abîmé, ouvert ou mal conçu | produit arrivé intact mais mauvais -> gout_texture ou fraicheur_securite |
| gout_texture | goût, texture ou odeur d'un produit arrivé intact et non périmé | rance, rassis, moisi -> fraicheur_securite |
| fraicheur_securite | périmé, rassis, rance, moisi, insectes, corps étranger, effet sur la santé, allergène non signalé | prioritaire sur gout_texture |
| non_conforme | ne correspond pas à la description : mauvais produit, format ou quantité annoncée, ingrédient non annoncé, étiquetage trompeur | reçu ce qui était annoncé mais jugé trop peu pour le prix -> prix |
| prix | trop cher, mauvais rapport quantité/prix, hausse de prix | |
| autre | aucun des motifs ci-dessus, y compris aucune plainte ; jamais combiné avec un autre | |

Multi-étiquette : un avis peut porter plusieurs motifs.
Seuils de l'étape 1 : chaque motif >= 5 % des avis, aucun motif > 50 %, « autre » <= 25 %. Sinon : nouvelle version du prompt.
Le seuil de 5 % sert surtout à la mesure : sur les 300 avis de test, 5 % = 15 avis positifs, le minimum pour un F1 par motif
dont l'intervalle de confiance reste utilisable (environ ± 0,2 sur le rappel).
Le plafond de 50 % sert à repérer un motif fourre-tout. Précision du 12 oct. 2026 (écrite après la mesure v2, en toute
transparence) : un motif au-dessus de 50 % est gardé s'il est homogène, c'est-à-dire si au moins 80 % de ses avis relèvent
d'une même décision métier ; sinon on le coupe. `verdict` affiche l'alerte, la vérification d'homogénéité se fait à la main.

Résultats v2 (prompt v2, mistral-small, 200 avis d'explo, 12 oct. 2026) : gout_texture 56 % (IC 95 % 49-63 %),
fraicheur_securite 15 %, reception 14,5 %, non_conforme 12,5 %, prix 11,5 %, autre 6,5 % ; invalides 1,5 % ;
accord avec la relecture : 30/30 pour Martin, 28/30 pour un second relecteur (n° 7 effet sur la santé -> autre ; n° 24 lait -> gout_texture, prix).
gout_texture gardé : 85 % de ses avis parlent de goût ou de texture (tri par mots-clés sur motif_libre). Le couper en
gout / texture recréerait une frontière floue (« aqueux », « fade et sec ») pour un faible gain métier.
Défaut connu du prompt v2, à corriger en v3 avant l'étape 3 : les effets sur la santé partent parfois dans « autre » (2/200).

Historique v1 (livraison, emballage, qualite, prix, service, autre), mesuré sur les 200 avis d'explo le 12 oct. 2026 :
qualite 77 %, emballage 15 %, prix 14,5 %, livraison 6 %, autre 3,5 %, service 0 % ; accord avec la relecture de Martin 27/30.
Raisons du changement : service absent de ces données (les remarques sur le vendeur vont dans les avis vendeurs d'Amazon),
qualite trop large, frontière livraison / emballage incohérente (21 avis « arrivés abîmés », 6 étiquetages différents).

## Annotation et évaluation (étape 2, décisions des 12 et 13 oct. 2026)
- Guide : `docs/guide_annotation.md`, v1 gelé le 12 oct. 2026 avant le premier avis de test. Arbre « une plainte, un motif :
  celui de sa cause » (ordre reception, fraicheur_securite, non_conforme, gout_texture, prix) ; une conséquence compte à part
  si elle concerne une autre équipe ; le titre compte autant que le texte ; ingrédients et tailles « annoncés par défaut » ;
  13 groupes de cas limites tranchés sur explo. Une règle née d'un avis de test entre dans le guide (v1.x), jamais dans un
  prompt avant la mesure.
- Référence : passe 1 = 300 avis de test annotés par Martin avec le guide v1, lus dans un fichier d'aperçu puis saisis dans
  l'outil le 13 oct. (~7 h estimées ; `duree_s` = temps de saisie seulement). Parts : gout_texture 57 %, fraicheur_securite
  15,3 %, reception 14 %, prix 13 %, autre 10,7 %, non_conforme 8 % (chaque motif >= 24 positifs). Passe 2 : 50 avis tirés
  (graine 42), au moins 3 jours après, dans l'outil, sans les notes de passe 1, et avant de lire les étiquettes du LLM sur
  le test avis par avis. Passe 3 : arbitrage des désaccords ; référence finale = passe 3 si elle existe, sinon passe 1.
  `data/annotations_test.jsonl` est suivi par git (exception dans `.gitignore`).
- Métrique principale (13 oct. 2026, avant tout résultat sur le test) : macro-F1 sur les 6 motifs, avec IC 95 % par bootstrap
  percentile (2 000 tirages d'avis, graine 42). À côté : P / R / F1 / support par motif, micro-F1, accord exact, kappa
  référence / modèle par motif. Réponses manquantes ou invalides d'un modèle = liste vide (comptées comme erreurs) ;
  division par zéro = 0.
- Comparaisons : « B bat A » seulement si l'IC du bootstrap apparié de l'écart exclut 0 ; sinon, décision sur le coût et la
  latence. Comparaison principale (validée le 13 oct. 2026, avant tout résultat) : macro-F1 du meilleur modèle entraîné
  (TF-IDF + régression logistique ou DistilBERT, choisi sur un jeu de validation, jamais sur le test) contre mistral-small
  avec le prompt v3, par bootstrap apparié sur les 300 avis de test. Toutes les autres comparaisons sont exploratoires.

## Prompt v3 (étape 3, décisions du 13 oct. 2026)
- Contenu (`prompts/v3.txt`) : liste v2 + règles du guide v1, une ligne par règle + 6 exemples d'explo dans le prompt
  système (n° 178, 169, 149, 163 en extrait, 64, 61 : un par grappe d'erreurs de v2). 2 970 caractères, ~700 tokens
  d'entrée estimés. Budget : 4 000 caractères au plus (au-delà, la limite de 20 000 tokens/min ralentit l'étiquetage).
- Modèle figé : `mistral-small-2603` (cible de l'alias `mistral-small-latest`, vérifiée le 13 oct. 2026). Le même modèle
  et le même prompt étiquettent le train et, après la passe 2, le test.
- Diagnostic de v2 sur les 35 avis d'explo tranchés dans le guide : 10/35 conformes (échantillon de cas difficiles, ne
  mesure pas l'exactitude globale). Erreurs groupées : non_conforme à tort (8 avis), autre manqué (7), plaintes
  secondaires oubliées (4), reception et fraicheur_securite manqués.
- Critère d'acceptation, fixé avant tout appel v3 :
  1. au moins 22 réussis sur les 29 avis tranchés dans le guide qui ne servent pas d'exemples (v2 : 10/29).
     Réussi = étiquette identique quand le guide donne l'étiquette complète (24 a, 44 g, 50 r+f, 58 a, 66 f+r, 86 a,
     87 r+g, 112 a, 146 a, 147 r+g+p, 152 f+g, 160 f, 183 a) ; sinon, motif requis présent (r : 38, 75, 101, 180, 181 ;
     f : 7, 11, 22, 92, 96, 129, 134, 157 ; n : 26 ; g : 19, 65) ;
  2. moins de 2 % d'invalides sur les 200 avis d'explo ;
  3. sur les avis où v2 et v3 diffèrent, annotés à l'aveugle par Martin avec le guide, v3 a raison au moins aussi
     souvent que v2.
- Question ouverte : n° 38 (« kind of pricey per bottle ») -> r ou r + p ? Sans effet sur le critère (seul r est vérifié).

## Sortie du LLM
- Schéma Pydantic `Etiquetage` : `evidence` (str), puis `motif_libre` (str, 3-6 mots), puis `motifs` (liste).
  L'ordre compte : la justification est générée avant l'étiquette.
- Règles vérifiées après coup (`check_rules`) : liste non vide, « autre » seul, pas de doublon. Invalides < 2 %.
- Appels : temperature=0, random_seed=42, max_tokens=200. Prompts versionnés dans `prompts/vN.txt`.
- Erreurs : 401 -> arrêt ; 429 / 5xx / réseau -> backoff 1, 2, 4, 8 s, 5 essais max.
- Sauvegarde : une ligne JSONL par avis, écrite immédiatement ; reprise en sautant les review_id déjà étiquetés
  avec succès. Les lignes `{"review_id", "error"}` ne comptent pas comme faites : relancer le script refait ces avis
  (décision du 11 oct. 2026, option B). Le fichier peut donc contenir une ligne `error` puis une ligne réussie
  pour le même avis ; `report.py` ignore les lignes `error`.
  Champs : review_id, prompt_version, model, evidence, motif_libre, motifs, problems,
  prompt_tokens, completion_tokens, latency_s.
- Réponse vide (`parsed` à None) : evidence et motif_libre à None, motifs à [] -> `problems = ["liste_vide"]`
  (compté comme invalide, pas refait).

## Données
- Amazon Reviews 2023, Grocery_and_Gourmet_Food (14,3 M avis, 6 Go), lu en streaming.
- Filtre : notes 1 à 3, >= 5 mots, HTML nettoyé, dédoublonnage texte et review_id.
- review_id = f"{user_id}_{parent_asin}_{timestamp}" (le jeu n'a pas d'identifiant d'avis).
- Réservoir 20 000 avis, graine 42, découpé une fois : explo 200 / test 300 / train 5 000 / reserve.
- Le jeu `test` est GELÉ : annoté à la main, jamais dans un prompt (ni en exemple), jamais pour entraîner.
  Les exemples few-shot viennent de `explo`.
- Vérifier la diversité des 500 000 premiers avis (produits, années) : le streaming lit dans l'ordre du fichier.

## Coûts (hypothèses à remplacer par les mesures réelles)
- Mesuré le 12 oct. 2026 (prompt v1, mistral-small, 200 avis) : 266 tokens en entrée, 47 en sortie en moyenne ;
  latence médiane 0,95 s, p95 3,1 s (retries probables, non enregistrés : ajouter un champ `attempts`).
- Prompt v2 (1 121 caractères contre 795) : 335 tokens en entrée, 43 en sortie ; latence médiane 0,87 s, p95 1,7 s ;
  0,015 $ pour 200 avis, soit ~7,6 $ pour 100 000 avis avec les prix ci-dessous (prix encore à vérifier dans la console).
- Hypothèse initiale : ~600 tokens de prompt système + ~80 d'avis en entrée, ~50 en sortie.
- Mistral Small 4 : 0,15 $ / 0,60 $ par million de tokens (source tierce, 6 oct. 2026).
- 200 avis ~0,03 $ ; 5 000 avis ~0,66 $ ; 100 000 avis ~13 $.
