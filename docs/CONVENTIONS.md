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
- Passe 2 faite le 14 oct. 2026, 2 jours après l'annotation hors outil et 1 jour après la saisie, au lieu de 3 jours :
  écart décidé par Martin (la saisie, à 1,2 s par avis en médiane, n'était pas une relecture). 50 avis, 0 doute, durée
  médiane 18 s par avis (27 min). Accord exact 0,70 [0,58 ; 0,82], 15 avis en désaccord. Kappa (positifs passe 1 /
  passe 2) : gout_texture 0,68 [0,48 ; 0,84] (26 / 24) ; ordres de grandeur seulement : reception 0,49 (7 / 4),
  fraicheur_securite 0,39 (5 / 4), prix 0,74 (8 / 5) ; non interprétables (trop peu de positifs) : non_conforme 0,88
  (4 / 5), autre 0,46 (6 / 9). En passe 2, moins de motifs secondaires (reception 7 -> 4, prix 8 -> 5) et plus d'autre
  (6 -> 9). L'accord est peut-être gonflé par la reconnaissance des avis (délai court, annotation rapide).
  Plafond de mesure (cours 2, section 5) : 2e(1 − e) = 0,30 donne e ≈ 0,18 ; environ 18 % des avis de la référence
  portent une étiquette instable, et même un modèle parfait n'aurait qu'environ 82 % d'accord exact avec la passe 1
  (hypothèse d'erreurs indépendantes, optimiste).
- Passe 3 faite le 14 oct. 2026 (15 désaccords arbitrés, commit `ccb2d66`), avant l'étiquetage du test par le LLM.
  Référence finale : passe 3 sur ces 15 avis, passe 1 sur les 285 autres. Guide inchangé (v1).
- Métrique principale (13 oct. 2026, avant tout résultat sur le test) : macro-F1 sur les 6 motifs, avec IC 95 % par bootstrap
  percentile (2 000 tirages d'avis, graine 42). À côté : P / R / F1 / support par motif, micro-F1, accord exact, kappa
  référence / modèle par motif. Réponses manquantes ou invalides d'un modèle = liste vide (comptées comme erreurs) ;
  division par zéro = 0.
- Comparaisons : « B bat A » seulement si l'IC du bootstrap apparié de l'écart exclut 0 ; sinon, décision sur le coût et la
  latence. Comparaison principale (validée le 13 oct. 2026, avant tout résultat) : macro-F1 du meilleur modèle entraîné
  (TF-IDF + régression logistique ou DistilBERT, choisi sur un jeu de validation, jamais sur le test) contre mistral-small
  (`mistral-small-2603`) avec le prompt v4 (retenu le 13 oct. 2026 à la place de v3, prévu au départ : voir section
  Prompts v3 et v4), par bootstrap apparié sur les 300 avis de test. Toutes les autres comparaisons sont exploratoires.

## Prompts v3 et v4 (étape 3, décisions du 13 oct. 2026)
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
- Résultats v3 (200 avis d'explo) : critère 1 21/29 (v2 10/29), critère 2 3,5 % d'invalides (7 avis, tous
  `autre_combine` ; v2 1,5 %) -> **v3 refusé**, critère 3 non mesuré. Cause : la définition d'autre listait les plaintes
  qui ne donnent aucun motif, et le modèle a utilisé autre comme étiquette de ces plaintes (n° 44 : g + a).
  Mesures : attempts = 1 sur 200, latence médiane 0,71 s, p95 1,26 s ; 754 tokens d'entrée, 47 de sortie.
  Parts : gout_texture 53,5 %, fraicheur_securite 19 %, reception 18 %, prix 13 %, autre 13 %, non_conforme 5,5 %.
  Les 7 autres échecs du critère 1 ont chacun une cause différente (24, 58, 66, 87, 92, 160, 180) : non corrigés.
- v4 (`prompts/v4.txt`, 3 135 caractères) = v3 + une seule correction, la seule cause commune à plusieurs avis :
  autre réservé à la liste vide ; les plaintes sans motif passent à l'étape 4 de « How to decide ». Mêmes exemples,
  mêmes trois critères, toujours contre v2.
- Règle de repli, fixée avant le run v4 : plus d'itération après v4. Si v4 rate seulement le critère 1, le train est
  étiqueté avec v4 à condition que les critères 2 et 3 soient remplis ; l'échec est noté comme limite connue (ici et
  dans le livrable). Si le critère 2 ou 3 échoue : arrêt et discussion. Raison : les 29 avis du critère 1 ont servi à
  écrire le guide, v3 et v4 (le critère le plus contaminé) ; le critère 3 porte sur des avis jamais regardés.
- Résultats v4 (200 avis d'explo) : critère 2 0 % d'invalides (rempli) ; critère 1 18/29 (raté ; v3 21, v2 10).
  Seuls les avis où un seul prompt a raison comptent : v3 contre v4 6 contre 3 (test du signe p = 0,51), v4 contre v2
  10 contre 2 (p = 0,04). 13 des 29 cas ont changé d'étiquette entre v3 et v4 pour 2 lignes de prompt modifiées : sur
  les cas limites, le modèle est instable. Critère 3 : 29 avis divergents, annotés par un second annotateur et non par Martin
  (écart au critère ; 14 doutes sur 29 ; cet annotateur avait rédigé le brouillon du prompt et connaissait l'étiquette
  v3 de quelques avis par les motif_libre de v3) : v4 juste 8, v2 juste 12, les deux faux 9 (p = 0,50) -> raté. Défauts vus : v4 ajoute un second
  motif à une plainte unique (6 avis) et classe en autre des plaintes de goût peu détaillées (2 avis ; aussi n° 19 et 181).
  Mesures : attempts = 1 sur 200, latence médiane 0,70 s, p95 1,49 s ; 790 tokens d'entrée, 45,5 de sortie.
  Parts : gout_texture 53,5 %, fraicheur_securite 18,5 %, reception 17,5 %, prix 14 %, autre 9,5 %, non_conforme 4,5 %
  (référence de Martin sur le test : non_conforme 8 %).
- **Décision (13 oct. 2026, Martin), contraire à la règle de repli** (qui prévoyait arrêt et discussion) : le train est
  étiqueté avec v4, et v4 est le prompt de la comparaison principale. Raisons : sur les 58 avis mesurés (29 du guide +
  29 divergents), v4 gagne 18 contre 14 (p = 0,60, match nul) ; 0 % d'invalides ; plus proche du guide, qui a servi à
  annoter la référence du test. Limites connues, à reporter dans le livrable : critères 1 et 3 ratés, non_conforme
  peut-être sous-étiqueté, empilement de motifs. Contrôle facultatif : Martin annote les 29 avis divergents (document
  « Explo : 29 avis divergents v2 / v4 et prompt v4 ») ; on mesure alors l'accord entre les deux annotateurs.
- Train étiqueté avec v4 (13-14 oct. 2026, `data/labels_train_v4.jsonl`) : 5 000 / 5 000, 0 erreur ; invalides 0,4 %
  (21 avis, tous `autre_combine`) ; attempts = 1 pour 4 999 avis, 5 pour 1 ; latence médiane 0,67 s, p95 1,3 s ;
  790,6 tokens d'entrée, 45,6 de sortie ; coût 0,7296 $ (14,59 $ pour 100 000 avis). Parts : gout_texture 59 %,
  fraicheur_securite 15,5 %, prix 14,5 %, reception 11,4 %, non_conforme 8,7 %, autre 8,3 %. Plus proches de la
  référence de Martin sur le test que celles d'explo : la crainte d'un non_conforme sous-étiqueté (4,5 % sur explo)
  venait du bruit d'échantillonnage sur 200 avis (écart-type d'une part : √(p(1−p)/200), ±5 points à 95 % pour 15 %).
  Limite connue : dans les motif_libre des avis « autre », quelques vraies plaintes avalées par l'étape 4 (« difficile
  à ouvrir », « goût non apprécié ») ; estimation grossière ~1 % du train.

## Résultats sur le test (300 avis, référence finale de Martin)
| Modèle | macro-F1 [IC 95 %] | micro-F1 [IC 95 %] | Accord exact [IC 95 %] | Coût / 100 000 avis | Latence médiane / p95 | Date |
| --- | --- | --- | --- | --- | --- | --- |
| `mistral-small-2603` + prompt v4 | 0,770 [0,716 ; 0,816] | 0,843 [0,809 ; 0,876] | 0,763 [0,717 ; 0,810] | 14,6 $ | 0,76 s / 1,19 s | 14 oct. 2026 |
| Repère humain : Martin, passe 2 contre passe 1 (50 avis) | – | – | 0,70 [0,58 ; 0,82] | – | 18 s par avis | 14 oct. 2026 |

- mistral-small v4, F1 par motif [IC 95 %] (support) : reception 0,886 [0,800 ; 0,951] (42), gout_texture 0,909
  [0,876 ; 0,940] (171), fraicheur_securite 0,901 [0,829 ; 0,961] (46), prix 0,800 [0,696 ; 0,889] (39),
  non_conforme 0,560 [0,372 ; 0,702] (24 ; 14 vrais positifs, 12 faux positifs, 10 manqués), autre 0,566
  [0,390 ; 0,710] (32 ; 15 trouvés, 17 manqués, 6 en trop). Kappa Martin / modèle : reception 0,87, gout_texture 0,78,
  fraicheur_securite 0,88, non_conforme 0,52, prix 0,77, autre 0,53. 1 invalide (compté comme liste vide), 0,0436 $.
- Lecture : sur 4 motifs sur 6, le LLM s'accorde avec la référence au moins autant que Martin avec lui-même
  (gout_texture, seul kappa humain interprétable : 0,78 contre 0,68). Ses deux points faibles sont les deux frontières
  que le guide a dû trancher (« annoncé par défaut » pour non_conforme, « plainte vague » pour autre) et les deux motifs
  les plus rares ; le macro-F1 (0,77) les pèse autant que les autres, d'où l'écart avec le micro-F1 (0,84). Réserves :
  repère humain mesuré sur 50 avis seulement, passe 2 rapide, kappas humains des motifs rares non interprétables.
- Ce macro-F1 est la cible de la comparaison principale (meilleur modèle entraîné sur `labels_train_v4`, bootstrap
  apparié).

## Sortie du LLM
- Schéma Pydantic `Etiquetage` : `evidence` (str), puis `motif_libre` (str, 3-6 mots), puis `motifs` (liste).
  L'ordre compte : la justification est générée avant l'étiquette.
- Règles vérifiées après coup (`check_rules`) : liste non vide, « autre » seul, pas de doublon. Invalides < 2 %.
- Appels : temperature=0, random_seed=42, max_tokens=200. Prompts versionnés dans `prompts/vN.txt`.
  Modèle figé depuis le 13 oct. 2026 : `mistral-small-2603` (avant : l'alias `mistral-small-latest`).
- Erreurs : 401 -> arrêt ; 429 / 5xx / réseau -> backoff 1, 2, 4, 8 s, 5 essais max.
- Sauvegarde : une ligne JSONL par avis, écrite immédiatement ; reprise en sautant les review_id déjà étiquetés
  avec succès. Les lignes `{"review_id", "error"}` ne comptent pas comme faites : relancer le script refait ces avis
  (décision du 11 oct. 2026, option B). Le fichier peut donc contenir une ligne `error` puis une ligne réussie
  pour le même avis ; `report.py` ignore les lignes `error`.
  Champs : review_id, prompt_version, model, evidence, motif_libre, motifs, problems,
  prompt_tokens, completion_tokens, latency_s, attempts (depuis le 13 oct. 2026 ; `model` = `rep.model`, le nom
  annoncé par l'API, qui peut n'être que l'écho du nom demandé).
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
- Mistral Small 4 : 0,15 $ / 0,60 $ par million de tokens (source tierce, 6 oct. 2026 ; mêmes chiffres sur la page
  du modèle de docs.mistral.ai le 13 oct. 2026, qui ne précise pas lequel est l'entrée ; console pas encore vérifiée).
- Mesuré le 14 oct. 2026 (prompt v4, `mistral-small-2603`, 5 000 avis de train) : 790,6 tokens d'entrée, 45,6 de sortie,
  0,7296 $, soit 14,59 $ pour 100 000 avis ; ~4 h d'appels (pause 2,2 s + latence médiane 0,67 s par avis).
- 200 avis ~0,03 $ ; 5 000 avis ~0,66 $ ; 100 000 avis ~13 $.
