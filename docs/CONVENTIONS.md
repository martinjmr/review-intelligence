# Conventions du projet : décisions et résultats datés

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
  Plafond de mesure : 2e(1 − e) = 0,30 donne e ≈ 0,18 ; environ 18 % des avis de la référence
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
  peut-être sous-étiqueté, empilement de motifs. Contrôle facultatif (annotation des 29 avis divergents par Martin,
  pour mesurer l'accord entre les deux annotateurs) : écarté le 14 oct. 2026, non fait.
- Train étiqueté avec v4 (13-14 oct. 2026, `data/labels_train_v4.jsonl`) : 5 000 / 5 000, 0 erreur ; invalides 0,4 %
  (21 avis, tous `autre_combine`) ; attempts = 1 pour 4 999 avis, 5 pour 1 ; latence médiane 0,67 s, p95 1,3 s ;
  790,6 tokens d'entrée, 45,6 de sortie ; coût 0,7296 $ (14,59 $ pour 100 000 avis). Parts : gout_texture 59 %,
  fraicheur_securite 15,5 %, prix 14,5 %, reception 11,4 %, non_conforme 8,7 %, autre 8,3 %. Plus proches de la
  référence de Martin sur le test que celles d'explo : la crainte d'un non_conforme sous-étiqueté (4,5 % sur explo)
  venait du bruit d'échantillonnage sur 200 avis (écart-type d'une part : √(p(1−p)/200), ±5 points à 95 % pour 15 %).
  Limite connue : dans les motif_libre des avis « autre », quelques vraies plaintes avalées par l'étape 4 (« difficile
  à ouvrir », « goût non apprécié ») ; estimation grossière ~1 % du train.

## Modèles entraînés (étape 4, décisions du 14 oct. 2026)
- Étiquettes d'entraînement : `data/labels_train_v4.jsonl`, lignes `error` ignorées, les 21 invalides exclus (option A,
  plutôt que les réparer en retirant « autre » ou les garder : aucune étiquette inventée ; 0,4 % du train, effet
  attendu inférieur à 0,005 de macro-F1).
- Jeu de validation : 1 000 avis tirés parmi les 5 000 du train (`review_id` triés, `np.random.default_rng(42)
  .permutation`, les 1 000 premiers), tirage fait avant l'exclusion des invalides pour que le découpage ne dépende pas
  des étiquettes. Résultat : 3 983 avis d'entraînement, 996 de validation (positifs en validation : reception 104,
  gout_texture 571, fraicheur_securite 159, non_conforme 84, prix 150, autre 89). Tirage au hasard et non par produit :
  20,7 % des avis de test portent sur un produit présent dans le train, la validation doit reproduire cette situation.
- Rôles : l'entraînement apprend les poids ; la validation sert à choisir les hyperparamètres, la règle de sortie et le
  meilleur modèle entraîné (TF-IDF ou DistilBERT) ; le test mesure une seule fois chaque modèle figé, sans retour aux
  réglages ensuite.
- Les étiquettes de validation sont celles de mistral-small v4 : un score de validation mesure l'accord avec ce
  professeur, pas avec la référence de Martin.
- Ré-entraînement après le choix (décision B du 14 oct. 2026, prise avant toute mesure d'un modèle entraîné sur le
  test ; elle remplace la décision du même jour « pas de ré-entraînement ») : la configuration choisie sur la validation
  est ré-entraînée sur entraînement + validation (4 979 avis), et c'est ce modèle qui est mesuré une fois sur le test.
  Même règle pour DistilBERT : ré-entraînement sur les 4 979 avis avec le nombre d'époques choisi sur la validation.
  Le choix entre TF-IDF et DistilBERT se fait sur la validation, les deux entraînés sur les 3 983 mêmes avis.
  Raisons : le test mesure le modèle qu'on livrerait ; la courbe d'apprentissage monte encore (macro-F1 de validation
  0,542 avec 1 000 avis, 0,606 avec 2 000, 0,636 avec 3 000, 0,660 avec 3 983 ; gain attendu de +0,015 à +0,02,
  extrapolation peu sûre) ; l'argument contraire est tombé, car avec `class_weight="balanced"` régler les seuils
  n'apporte rien (0,651 contre 0,650) et le seuil reste fixé à 0,5 ; c'est la pratique par défaut de `GridSearchCV`
  (`refit=True`). Limite : le modèle mesuré sur le test n'a jamais été mesuré sur la validation (risque faible pour une
  régression logistique aux réglages fixés). Dans le code : `REFIT = True` dans `src/tfidf_lr.py`.
- TF-IDF + régression logistique, grille fixée avant toute mesure (`src/tfidf_lr.py`) : C ∈ {0,1 ; 0,3 ; 1 ; 3 ; 10}
  × n-grammes {(1, 1), (1, 2)} × `min_df` {1, 3} ; fixes : `class_weight="balanced"`, `sublinear_tf=True`, mots vides
  gardés, seuil 0,5, 5 classifieurs (un par motif de plainte) et règle de sortie « autre si et seulement si aucun motif
  de plainte n'est retenu ». Critère de choix : macro-F1 de validation sur les 6 motifs, après la règle.
- Résultat de la grille (14 oct. 2026, validation de 996 avis, étiquettes du professeur) : configuration choisie C = 3,
  mots + paires, `min_df` = 3 (13 831 termes) ; macro-F1 0,651 (IC 95 % par bootstrap [0,620 ; 0,679]), accord exact
  0,623, entraînement 0,950 ; 0 sortie invalide sur les 20 configurations. F1 par motif : reception 0,689,
  gout_texture 0,845, fraicheur_securite 0,730, non_conforme 0,525, prix 0,687, autre 0,429 (précision 0,36, rappel
  0,54 : autre sert de fourre-tout quand les 5 classifieurs hésitent). Plateau : 5 configurations entre 0,643 et 0,651,
  écarts probablement dus au bruit ; 0,651 est le maximum de 20 mesures, donc légèrement optimiste. Entraînement 0,5 s,
  prédiction par lot 0,04 ms par avis, modèle 0,9 Mo. Prévision écrite avant la mesure sur le test :
  macro-F1 entre 0,55 et 0,68.
- DistilBERT (`src/distilbert.py`, entraînement sur Colab, GPU Tesla T4). Réglages fixés avant la grille :
  `distilbert-base-uncased` (66,96 M de paramètres, dont 0,59 M pour la tête neuve), même texte que TF-IDF (titre +
  texte), 512 tokens au plus (mesuré sur l'entraînement : médiane 47, 1,9 % au-delà de 256, 0,2 % au-delà de 512),
  remplissage dynamique, lots de 16, 5 sorties (une par motif de plainte), entropie croisée binaire pondérée par
  n_négatifs / n_positifs (même rapport que `class_weight="balanced"`), AdamW (décroissance des poids 0,01, sauf biais
  et LayerNorm), montée linéaire du taux sur 10 % des pas puis descente jusqu'à 0, gradients bornés à 1,
  demi-précision, même règle de sortie que TF-IDF (seuil 0,5, autre si aucun motif). Mesure sur la validation à la fin
  de chaque époque.
- Grille : (taux 2e-5, 4 époques) et (5e-5, 4 époques) x graines 42, 43, 44 (14 oct. 2026), puis (5e-5, 6 époques) x
  les 3 mêmes graines, ajouté le 15 oct. 2026 parce que le meilleur réglage du 14 était au bord de la grille (taux le
  plus haut, dernière époque) ; ajout décidé sur la validation, avant tout contact avec le test. Un entraînement est
  identifié par (taux, calendrier, graine) : l'époque 4 d'un calendrier de 6 (taux encore à 37 %) n'est pas l'époque 4
  d'un calendrier de 4. Règle de choix, fixée avant les résultats : meilleure moyenne des 3 graines sur les triplets
  (taux, calendrier, époque d'arrêt) ; à égalité, le plus petit taux, puis le calendrier le plus court, puis le moins
  d'époques.
- Résultats de la grille (macro-F1 de validation, moyenne [min ; max] des graines) : 2e-5 / 4 époques 0,630 à
  l'époque 4 (sous-apprentissage) ; 5e-5 / 4 époques 0,707 [0,702 ; 0,714] ; 5e-5 / 6 époques 0,696 à l'époque 4,
  0,7200 [0,704 ; 0,732] à l'époque 5, 0,7203 [0,711 ; 0,727] à l'époque 6. Choix : taux 5e-5, calendrier de
  6 époques, arrêt à l'époque 6, macro-F1 0,720 (écart-type 0,0083). Gain de l'extension +0,013 (~2 erreurs types) ;
  époques 5 et 6 à égalité de fait (plateau, pas de nouvelle extension). Maximum de 14 triplets : légèrement flatteur.
  ~26 s par époque sur T4 ; 0 sortie invalide.
- Comparaison TF-IDF / DistilBERT sur la validation (bootstrap apparié, mêmes 3 983 avis d'entraînement, mêmes 996
  avis de validation, étiquettes du professeur), écart DistilBERT − TF-IDF en macro-F1 : graine 42 +0,060
  [+0,029 ; +0,092], graine 43 +0,072 [+0,042 ; +0,103], graine 44 +0,076 [+0,047 ; +0,107] ; accord exact +0,051 à
  +0,075, intervalles excluant 0. **Décision (15 oct. 2026) : DistilBERT est le meilleur modèle entraîné** ; la
  comparaison principale portera sur DistilBERT contre mistral-small v4. F1 par motif (graine 42 contre TF-IDF) :
  reception 0,743 / 0,689, gout_texture 0,869 / 0,845, fraicheur_securite 0,775 / 0,730, non_conforme 0,563 / 0,525,
  prix 0,791 / 0,687, autre 0,526 / 0,429. fraicheur_securite : 23 plaintes manquées sur 159 au lieu de 44 (rappel
  0,86 contre 0,72), 56 en trop au lieu de 41 ; plus aucune manquée classée autre (10 pour TF-IDF), 14 encore classées
  gout_texture.
- Mesure finale (`final`, une fois) : le réglage choisi est ré-entraîné sur les 4 979 avis (décision B), graine 42
  fixée d'avance (la moins bonne des trois en validation, 0,711 : pas de choix de graine), même calendrier et même
  époque d'arrêt que dans la grille, puis prédit les 300 avis de test. Prévision écrite avant la mesure :
  macro-F1 sur le test entre 0,66 et 0,74. Dans le code : `GRILLE_VALIDEE = True`.

## Résultats sur le test (300 avis, référence finale de Martin)
| Modèle | macro-F1 [IC 95 %] | micro-F1 [IC 95 %] | Accord exact [IC 95 %] | Coût / 100 000 avis | Latence médiane / p95 | Date |
| --- | --- | --- | --- | --- | --- | --- |
| `mistral-small-2603` + prompt v4 | 0,770 [0,716 ; 0,816] | 0,843 [0,809 ; 0,876] | 0,763 [0,717 ; 0,810] | ~4,6 € (~5,3 $), facture réelle avec cache automatique (14,6 $ au prix catalogue sans cache) | 0,76 s / 1,19 s | 14 oct. 2026 |
| TF-IDF + régression logistique (C = 3, mots + paires, `min_df` = 3, 4 979 avis étiquetés par v4) | 0,634 [0,578 ; 0,683] | 0,739 [0,700 ; 0,777] | 0,607 [0,550 ; 0,660] | ~0 $ (~5 s de CPU ; étiquettes : 0,73 $ une fois) | 0,95 ms un avis à la fois (moyenne) ; 0,045 ms par lot | 14 oct. 2026 |
| DistilBERT (`distilbert-base-uncased`, 5e-5, 6 époques, graine 42, 4 979 avis étiquetés par v4) | 0,712 [0,657 ; 0,758] | 0,788 [0,749 ; 0,823] | 0,683 [0,633 ; 0,733] | ~0 $ par avis (entraînement : 3,2 min de GPU T4, Colab gratuit ; étiquettes : 0,73 $ une fois) | GPU T4 : 7,6 ms un avis à la fois (moyenne de 100) ; 0,79 ms par lot ; CPU non mesuré | 15 oct. 2026 |
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
- TF-IDF + régression logistique (mesure unique du 14 oct. 2026, `data/preds_test_tfidf.jsonl`, 0 invalide), F1 par
  motif [IC 95 %] (support) : reception 0,786 [0,676 ; 0,872] (42), gout_texture 0,874 [0,834 ; 0,909] (171),
  fraicheur_securite 0,652 [0,530 ; 0,759] (46), non_conforme 0,440 [0,261 ; 0,604] (24), prix 0,675
  [0,550 ; 0,778] (39), autre 0,375 [0,219 ; 0,514] (32 ; 32 prédits, 12 justes). Kappa Martin / modèle : reception
  0,75, gout_texture 0,70, fraicheur_securite 0,59, non_conforme 0,39, prix 0,62, autre 0,30. Prévision écrite avant
  la mesure (0,55 à 0,68) tenue ; validation 0,651 contre le professeur, test 0,634 contre Martin.
- Comparaison appariée TF-IDF − mistral-small v4 (bootstrap apparié, 2 000 tirages) : macro-F1 −0,137
  [−0,200 ; −0,072], accord exact −0,157 [−0,213 ; −0,100]. Les intervalles excluent 0 : le LLM bat TF-IDF. Comparaison
  provisoire : la comparaison principale portera sur le meilleur modèle entraîné, choisi sur la validation après
  DistilBERT. L'élève garde 82 % du macro-F1 du professeur (0,634 / 0,770) et 88 % de son micro-F1.
- Lecture : TF-IDF tient sur gout_texture (kappa 0,70, au niveau de Martin avec lui-même, 0,68) et sur reception ;
  l'écart le plus grand porte sur fraicheur_securite (F1 −0,25 ; rappel 0,63 contre 0,89) : 17 plaintes manquées sur
  46, dont 7 classées gout_texture et 5 autre ; le LLM en retrouve 12. Hypothèse (non vérifiée) : la règle « rance,
  rassis, moisi -> fraicheur, prioritaire sur gout » et les effets sur la santé demandent de lire la phrase, alors que
  « taste » pèse lourd pour gout_texture. Ordre de grandeur métier : sur 100 000 avis (15 % de fraicheur_securite),
  l'écart de rappel représente ~4 000 plaintes de sécurité manquées en plus, à comparer au coût du LLM (~4,6 € pour
  100 000 avis, facture réelle ; 14,6 $ au prix catalogue sans cache).
- DistilBERT (mesure unique du 15 oct. 2026, `data/preds_test_distilbert.jsonl`, 0 invalide), F1 par motif
  [IC 95 %] (support) : reception 0,810 [0,712 ; 0,894] (42), gout_texture 0,877 [0,839 ; 0,911] (171),
  fraicheur_securite 0,736 [0,622 ; 0,833] (46 ; précision 0,78, rappel 0,70 : 32 trouvées, 14 manquées), non_conforme
  0,528 [0,350 ; 0,677] (24), prix 0,771 [0,658 ; 0,861] (39), autre 0,552 [0,383 ; 0,696] (32). Kappa Martin /
  modèle : reception 0,78, gout_texture 0,71, fraicheur_securite 0,69, non_conforme 0,48, prix 0,73, autre 0,50.
  Prévision écrite avant la mesure (0,66 à 0,74) tenue ; validation 0,720 (moyenne des graines ; graine 42 : 0,711)
  contre le professeur, test 0,712 contre Martin. L'élève garde 92 % du macro-F1 du professeur (TF-IDF : 82 %).
- **Comparaison principale** (règle du 13 oct. 2026 : meilleur modèle entraîné, choisi sur la validation, contre
  mistral-small v4 ; bootstrap apparié, 2 000 tirages) : écart DistilBERT − LLM en macro-F1 −0,058 [−0,115 ; +0,003] ;
  accord exact −0,080 [−0,133 ; −0,030]. Sur la métrique principale, l'intervalle contient 0 (de justesse) : selon la
  règle, la supériorité du LLM n'est pas démontrée et la décision se prend sur le coût et la latence. Réserves à
  reporter dans le livrable : l'estimation ponctuelle favorise le LLM de 0,058 et l'intervalle admet jusqu'à −0,115
  (300 avis ne suffisent pas à trancher un écart de cette taille) ; l'accord exact, métrique secondaire, montre le LLM
  meilleur (intervalle excluant 0) ; sur fraicheur_securite, rappel 0,70 contre 0,89, soit ~3 000 plaintes de sécurité
  manquées en plus pour 100 000 avis (15 % de ce motif), à comparer aux ~4,6 € du LLM. Recommandation finale à rédiger
  dans le livrable, pas tranchée ici.
- Rappel fraicheur_securite, bootstrap apparié (16 oct. 2026, 2 000 tirages, prédictions existantes, aucune nouvelle
  mesure) : DistilBERT − LLM −0,196 [−0,340 ; −0,069] (46 plaintes : 10 retrouvées par le LLM seul, 1 par DistilBERT
  seul) ; TF-IDF − LLM −0,261 [−0,395 ; −0,135]. Motif choisi après coup (écart repéré à la mesure de TF-IDF) : avec la
  correction de Bonferroni sur les 6 motifs (alpha = 0,05/6), −0,196 [−0,383 ; −0,022] et −0,261 [−0,452 ; −0,103],
  toujours négatifs. F1 du motif, DistilBERT − LLM : −0,165 [−0,272 ; −0,073] (Bonferroni [−0,318 ; −0,038]).
  Précision : LLM 0,911, DistilBERT 0,780, −0,131 [−0,248 ; −0,015] (Bonferroni [−0,296 ; +0,030]). Pour 100 000 avis
  (15 % de ce motif) : ~3 000 plaintes de sécurité manquées en plus par DistilBERT [~1 000 ; ~5 100]. Seuil de
  DistilBERT (0,5) non optimisé pour ce motif : le baisser remonterait son rappel au prix de sa précision (non essayé).
- Repère humain (précision du 16 oct. 2026) : 0,70 compare la passe 2 à la passe 1 sur 50 des 300 avis ; les modèles
  sont mesurés contre la référence arbitrée sur les 300. Cible et taille différentes, et l'intervalle [0,58 ; 0,82]
  contient 0,763 : les deux chiffres ne se comparent pas directement. Le livrable ne dit plus « au niveau d'un
  annotateur humain » ; le repère n'y sert qu'à montrer que la tâche est ambiguë. Même réserve pour les lectures
  « au niveau de Martin avec lui-même » ci-dessus (kappas).

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

## RAG (jalon 2, décisions du 22 oct. 2026)
- Périmètre : RAG écrit à la main (`src/rag.py`, numpy), sans framework ; une version LangChain viendra au jalon 3,
  quand le RAG deviendra un outil de l'agent.
- Corpus : `corpus/politique_client.jsonl`, politique client d'une épicerie en ligne fictive (nom provisoire
  `MARQUE = "Larder Lane"`), en anglais comme les avis. Un passage par ligne : `id` (P01...), `titre`, `motifs` (motifs
  v2 sauf `autre`, ou `general`), `texte`. Critères vérifiés par les tests : 40 à 80 passages, au moins 5 par motif,
  100 à 250 mots chacun. Premier jet généré par un LLM le 22 oct. (46 passages, 105 à 135 mots), **validé tel quel par
  Martin le 22 oct.**, avec ses règles chiffrées (30 jours pour réclamer, remboursement sous 5 jours ouvrés sans retour
  du produit, remplacement sous 2 jours ouvrés, garantie goût de 15 $ maximum, 30 jours de durée de vie garantie,
  équipe sécurité sous 24 h, escalade sous 1 jour ouvré, livraison offerte dès 35 $). Le corpus est la référence de
  l'évaluation : toute modification ultérieure se date ici et impose de reconstruire l'index.
  - 22 oct., P08 : phrase ajoutée avec les mots des clients (« This applies even when the outside of the box looks
    fine and the damage is only visible once the box is opened. »), 157 mots. Raison : au test d'échauffement, un avis
    « pot cassé, carton intact » classait P14 (Damaged box but intact product) devant P08 (0,808 contre 0,799).
    Aucune règle chiffrée changée.
- Embeddings : `mistral-embed` (1 024 dimensions, norme 1), vecteurs renormalisés par le code ; on indexe titre + texte.
  Index : `data/rag/index.npy` + `index.json`, recalculé seulement si l'empreinte (SHA-256 des textes et du modèle) change.
- Recherche exacte (`M @ q`). **Décision du 23 oct. 2026 : k = 8, sans filtre par motif par défaut** (option
  `--filtre` pour le réactiver). Mesure sur 26 avis annotés de `rag_dev` : rappel@5 0,769 [0,615 ; 0,923] sans filtre,
  0,808 [0,654 ; 0,962] avec (un seul avis d'écart) ; rappel@8 sans filtre 0,885. Raisons : le filtre dur (motifs
  prédits par v4 + `general`) excluait des passages trouvés en tête sans filtre quand le classifieur se trompait de
  motif, dont un passage de sécurité (P32, avis classé `reception`) ; k = 8 coûte ~70 µ$ de plus par réponse.
  Écarté pour rester simple : réécrire P17 (garantie goût, rang médian ~7,5 : le passage du remède ne ressemble pas
  à la plainte) et un filtre souple (rappel@8 0,923). Mesure finale à refaire sur `rag_eval`.
- Jeux du jalon 2, tirés une fois de la réserve (graine 42, après tri par review_id) : `rag_dev` 50 avis (mise au
  point libre ; 30 annotés pour la recherche), `rag_eval` 150 avis gelés comme le test. Le test de 300 avis n'entre
  jamais dans un prompt RAG. Liste dans `data/rag/jeux.json` ; motifs prédits dans `data/labels_<jeu>_v4.jsonl`.
- Prompt `rag_v2` depuis le 23 oct. 2026 (`rag_v1` gardé pour comparaison). Raisons, lues sur 5 réponses v1 de
  `rag_dev` : citations absentes du texte (3/5), conditions supposées remplies (« since this is your first purchase »),
  promesse hors politique (remboursement à la place de l'avoir de la garantie goût), avis sans plainte traité comme une
  plainte, et « they do not affect safety » pour des éclats de noyaux. Ajouts de la v2 : cas A sécurité
  (`escalade_securite` même sans passage fourni, jamais « safe »), cas B sans plainte (`aucune`), conditions au
  conditionnel, aucune alternative non prévue, citation à la fin de chaque phrase de règle, avec un exemple.
  `escalade_securite` rejoint `escalade` et `aucune` parmi les actions dispensées de citation (`SANS_PROMESSE`).
- Prompt `rag_v3` le 23 oct. 2026, après des réponses v2 pas meilleures que v1 sur les mêmes 5 avis (le cas sécurité
  ignoré malgré « pits » cité dans le prompt ; « since » et le remboursement à la place de l'avoir réapparus) :
  `ReponseClient` commence par `raison` (ce que signale le client) et `cas` (securite, sans_plainte, plainte),
  générés avant la réponse, comme `evidence` avant `motifs` au jalon 1. Garde-fous en code, sans appel, ajoutés à
  `problems` (`signaux_redaction`) : cas_action_incoherent (securite -> escalade_securite, sans_plainte -> aucune),
  condition_supposee, alternative_non_prevue (face à un avoir), affirme_securite, securite_a_verifier (mots de
  sécurité dans l'avis lui-même ; 3,1 % des 4 979 avis du train, dont 70 % classés fraicheur_securite par v4). Ils
  signalent, ne corrigent pas : la réponse reste un brouillon relu par un humain. Le contrôle des citations accepte
  aussi `[P20, P23]`. Rejoués sur les 5 réponses v2, ils signalent toutes les fautes vues à la lecture.
- Prompt `rag_v4` le 23 oct. 2026 : la v3 a réglé le cas sécurité (éclats de noyaux -> escalade_securite) mais a
  classé trois plaintes de goût en `sans_plainte` ; seule la définition du cas B change : « the customer reports
  nothing negative at all », et un goût, une odeur ou une texture qui déplaît, une préférence pour un autre produit
  ou un prix jugé trop élevé relèvent du cas C. Dernière itération à l'œil : la suite se mesure avec le LLM-juge.
- Génération : `mistral-small-2603`, prompt `prompts/rag_v4.txt` (v1, v2, v3 le 23 oct.), temperature=0, random_seed=42, max_tokens=500,
  sortie structurée `ReponseClient` (`reponse`, `citations`, `action`) ; actions : remboursement, remplacement,
  geste_commercial, escalade_securite, escalade, aucune. Deux modes : `rag` (k passages retrouvés) et `plein` (tout le
  corpus, même gabarit, pour profiter du cache).
- Vérification en code (`verifier_citations`) : citation_inconnue, citation_non_declaree, citation_absente_du_texte,
  sans_citation. Une citation s'écrit `[P12]`, un identifiant par crochet.
- Mesure de la recherche : rappel@k (au moins un passage pertinent dans les k premiers) et MRR, sur des passages
  pertinents annotés à la main avant de voir les résultats (`data/rag/pertinents_<jeu>.jsonl`) ; avis sans passage
  pertinent exclus. Seuil visé : rappel@5 >= 0,80 ; atteint à k = 8 sans filtre (0,885, voir ci-dessus).
- Coût estimé (1,267 token par mot mesuré sur l'explo ; réponse ~150 tokens) : ~226 µ$ par réponse en mode `rag`,
  ~214 µ$ en mode `plein` avec le cache (1 225 µ$ sans) ; indexation du corpus ~7 100 tokens, 0,0007 $.

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
- Prix vérifiés le 15 oct. 2026 sur la page publique des prix (docs.mistral.ai, section Pricing), en $ par million de
  tokens : Mistral Small 4 entrée 0,15, entrée en cache 0,015, sortie 0,60 ; Ministral 3 8B entrée 0,15, entrée en
  cache 0,015, sortie 0,15. Les 14,59 $ pour 100 000 avis mesurés le 14 oct. sont cohérents avec ces prix
  (790,6 x 0,15 + 45,6 x 0,60 = 146 $ par million d'avis), mais seulement en l'absence de cache.
- **Facture réelle (console, 15 oct. 2026)** : 6,1 k requêtes (= les 6 100 appels du mois), 0,27 € sur le forfait API
  d'octobre (8,5 €). Prix du compte en € par million de tokens : entrée 0,13, entrée en cache ~0,013 (affiché 0,01),
  sortie 0,51. Détail `mistral-small-2603` (explo v3 et v4, test, train : 5 700 appels) : entrée 596,88 k, **en cache
  3,9 M**, sortie 259,74 k, soit 0,26 € ; `mistral-small-latest` (explo v1 et v2) : entrée 42,8 k, en cache 77,58 k,
  sortie 17,97 k. Totaux identiques aux tokens enregistrés par `label_llm.py` (entrée 4,62 M, sortie 277 708). **Le cache
  du prompt s'est appliqué automatiquement**, sans `prompt_cache_key` : 86,7 % des tokens d'entrée de
  `mistral-small-2603` servis depuis le cache (le prompt système v4, identique à chaque appel).
- **Coût réel pour 100 000 avis avec le prompt v4 : ~4,6 € (~5,3 $)** (0,26 € / 5 700 appels ; même résultat avec les
  prix en $ et la part en cache mesurée), contre 14,6 $ calculés sans cache. La sortie (`evidence`, `motif_libre`,
  `motifs` : ~46 tokens) fait désormais 51 % de la facture. Avec la même part en cache, Ministral 3 8B coûterait
  ~3,3 $ pour 100 000 avis (estimation, sortie 4 fois moins chère). Leviers non essayés : traitement par lots, annoncé
  à −50 % (mistral.ai, page Pricing) ; sortie plus courte (sans `evidence`), au risque de perdre en qualité.
- Cache du prompt (documentation Mistral, page Prompt caching) : blocs de 64 tokens, tokens en cache facturés 10 % du
  prix d'entrée ; la documentation le présente comme activé par `prompt_cache_key`, mais la facture montre qu'il
  s'applique aussi sans ce paramètre quand le début du prompt est identique d'un appel à l'autre.
