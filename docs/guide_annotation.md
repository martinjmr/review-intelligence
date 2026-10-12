# Guide d'annotation des motifs d'insatisfaction : v1

**Statut** : brouillon. **Gelé le** : À TOI (date, avant le premier avis de test).
**Écrit à partir de** : liste v2 (CONVENTIONS.md, 12 oct. 2026) et des 200 avis d'explo. Aucun avis de test.
Pour relire un avis d'explo cité : `python -m src.annotate --voir 7 61 96`.

## 1. La tâche

- Lire le **titre et le texte** de l'avis (pas la note : les modèles ne la voient pas).
- Cocher **tous les motifs de plainte** présents. Une plainte = un problème que le client reproche à ce qu'il a reçu.
- Une mention positive ne compte pas (« le goût est bon, mais... » ne donne pas `gout_texture`).
- Aucun motif ne s'applique, ou aucune plainte → `autre`, seul.

## 2. Les motifs (liste v2)

| Touche | Motif | Définition | Frontière |
| --- | --- | --- | --- |
| r | reception | problème lié au transport ou au colis : retard, colis perdu ou incomplet, produit cassé, écrasé, fondu ou qui fuit à l'arrivée, emballage abîmé, ouvert ou mal conçu | produit arrivé intact mais mauvais -> g ou f |
| f | fraicheur_securite | périmé, rassis, rance, moisi, insectes, corps étranger, effet sur la santé, allergène non signalé | prioritaire sur g |
| n | non_conforme | ne correspond pas à la description : mauvais produit, format ou quantité annoncée, ingrédient non annoncé, étiquetage trompeur | reçu ce qui était annoncé mais jugé trop peu pour le prix -> p |
| g | gout_texture | goût, texture ou odeur d'un produit arrivé intact et non périmé | rance, rassis, moisi -> f |
| p | prix | trop cher, mauvais rapport quantité/prix, hausse de prix | |
| a | autre | aucun des motifs ci-dessus, y compris aucune plainte ; jamais combiné | |

## 3. L'arbre de décision : une plainte, un motif

Chaque plainte reçoit **un seul motif, celui de sa cause** : on pose les questions dans l'ordre et on s'arrête
à la première réponse « oui ». Un avis qui contient plusieurs plaintes reçoit plusieurs motifs.
Une **conséquence** compte comme une plainte à part si elle concerne une autre équipe que sa cause
(n° 160 : rassis donc dur -> `f` seul ; n° 50 : sac percé, peut-être contaminé -> `r` + `f`).

Validé le 12 oct. 2026 :

1. L'avis contient-il au moins une plainte ? Non -> `a`. Fin.
2. Pour chaque plainte :
   1. Vient-elle du transport ou du colis ? -> `r`
   2. Fraîcheur ou sécurité ? -> `f`
   3. Écart avec l'annonce ? -> `n`
   4. Goût, texture, odeur ? -> `g`
   5. Prix ? -> `p`
   6. Aucune case : si c'est la seule plainte -> `a`.

## 4. Règles transversales

- Décider sur ce qui est **écrit**, pas sur ce qui s'est « vraiment » passé.
- Le **titre compte autant que le texte** : un motif nommé seulement dans le titre compte (n° 65 : titre « taste », texte « mon chien refuse d'en manger » -> `g`).
- Une plainte sur une variante d'un assortiment (une saveur sur trois) compte comme une plainte.
- À TOI : d'autres règles qui valent pour tous les motifs, si l'annotation en fait apparaître.

## 5. Jurisprudence : les cas limites tranchés

Une ligne par groupe : la règle en une phrase, décidable sur le texte seul, et le n° d'explo qui l'illustre.

| # | Groupe | Avis d'explo | Règle |
| --- | --- | --- | --- |
| 1 | Contenant cabossé, cassé ou ouvert à l'arrivée | 75, 101, 178 | `r`, même si le client ne sait pas quand le choc a eu lieu (CONVENTIONS : état à l'arrivée). |
| 2 | Allergène absent de la fiche produit | 92, 129 | `f` (CONVENTIONS : allergène non signalé). |
| 3 | « Goût de rassis » | 134 | `f` (prompt v2 : rassis prioritaire sur le goût). |
| 4 | Effet sur la santé ressenti après consommation | 7, 96 | `f` (CONVENTIONS : effet sur la santé). |
| 4b | Effet dû à une propriété normale du produit chez une personne sensible | 61 | La règle 4 s'applique toujours : dès qu'un effet sur la santé est ressenti -> `f`, même si le client l'attribue à sa propre sensibilité. n° 61 : « trop acide » + donné à sa mère -> `g` ; reflux et estomac noué -> `f` ; donc `g` + `f`. |
| 5 | Inquiétude pour la santé sans effet vécu | 22, 50, 86 | Inquiétude sur l'état de l'exemplaire reçu (odeur suspecte, emballage percé, contamination possible) -> `f` (n° 22 : odeur chimique ou d'insecticide sur les sacs ; n° 50 : « pas sûr que ce soit sans danger » après un sac percé, donc `r` + `f` : une conséquence qui concerne une autre équipe, section 3). Inquiétude sur un ingrédient en général, sans rien d'anormal dans l'exemplaire -> aucun motif ; si c'est la seule plainte, `a` (n° 86 : propylène glycol « cancérigène » ; n° 24 : caséine ; n° 146 : glutamate). |
| 6 | Ingrédient annoncé sur l'étiquette mais non désiré | 24, 146, 152 | Un ingrédient qui figure sur l'étiquette et que le client n'aime pas ou ne veut pas (lait, glutamate, huile de colza, sucre) ne donne aucun motif ; si c'est la seule plainte -> `a` (n° 24, 146 ; leur inquiétude pour la santé est générale, groupe 5). Par défaut, un ingrédient est supposé annoncé : `n` seulement si le client écrit qu'il ne l'était pas, `f` si c'est un allergène non signalé (groupe 2). Exemple complet, n° 152 : l'huile de colza n'ajoute rien ; `f` (date trop proche, pas frais, crampes) + `g` (« enveloppes de pop-corn entre les dents », règle 12). |
| 7 | Unités manquantes dans le colis ; contenu d'une unité inférieur à l'annonce | 38, 180 ; 26 | Unités manquantes dans le colis -> `r` (colis incomplet ; n° 38 : 5 bouteilles sur 6 ; n° 180 : la moitié de la commande). Contenu d'une unité inférieur à ce qu'annonce son étiquette -> `n` (n° 26 : « annonce 70 oz, en contient environ 17 »). |
| 8 | Produit qui a rétréci par rapport aux achats précédents | 163 | La taille est supposée annoncée par défaut : un produit plus petit qu'aux achats précédents -> `p` (rapport quantité/prix) ; `n` seulement si le client écrit que la taille ou la quantité reçue diffère de l'annonce (comme au n° 26, groupe 7). n° 163 : « elles ont rétréci » + « une autre marque offre un meilleur rapport » -> `p`. |
| 9 | Date de péremption proche mais non dépassée | 11, 66, 157 | Date limite jugée trop proche, ou durée de conservation trop courte pour la quantité reçue, même si le produit n'est pas encore périmé -> `f` (n° 11 : « moins de 2 mois pour manger 4 boîtes » ; n° 66 : « 4 mois avant péremption » ; n° 157 : périmé moins d'un mois après l'arrivée). Exemple complet, n° 66 : `f` (date) + `r` (trous du couvercle trop grands : emballage mal conçu, règle 11). |
| 10 | Plainte vague, sans goût ni défaut nommé | 58, 169 ; 19, 149 | Une plainte qui ne nomme, ni dans le titre ni dans le texte, aucun goût, texture, odeur ou défaut précis -> `a` (n° 58, 169). Exception : refuser de manger ou de boire le produit (une personne ou un animal), y compris l'abandonner après l'avoir goûté (jeté, donné, pas fini) -> `g` (n° 19, 149). |
| 11 | Praticité, fonctionnement, emballage difficile à utiliser | 147, 181, 183 | Emballage difficile à ouvrir, à utiliser ou mal conçu -> `r` (n° 147 : bouteilles presque impossibles à ouvrir ; n° 181 : portes du calendrier qui se détachent). Produit qui ne fonctionne pas ou peu pratique à l'usage -> `a` (n° 183 : capsule réutilisable qui bouche la machine). Exemple complet, n° 147 : `r` (cabossé, dur à ouvrir) + `g` (goût d'eau du robinet) + `p` (« not worth the premium price »). |
| 12 | Plainte mineure à côté de la plainte principale | 64, 87 | Toute plainte nommée compte, même dite en passant ou présentée comme secondaire (n° 64 : « arrivé un peu abîmé malgré le papier bulle » + peu de goût -> `r` + `g` ; n° 87 : livraison en 8 jours « un peu longue » + gazéification trop forte -> `r` + `g`). |
| 13 | « Waste of money » sans rien dire du prix | 44, 112 | « Waste of money » ou équivalent, sans rien dire du prix ni du rapport qualité/prix, n'ajoute pas `p` : l'avis reçoit les motifs de ses autres plaintes (n° 44 : « did not taste good » -> `g`), sinon `a` (n° 112 : « didn't know how to put it up »). |

## 6. Journal des versions

| Date | Version | Changement | Né d'un avis de test ? | Avis déjà annotés à revoir |
| --- | --- | --- | --- | --- |
| À TOI | v1 | Gel du guide | non | aucun |

Règle : une règle née d'un avis de test entre dans ce guide (pour annoter), jamais dans un prompt avant la mesure.
