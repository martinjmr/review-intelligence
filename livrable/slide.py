"""Slide de synthèse du jalon 1 (une page, 16:9), en anglais et en français.

Usage :  python livrable/slide.py        (demande matplotlib, voir requirements.txt)
Écrit livrable/synthese.png et .pdf (anglais, pour README.md) et livrable/synthese_fr.png et .pdf (français, pour
README.fr.md). Chiffres recopiés de docs/CONVENTIONS.md (sections Résultats sur le test et Coûts, 15 oct. 2026 ;
écart de rappel sécurité ajouté le 16 oct. 2026).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).parent
SURFACE, PANEL = "#fcfcfb", "#f3f2ee"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
ACCENT, OTHER = "#2a78d6", "#8a8984"   # modèle recommandé ; autres modèles (contraste 4,3:1 et 3,4:1 sur le fond)

# macro-F1, IC bas, IC haut, rappel sécurité, recommandé ; même ordre que "modeles" dans TEXTES
CHIFFRES = [
    (0.770, 0.716, 0.816, 0.89, True),
    (0.712, 0.657, 0.758, 0.70, False),
    (0.634, 0.578, 0.683, 0.63, False),
]
ROWS = [2, 1, 0]                       # le modèle recommandé en haut

# Textes par langue. "lignes" : (puce, texte, couleur) ; sans puce = suite de la ligne précédente.
TEXTES = {
    "fr": {
        "fichier": "synthese_fr",
        "modeles": [("mistral-small + prompt v4", "~4,6 €", "0,76 s (API)"),
                    ("DistilBERT", "~0 € + hébergement", "7,6 ms (GPU)"),
                    ("TF-IDF + régression logistique", "~0 € + hébergement", "0,95 ms (CPU)")],
        "titre": "Trier les avis clients négatifs par motif : quel modèle ?",
        "sous_titres": ["Avis Amazon Grocery notés 1 à 3, 6 motifs (un avis peut en porter plusieurs).",
                        "Test : 300 avis tirés au hasard, annotés à la main avant toute prédiction ; intervalles de "
                        "confiance à 95 % par bootstrap."],
        "qualite": ("Qualité : macro-F1 sur les 6 motifs", "point : estimation ; trait : intervalle à 95 %"),
        "securite": ("Rappel sécurité", "part des 46 plaintes retrouvées"),
        "cout": ("Coût et latence", "pour 100 000 avis ; par avis"),
        "reco": "Recommandation : mistral-small (version figée) avec le prompt v4, pour un tri différé",
        "lignes": [
            (True, "Meilleur des trois sur chaque métrique de qualité ; contre DistilBERT, écart démontré en accord exact "
                   "(0,763 contre 0,683).", INK),
            (True, "Macro-F1 : écart avec DistilBERT de −0,058 [−0,115 ; +0,003], non démontré sur 300 avis.", INK),
            (True, "Sécurité : rappel 0,89 contre 0,70 (écart +0,20 [+0,07 ; +0,34]), ~3 000 plaintes de plus "
                   "retrouvées pour 100 000 avis.", INK),
            (True, "Coût réel ~4,6 € pour 100 000 avis : le cache du prompt divise le prix catalogue (14,6 $) par "
                   "2,7.", INK),
            (True, "Conditions : version figée ; 100 avis annotés chaque mois pour suivre la qualité ; DistilBERT en "
                   "solution de repli.", INK_2),
            (True, "Basculer vers DistilBERT si : les données ne doivent pas sortir, temps réel, ou ~1 million d'avis "
                   "par mois.", INK_2),
        ],
        "notes": ["Modèles internes entraînés sur 4 979 avis étiquetés par le LLM (distillation). DistilBERT : latence "
                  "sur GPU T4. Détail des décisions et des chiffres : docs/CONVENTIONS.md.",
                  "Motif sécurité repéré après la mesure de TF-IDF ; avec la correction de Bonferroni sur les 6 motifs, "
                  "l'écart de rappel reste positif : [+0,02 ; +0,38]."],
    },
    "en": {
        "fichier": "synthese",
        "modeles": [("mistral-small + prompt v4", "~€4.6", "0.76 s (API)"),
                    ("DistilBERT", "~€0 + hosting", "7.6 ms (GPU)"),
                    ("TF-IDF + logistic regression", "~€0 + hosting", "0.95 ms (CPU)")],
        "titre": "Sorting negative customer reviews by complaint reason: which model?",
        "sous_titres": ["Amazon Grocery reviews rated 1 to 3 stars, 6 reasons (a review can carry several).",
                        "Test: 300 randomly drawn reviews, hand-annotated before any prediction; 95% bootstrap "
                        "confidence intervals."],
        "qualite": ("Quality: macro-F1 over the 6 reasons", "dot: estimate; line: 95% interval"),
        "securite": ("Food-safety recall", "share of the 46 safety complaints found"),
        "cout": ("Cost and latency", "per 100,000 reviews; per review"),
        "reco": "Recommendation: mistral-small (pinned version) with prompt v4, for batch sorting",
        "lignes": [
            (True, "Best of the three on every quality metric; against DistilBERT, the gap is demonstrated on exact match "
                   "(0.763 vs 0.683).", INK),
            (True, "Macro-F1 gap with DistilBERT: −0.058 [−0.115, +0.003], not demonstrated on 300 reviews.", INK),
            (True, "Food safety: recall 0.89 vs 0.70 (gap +0.20 [+0.07, +0.34]), ~3,000 more complaints caught per "
                   "100,000 reviews.", INK),
            (True, "Actual cost ~€4.6 per 100,000 reviews: prompt caching cuts the list price ($14.6) by 2.7×.", INK),
            (True, "Conditions: pinned model version; 100 reviews annotated each month to track quality; DistilBERT "
                   "as a fallback.", INK_2),
            (True, "Switch to DistilBERT if: data must stay in-house, real-time is required, or ~1 million reviews "
                   "per month.", INK_2),
        ],
        "notes": ["In-house models trained on 4,979 reviews labelled by the LLM (distillation). DistilBERT latency on "
                  "a T4 GPU. All decisions and figures: docs/CONVENTIONS.md (French).",
                  "Safety label singled out after the TF-IDF test run; with a Bonferroni correction over the 6 "
                  "labels, the recall gap stays positive: [+0.02, +0.38]."],
    },
}


def num(x, lang, d=3):
    s = f"{x:.{d}f}"
    return s.replace(".", ",") if lang == "fr" else s


def intervalle(lo, hi, lang):
    return f"[{num(lo, lang)} ; {num(hi, lang)}]" if lang == "fr" else f"[{num(lo, lang)}, {num(hi, lang)}]"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=11, length=0)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])


def slide(lang):
    t = TEXTES[lang]
    modeles = [(*txt, *chiffres) for txt, chiffres in zip(t["modeles"], CHIFFRES)]
    fig = plt.figure(figsize=(16, 9), dpi=125, facecolor=SURFACE)

    fig.text(0.04, 0.935, t["titre"], fontsize=24, fontweight="bold", color=INK)
    fig.text(0.04, 0.892, t["sous_titres"][0], fontsize=12.5, color=INK_2)
    fig.text(0.04, 0.862, t["sous_titres"][1], fontsize=12.5, color=INK_2)

    def titre(x, texte, legende):
        fig.text(x, 0.795, texte, fontsize=13.5, fontweight="bold", color=INK)
        fig.text(x, 0.768, legende, fontsize=10.5, color=MUTED)

    # 1. Qualité : macro-F1 et intervalle
    ax = fig.add_axes([0.27, 0.42, 0.28, 0.33])
    style(ax)
    ax.set_xlim(0.55, 0.98)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9])
    ax.set_xticklabels([num(x, lang, 1) for x in (0.6, 0.7, 0.8, 0.9)])
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for (nom, _cout, _lat, f1, lo, hi, _rappel, reco), y in zip(modeles, ROWS):
        c = ACCENT if reco else OTHER
        ax.plot([lo, hi], [y, y], color=c, linewidth=2, solid_capstyle="round")
        ax.plot(f1, y, "o", markersize=10, color=c, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(hi + 0.008, y, f"{num(f1, lang)}  {intervalle(lo, hi, lang)}", va="center", fontsize=11.5,
                color=INK if reco else INK_2, fontweight="bold" if reco else "normal")
        fig.text(0.04, ax.transData.transform((0, y))[1] / fig.bbox.height, nom, va="center", fontsize=13,
                 color=INK, fontweight="bold" if reco else "normal")
    titre(0.27, *t["qualite"])

    # 2. Sécurité alimentaire : rappel
    ax2 = fig.add_axes([0.60, 0.42, 0.15, 0.33])
    style(ax2)
    ax2.set_xlim(0, 1.18)
    ax2.set_xticks([0, 0.5, 1])
    ax2.set_xticklabels(["0", num(0.5, lang, 1), "1"])
    for (*_a, rappel, reco), y in zip(modeles, ROWS):
        ax2.barh(y, rappel, height=0.42, color=ACCENT if reco else OTHER)
        ax2.text(rappel + 0.03, y, num(rappel, lang, 2), va="center", fontsize=11.5,
                 color=INK if reco else INK_2, fontweight="bold" if reco else "normal")
    titre(0.60, *t["securite"])

    # 3. Coût et latence
    ax3 = fig.add_axes([0.80, 0.42, 0.18, 0.33])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-0.6, 2.6)
    ax3.axis("off")
    for (_nom, cout, latence, *_c, reco), y in zip(modeles, ROWS):
        w = "bold" if reco else "normal"
        ax3.text(0, y + 0.13, cout, va="center", fontsize=12, color=INK, fontweight=w)
        ax3.text(0, y - 0.2, latence, va="center", fontsize=11, color=INK_2)
    titre(0.80, *t["cout"])

    # 4. Recommandation
    fig.patches.append(FancyBboxPatch((0.04, 0.085), 0.94, 0.275, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, facecolor=PANEL, edgecolor="none", zorder=0))
    fig.text(0.06, 0.315, t["reco"], fontsize=15.5, fontweight="bold", color=INK)
    y = 0.265
    for puce, txt, couleur in t["lignes"]:
        fig.text(0.065 if puce else 0.078, y, ("•  " if puce else "") + txt, fontsize=12.5, color=couleur)
        y -= 0.033

    for y, note in zip((0.048, 0.022), t["notes"]):
        fig.text(0.04, y, note, fontsize=10, color=MUTED)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{t['fichier']}.{ext}", facecolor=SURFACE)
    plt.close(fig)
    print(f"{OUT / t['fichier']}.png et .pdf écrits")


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    for lang in TEXTES:
        slide(lang)


if __name__ == "__main__":
    main()
