"""Slide de synthèse du jalon 1 (une page, 16:9) -> livrable/synthese.png et livrable/synthese.pdf.

Usage :  python livrable/slide.py        (demande matplotlib, voir requirements.txt)
Chiffres recopiés de docs/CONVENTIONS.md (sections Résultats sur le test et Coûts, 15 oct. 2026).
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

# nom, macro-F1, IC bas, IC haut, rappel sécurité, coût pour 100 000 avis, latence, recommandé
MODELES = [
    ("mistral-small + prompt v4", 0.770, 0.716, 0.816, 0.89, "~4,6 €", "0,76 s (API)", True),
    ("DistilBERT", 0.712, 0.657, 0.758, 0.70, "~0 € + hébergement", "7,6 ms (GPU)", False),
    ("TF-IDF + régression logistique", 0.634, 0.578, 0.683, 0.63, "~0 € + hébergement", "0,95 ms (CPU)", False),
]
ROWS = [2, 1, 0]                       # le modèle recommandé en haut


def fr(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=11, length=0)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(16, 9), dpi=125, facecolor=SURFACE)

    fig.text(0.04, 0.935, "Trier les avis clients négatifs par motif : quel modèle ?", fontsize=24,
             fontweight="bold", color=INK)
    fig.text(0.04, 0.892, "Avis Amazon Grocery notés 1 à 3, 6 motifs (un avis peut en porter plusieurs).",
             fontsize=12.5, color=INK_2)
    fig.text(0.04, 0.862, "Test : 300 avis tirés au hasard, annotés à la main avant toute prédiction ; intervalles de "
             "confiance à 95 % par bootstrap.", fontsize=12.5, color=INK_2)

    def titre(x, texte, legende):
        fig.text(x, 0.795, texte, fontsize=13.5, fontweight="bold", color=INK)
        fig.text(x, 0.768, legende, fontsize=10.5, color=MUTED)

    # 1. Qualité : macro-F1 et intervalle
    ax = fig.add_axes([0.27, 0.42, 0.28, 0.33])
    style(ax)
    ax.set_xlim(0.55, 0.98)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9])
    ax.set_xticklabels(["0,6", "0,7", "0,8", "0,9"])
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for (nom, f1, lo, hi, *_rest, reco), y in zip(MODELES, ROWS):
        c = ACCENT if reco else OTHER
        ax.plot([lo, hi], [y, y], color=c, linewidth=2, solid_capstyle="round")
        ax.plot(f1, y, "o", markersize=10, color=c, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(hi + 0.008, y, f"{fr(f1)}  [{fr(lo)} ; {fr(hi)}]", va="center", fontsize=11.5,
                color=INK if reco else INK_2, fontweight="bold" if reco else "normal")
        fig.text(0.04, ax.transData.transform((0, y))[1] / fig.bbox.height, nom, va="center", fontsize=13,
                 color=INK, fontweight="bold" if reco else "normal")
    titre(0.27, "Qualité : macro-F1 sur les 6 motifs", "point : estimation ; trait : intervalle à 95 %")

    # 2. Sécurité alimentaire : rappel
    ax2 = fig.add_axes([0.60, 0.42, 0.15, 0.33])
    style(ax2)
    ax2.set_xlim(0, 1.18)
    ax2.set_xticks([0, 0.5, 1])
    ax2.set_xticklabels(["0", "0,5", "1"])
    for (*_a, rappel, _c, _l, reco), y in zip(MODELES, ROWS):
        ax2.barh(y, rappel, height=0.42, color=ACCENT if reco else OTHER)
        ax2.text(rappel + 0.03, y, fr(rappel, 2), va="center", fontsize=11.5,
                 color=INK if reco else INK_2, fontweight="bold" if reco else "normal")
    titre(0.60, "Rappel sécurité", "part des plaintes de sécurité retrouvées")

    # 3. Coût et latence
    ax3 = fig.add_axes([0.80, 0.42, 0.18, 0.33])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-0.6, 2.6)
    ax3.axis("off")
    for (*_a, cout, latence, reco), y in zip(MODELES, ROWS):
        w = "bold" if reco else "normal"
        ax3.text(0, y + 0.13, cout, va="center", fontsize=12, color=INK, fontweight=w)
        ax3.text(0, y - 0.2, latence, va="center", fontsize=11, color=INK_2)
    titre(0.80, "Coût et latence", "pour 100 000 avis ; par avis")

    # 4. Recommandation
    fig.patches.append(FancyBboxPatch((0.04, 0.07), 0.94, 0.29, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, facecolor=PANEL, edgecolor="none", zorder=0))
    fig.text(0.06, 0.315, "Recommandation : mistral-small (version figée) avec le prompt v4, pour un tri différé",
             fontsize=15.5, fontweight="bold", color=INK)
    lignes = [  # (puce, texte, couleur) ; sans puce = suite de la ligne précédente
        (True, "Meilleure qualité, au niveau d'un annotateur humain comparé à lui-même (accord exact 0,763 contre "
               "0,70).", INK),
        (True, "Écart avec DistilBERT de −0,058 [−0,115 ; +0,003] en macro-F1, non démontré sur 300 avis ; mais "
               "~3 000 plaintes de sécurité", INK),
        (False, "de plus retrouvées pour 100 000 avis (rappel 0,89 contre 0,70).", INK),
        (True, "Coût réel ~4,6 € pour 100 000 avis : le cache du prompt divise le prix catalogue (14,6 $) par 2,7.", INK),
        (True, "Conditions : version figée ; 100 avis annotés chaque mois pour suivre la qualité ; DistilBERT en "
               "solution de repli.", INK_2),
        (True, "Basculer vers DistilBERT si : les données ne doivent pas sortir, temps réel, ou ~1 million d'avis "
               "par mois.", INK_2),
    ]
    y = 0.265
    for puce, txt, couleur in lignes:
        fig.text(0.065 if puce else 0.078, y, ("•  " if puce else "") + txt, fontsize=12.5, color=couleur)
        y -= 0.033

    fig.text(0.04, 0.03, "Modèles internes entraînés sur 4 979 avis étiquetés par le LLM (distillation). DistilBERT : "
             "latence sur GPU T4. Détail des décisions et des chiffres : docs/CONVENTIONS.md.",
             fontsize=10, color=MUTED)

    fig.savefig(OUT / "synthese.png", facecolor=SURFACE)
    fig.savefig(OUT / "synthese.pdf", facecolor=SURFACE)
    print(f"{OUT / 'synthese.png'} et {OUT / 'synthese.pdf'} écrits")


if __name__ == "__main__":
    main()
