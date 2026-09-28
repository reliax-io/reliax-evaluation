"""Figure style shared by the make_*_figure scripts: the Reliax design system,
revision 2 (September 2026).

Colour carries meaning: the series argued for is `SUBJECT` (the signature blue),
the comparison is `COMPARISON` (ink-muted), thresholds and ALARM lines are
`THRESHOLD` (alarm red, drawn dashed), a WATCH line is `REVIEW`. Charts sit on
a white surface. Labels are set in Inter when the font files are present in
eval/fonts/ (SIL Open Font License), otherwise in the default sans."""
import pathlib

import matplotlib
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

PAPER = "#f3f5f8"
SURFACE = "#ffffff"
LINE = "#d6dce2"
LINE_STRONG = "#7d8894"
INK = "#303a48"
INK_MUTED = "#535e6a"
SIGNATURE = "#2a3f8e"
SIGNATURE_SOFT = "#e4e8f5"
ALLOW = "#256240"
REVIEW = "#8a4f00"
ALARM = "#b0261c"
QUIET = "#a5afba"

SUBJECT = SIGNATURE
COMPARISON = INK_MUTED
THRESHOLD = ALARM

# For the few charts that carry more than two series, in this order.
SERIES = [SIGNATURE, INK_MUTED, LINE_STRONG, REVIEW, ALLOW, QUIET]

# Sequential map for cell heat maps: surface to signature.
SIGNATURE_MAP = LinearSegmentedColormap.from_list("reliax_signature", [SURFACE, SIGNATURE_SOFT, SIGNATURE])

FONT_DIR = pathlib.Path(__file__).resolve().parent / "fonts"


def register_fonts():
    """Register Inter from eval/fonts if present; return the family name to use."""
    found = False
    if FONT_DIR.is_dir():
        for f in sorted(FONT_DIR.glob("*.ttf")):
            try:
                font_manager.fontManager.addfont(str(f))
                found = True
            except Exception:
                pass
    return "Inter" if found else "DejaVu Sans"


def apply(font_size=9):
    family = register_fonts()
    plt.rcParams.update({
        "font.family": family,
        "font.size": font_size,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titleweight": "semibold",
        "axes.edgecolor": LINE_STRONG,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_MUTED,
        "ytick.labelcolor": INK_MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.grid": True,
        "grid.color": LINE,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.labelcolor": INK,
    })
    return family
