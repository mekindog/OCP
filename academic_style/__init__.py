"""
academic_style
==============

House style for academic matplotlib figures. It sets fonts, sizes, line
widths, ticks, colours and export settings only; the plot type is up to
the script.

Usage
-----
    import academic_style as acs

    acs.use("report")                 # "report" | "aps" | "nature" | "slides"
    fig, ax = acs.figure()            # created at final printed width
    ax.plot(x, y, label="data")
    ax.set_xlabel("Wavelength (nm)")  # units in parentheses
    acs.save(fig, "figures/spectrum") # writes spectrum.pdf and spectrum.png

Modes
-----
report  A4 single-column report or thesis. Computer Modern serif to match
        default LaTeX body text, 10 pt.
aps     APS / Optica two-column journals. Single column 8.5 cm
        (3 3/8 in), Times-like serif, 8 pt (APS: lettering >= 2 mm).
nature  Nature-family journals. Single column 88 mm, sans-serif, 7 pt
        (Nature: 5-7 pt sans-serif).
slides  Talks and posters. 16:9 friendly, large text, thick lines.

Design rules (sources: APS Journals Style Guide; Nature final artwork
guide; AAS Graphics Guide; Rougier et al., PLoS Comput. Biol. 10,
e1003833 (2014))
    * no gridlines, white background, frameless legend
    * lines >= 0.5 pt, solid markers >= 1 mm
    * colour-blind-safe Okabe-Ito colours, each series also gets its own
      line style so figures survive greyscale printing
    * viridis for sequential maps, a diverging map for signed data
    * vector PDF plus 600 dpi PNG
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from string import ascii_lowercase

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

__all__ = [
    "use", "style", "figure", "save", "label_panels", "width",
    "COLORS", "CYCLE", "LINESTYLES", "MARKERS", "CMAP_SEQ", "CMAP_DIV",
    "MODES",
]

# ---------------------------------------------------------------------------
# Colours (Okabe & Ito, "Color Universal Design", 2008)
# Yellow (#F0E442) is left out of the default cycle: too little contrast on
# white for thin lines. It is still available by name.
# ---------------------------------------------------------------------------
COLORS = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "skyblue": "#56B4E9",
    "orange": "#E69F00",
    "black": "#000000",
    "yellow": "#F0E442",
    "grey": "#7F7F7F",
}
CYCLE = [COLORS[k] for k in
         ("blue", "vermillion", "green", "purple", "skyblue", "orange", "black")]
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 3))]
MARKERS = ["o", "s", "^", "D", "v", "p", "X"]

CMAP_SEQ = "viridis"   # magnitudes, intensities
CMAP_DIV = "RdBu_r"    # signed data; centre it with matplotlib.colors.CenteredNorm

# ---------------------------------------------------------------------------
# Per-mode settings. Widths in inches.
# ---------------------------------------------------------------------------
_MM = 1 / 25.4

MODES = {
    "report": dict(
        single=6.0, double=6.0, aspect=0.62,
        font=10, small=9, line=1.5, axes=0.8, marker=4.5,
        family="cm", panel="({})", tick=4.0,
    ),
    "aps": dict(
        single=3.375, double=7.0, aspect=0.75,
        font=8, small=8, line=1.0, axes=0.6, marker=3.5,
        family="times", panel="({})", tick=3.5,
    ),
    "nature": dict(
        single=88 * _MM, double=180 * _MM, aspect=0.75,
        font=7, small=6, line=1.0, axes=0.6, marker=3.0,
        family="sans", panel="{}", tick=3.0,          # Nature uses bold lowercase a, b
    ),
    "slides": dict(
        single=8.0, double=12.0, aspect=0.56,
        font=18, small=16, line=2.5, axes=1.2, marker=8.0,
        family="sans", panel="({})", tick=7.0,
    ),
}

_FONTS = {
    # Computer Modern ships with matplotlib (cmr10), but its TeX encoding puts
    # ligatures at Latin-1 code points (a typed degree sign shows as "fl").
    # A copy restricted to ASCII is swapped in; STIX supplies every other
    # glyph (degree, micro, plus-minus, Greek typed as text).
    "cm": {
        "font.family": ["cmr10", "STIXGeneral"],
        "mathtext.fontset": "cm",
    },
    "times": {
        "font.family": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
    },
    "sans": {
        "font.family": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
    },
}

_current = {"mode": "report"}


def _register_cm_ascii() -> None:
    """Swap matplotlib's cmr10 for a copy whose Latin-1 code points are empty.

    The family name stays "cmr10", so matplotlib's own cmr10 handling (minus
    and times signs from cmsy10) keeps working, while typed characters such
    as the degree sign fall through to STIX instead of showing ligatures.
    """
    from matplotlib import font_manager
    fm = font_manager.fontManager
    dst = Path(mpl.get_cachedir()) / "academic_style" / "cmr10_ascii.ttf"
    if any(Path(f.fname) == dst for f in fm.ttflist):
        return
    if not dst.exists():
        from fontTools.ttLib import TTFont
        font = TTFont(Path(mpl.get_data_path()) / "fonts" / "ttf" / "cmr10.ttf")
        for table in font["cmap"].tables:
            table.cmap = {k: v for k, v in table.cmap.items() if k < 128}
        dst.parent.mkdir(parents=True, exist_ok=True)
        font.save(dst)
    fm.ttflist = [f for f in fm.ttflist if f.name != "cmr10"]
    fm.addfont(str(dst))
    fm._findfont_cached.cache_clear()


def _installed(families: list[str]) -> list[str]:
    """Keep only fonts present on this machine (avoids findfont warnings)."""
    from matplotlib import font_manager
    if "cmr10" in families:
        try:
            _register_cm_ascii()
        except Exception:  # cache not writable: keep the stock cmr10
            pass
    names = {f.name for f in font_manager.fontManager.ttflist}
    found = [f for f in families if f in names]
    return found or ["DejaVu Sans"]


def _rc(mode: str, ticks: str = "in", minor: bool = True) -> dict:
    m = MODES[mode]
    fs, sm = m["font"], m["small"]
    major = m["tick"]
    rc = {
        # Figure
        "figure.figsize": (m["single"], m["single"] * m["aspect"]),
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "figure.constrained_layout.use": True,
        # Text
        "font.size": fs,
        "axes.labelsize": fs,
        "axes.titlesize": fs,
        "xtick.labelsize": sm,
        "ytick.labelsize": sm,
        "legend.fontsize": sm,
        "legend.title_fontsize": sm,
        "axes.unicode_minus": True,
        # Axes: no grid, plain white background, all four spines
        "axes.grid": False,
        "axes.facecolor": "white",
        "axes.linewidth": m["axes"],
        "axes.prop_cycle": cycler(color=CYCLE) + cycler(linestyle=LINESTYLES),
        "axes.formatter.limits": (-3, 4),
        "axes.formatter.use_mathtext": True,
        "axes.labelpad": 3.0,
        # Ticks
        "xtick.direction": ticks,
        "ytick.direction": ticks,
        "xtick.top": ticks == "in",
        "ytick.right": ticks == "in",
        "xtick.minor.visible": minor,
        "ytick.minor.visible": minor,
        "xtick.major.size": major,
        "ytick.major.size": major,
        "xtick.minor.size": major * 0.55,
        "ytick.minor.size": major * 0.55,
        "xtick.major.width": m["axes"],
        "ytick.major.width": m["axes"],
        "xtick.minor.width": m["axes"] * 0.75,
        "ytick.minor.width": m["axes"] * 0.75,
        # Lines and markers
        "lines.linewidth": m["line"],
        "lines.markersize": m["marker"],
        "lines.markeredgewidth": m["axes"],
        "errorbar.capsize": m["marker"] * 0.6,
        "patch.linewidth": m["axes"],
        "hatch.linewidth": m["axes"],
        "scatter.edgecolors": "face",
        # Legend: no frame, compact
        "legend.frameon": False,
        "legend.handlelength": 2.0,
        "legend.borderaxespad": 0.4,
        "legend.labelspacing": 0.3,
        # Images
        "image.cmap": CMAP_SEQ,
        "image.interpolation": "nearest",
        # Export: vector PDF with editable, embedded TrueType fonts
        "savefig.dpi": 600,
        "savefig.format": "pdf",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    rc.update(_FONTS[m["family"]])
    return rc


def use(mode: str = "report", *, ticks: str = "in", minor: bool = True,
        latex: bool = False) -> None:
    """Apply the house style globally.

    mode   "report", "aps", "nature" or "slides"
    ticks  "in" (inward, on all four sides) or "out" (bottom and left only)
    minor  show minor ticks
    latex  render all text with a local LaTeX install (slower; needs LaTeX
           on PATH). Gives an exact font match with the document.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
    if ticks not in ("in", "out"):
        raise ValueError("ticks must be 'in' or 'out'")
    plt.rcdefaults()
    rc = _rc(mode, ticks, minor)
    rc["font.family"] = _installed(rc["font.family"])
    mpl.rcParams.update(rc)
    if latex:
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif" if MODES[mode]["family"] == "sans" else "serif",
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{siunitx}",
        })
    _current["mode"] = mode


@contextmanager
def style(mode: str = "report", **kwargs):
    """Temporarily apply a mode, e.g. to make one slide figure in a report script."""
    saved = mpl.rcParams.copy()
    previous = _current["mode"]
    try:
        use(mode, **kwargs)
        yield
    finally:
        mpl.rcParams.update(saved)
        _current["mode"] = previous


def width(span: str | float = "single") -> float:
    """Figure width in inches for the active mode.

    span  "single", "double", or a number: a fraction (<= 1) of the double
          width, or an absolute width in inches (> 1).
    """
    m = MODES[_current["mode"]]
    if span in ("single", "double"):
        return m[span]
    span = float(span)
    return span * m["double"] if span <= 1 else span


def figure(nrows: int = 1, ncols: int = 1, *, span: str | float = "single",
           aspect: float | None = None, height: float | None = None, **kwargs):
    """plt.subplots() at the final printed size.

    aspect  height / width (mode default if None); ignored if height is given
    height  explicit height in inches
    kwargs  passed to plt.subplots (sharex, sharey, gridspec_kw, ...)
    """
    w = width(span)
    if height is None:
        height = w * (aspect if aspect is not None else MODES[_current["mode"]]["aspect"])
    return plt.subplots(nrows, ncols, figsize=(w, height), **kwargs)


def label_panels(axes, *, loc: tuple[float, float] | None = None,
                 outside: bool = False, fmt: str | None = None, start: int = 0,
                 **kwargs) -> None:
    """Add (a), (b), ... to the top-left corner of each axis.

    outside  False: inside the frame (default). True: just above the frame,
             for panels where data fills the top-left corner.
    loc      position in axis coordinates (overrides `outside`)
    fmt      format string, e.g. "({})" or "{}"; mode default if None
    kwargs   passed to ax.text (e.g. color="white" on dark images)
    """
    if loc is None:
        loc = (0.0, 1.02) if outside else (0.03, 0.97)
    m = MODES[_current["mode"]]
    fmt = fmt or m["panel"]
    axes = axes.flat if hasattr(axes, "flat") else axes
    style_kw = dict(ha="left", va="bottom" if outside else "top", fontsize=m["font"],
                    fontweight="bold" if m["family"] == "sans" else "normal")
    style_kw.update(kwargs)
    for i, ax in enumerate(axes):
        ax.text(*loc, fmt.format(ascii_lowercase[start + i]),
                transform=ax.transAxes, **style_kw)


def save(fig, path: str | Path, formats=("pdf", "png"), **kwargs) -> list[Path]:
    """Save a figure as vector PDF plus 600 dpi PNG (by default).

    The figure is saved at its exact size (no tight cropping), so text keeps
    its true point size when included at full width in LaTeX.
    Creates the output folder if needed. Returns the written paths.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        out = path.with_suffix(f".{ext}")
        fig.savefig(out, **kwargs)
        written.append(out)
    return written


# Apply the default mode on import so a bare `import academic_style` already
# gives sensible figures.
use("report")
