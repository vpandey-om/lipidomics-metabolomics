#!/usr/bin/env python3
# individual_box_plot.py
# Examples:
#   python individual_box_plot.py --file "/path/to/Final_Combined_Lipidomics.txt" --species "PC 34:1"
#   python individual_box_plot.py --file "/path/to/Final_Combined_Lipidomics.txt" --species "PC 34:1,PE 36:2"
#   python individual_box_plot.py --file "/path/to/Final_Combined_Lipidomics.txt" --pattern "^PC\\s"
#   python individual_box_plot.py --file "/path/to/Final_Combined_Lipidomics.txt" --species-file species.txt --format png

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from plotnine import (
    ggplot, aes, geom_boxplot, geom_jitter, geom_text, labs, theme_minimal, theme,
    element_text, element_blank, element_line, scale_x_discrete, scale_fill_manual,
    facet_wrap, coord_cartesian, geom_segment
)

# -------------------------
# CONFIG
# -------------------------
DEFAULT_OUTPUT_DIR = "species_boxplots"

sample_cols = [
    'Ctrl_1','Ctrl_2','Ctrl_3','Ctrl_4',
    'POI7_1','POI7_2','POI7_3','POI7_4',
    'POI8_1','POI8_2','POI8_3','POI8_4'
]
condition_order = ["Ctrl","POI7","POI8"]
condition_labels = {
    "Ctrl":"Control (mCherry_Luc)",
    "POI7":"RhoSH KO (POI7)",
    "POI8":"MAP1 KO (POI8)"
}
condition_colors = {"Ctrl":"#C1C0C0","POI7":"#010101","POI8":"#EE377A"}

EPS = 1e-12

# -------------------------
# Utils
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", str(name)).strip("_")

def to_numeric_1d(values) -> np.ndarray:
    arr = np.asarray(pd.Series(values).astype(float), dtype=float)
    return arr[~np.isnan(arr)]

def nice_p(p):
    if p is None or np.isnan(p): return "n/a"
    return f"{p:.2g}"

def p_to_stars(p: float | None) -> str:
    # Prism-style stars, but return "" when not significant (NO 'n.s.')
    if p is None or np.isnan(p): return ""
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return ""  # no 'n.s.' text

def detect_headgroup_column(df: pd.DataFrame) -> str | None:
    if "Headgroup" in df.columns:
        return "Headgroup"
    inferred = df["Metabolite name"].astype(str).str.extract(r"^([A-Za-z0-9\-]+)", expand=False)
    return "Headgroup_inferred" if inferred.notna().any() else None

def make_long(df_species_block: pd.DataFrame) -> pd.DataFrame:
    m = df_species_block.melt(
        id_vars=["Species"], value_vars=sample_cols,
        var_name="Sample", value_name="Value"
    )
    m["Condition"] = m["Sample"].str.extract(r"^(Ctrl|POI7|POI8)", expand=False)
    m = m.dropna(subset=["Condition"]).copy()
    m["Condition"] = pd.Categorical(m["Condition"], categories=condition_order, ordered=True)
    m["ConditionLabel"] = m["Condition"].map(condition_labels)
    return m

def median_log2fc_vs_ctrl(m_long: pd.DataFrame, cond: str) -> float | None:
    ctrl = to_numeric_1d(m_long.loc[m_long["Condition"]=="Ctrl","Value"])
    test = to_numeric_1d(m_long.loc[m_long["Condition"]==cond,"Value"])
    if ctrl.size==0 or test.size==0: return None
    return float(np.log2((np.median(test)+EPS)/(np.median(ctrl)+EPS)))

def welch_p_vs_ctrl(m_long: pd.DataFrame, cond: str) -> float | None:
    ctrl = to_numeric_1d(m_long.loc[m_long["Condition"]=="Ctrl","Value"])
    test = to_numeric_1d(m_long.loc[m_long["Condition"]==cond,"Value"])
    if ctrl.size==0 or test.size==0: return None
    _, p = ttest_ind(test, ctrl, equal_var=False, nan_policy="omit")
    return float(p)

# -------------------------
# Plotting
# -------------------------
def plot_species_box(m_long: pd.DataFrame, species: str, out_path: str,
                     y_limits=None, point_alpha=0.75, width=6, height=6):
    """
    Boxplot + jitter with:
      - Subtitle (small text) showing: 'RhoSH: p=.., log2FC=.. ; MAP1: p=.., log2FC=..'
      - Prism-style brackets + stars for Ctrl–RhoSH and Ctrl–MAP1
      - No ANOVA line, no 'n.s.' text anywhere
    """
    # Stats
    fc7 = median_log2fc_vs_ctrl(m_long, "POI7")
    p7  = welch_p_vs_ctrl(m_long, "POI7")
    fc8 = median_log2fc_vs_ctrl(m_long, "POI8")
    p8  = welch_p_vs_ctrl(m_long, "POI8")

    # y-positions for labels/brackets
    vmax = np.nanmax(m_long["Value"]) if np.isfinite(np.nanmax(m_long["Value"])) else 1.0
    ymax = float(vmax if vmax > 0 else 1.0)
    pad  = 0.08 * ymax
    ytext = ymax + pad

    # Build small subtitle with stats (NO 'n.s.')
    sub_parts = []
    if (fc7 is not None) or (p7 is not None):
        sub_parts.append(f"RhoSH: p={nice_p(p7)}, log2FC={fc7:.2f}")
    if (fc8 is not None) or (p8 is not None):
        sub_parts.append(f"MAP1: p={nice_p(p8)}, log2FC={fc8:.2f}")
    subtitle_txt = " ; ".join(sub_parts) if sub_parts else None

    # Base plot
    p = (
        ggplot(m_long, aes(x="ConditionLabel", y="Value", fill="Condition"))
        + geom_boxplot(outlier_shape=None, alpha=0.95, width=0.7)
        + geom_jitter(aes(x="ConditionLabel", y="Value"), width=0.12, size=2, alpha=point_alpha)
        + scale_x_discrete(limits=[condition_labels[c] for c in condition_order])
        + scale_fill_manual(values=condition_colors)
        + labs(
            title=f"{species}",
            subtitle=subtitle_txt,
            x="", y="Normalized intensity"
        )
        + theme_minimal(base_size=12)
        + theme(
            panel_grid=element_blank(),
            panel_background=element_blank(),
            axis_line=element_line(color="black"),
            axis_ticks_major=element_line(color="black"),
            axis_text_x=element_text(size=11, rotation=20, ha="right"),
            axis_text_y=element_text(size=11),
            axis_title_y=element_text(size=12, weight="bold"),
            plot_title=element_text(size=14, weight="bold"),
            plot_subtitle=element_text(size=10),  # smaller subtitle
            legend_position="none"
        )
    )

    # ------------- Prism-style brackets + stars -------------
    ctrl_x, poi7_x, poi8_x = 1, 2, 3
    y_bracket1 = ytext + 0.05 * ymax
    y_bracket2 = ytext + 0.15 * ymax

    brk_rows = []
    # Ctrl vs RhoSH (POI7)
    if p7 is not None:
        brk_rows += [
            {"x": ctrl_x, "xend": poi7_x, "y": y_bracket1, "yend": y_bracket1},  # horizontal
            {"x": ctrl_x, "xend": ctrl_x, "y": y_bracket1-0.02*ymax, "yend": y_bracket1},  # left tick
            {"x": poi7_x, "xend": poi7_x, "y": y_bracket1-0.02*ymax, "yend": y_bracket1},  # right tick
        ]
    # Ctrl vs MAP1 (POI8)
    if p8 is not None:
        brk_rows += [
            {"x": ctrl_x, "xend": poi8_x, "y": y_bracket2, "yend": y_bracket2},
            {"x": ctrl_x, "xend": ctrl_x, "y": y_bracket2-0.02*ymax, "yend": y_bracket2},
            {"x": poi8_x, "xend": poi8_x, "y": y_bracket2-0.02*ymax, "yend": y_bracket2},
        ]

    if brk_rows:
        brk_df = pd.DataFrame(brk_rows)
        star_rows = []
        s7 = p_to_stars(p7)
        s8 = p_to_stars(p8)
        if p7 is not None and s7:
            star_rows.append({"x": 1.5, "y": y_bracket1 + 0.02*ymax, "label": s7})
        if p8 is not None and s8:
            star_rows.append({"x": 2.0, "y": y_bracket2 + 0.02*ymax, "label": s8})
        p = p + geom_segment(aes(x="x", xend="xend", y="y", yend="yend"),
                             data=brk_df, inherit_aes=False)
        if star_rows:
            p = p + geom_text(aes(x="x", y="y", label="label"),
                              data=pd.DataFrame(star_rows), size=10, va="bottom",
                              inherit_aes=False)
    # ------------- end brackets -------------

    if y_limits:
        p = p + coord_cartesian(ylim=y_limits)

    ensure_dir(os.path.dirname(out_path))
    p.save(out_path, width=width, height=6, verbose=False)

def plot_faceted(species_to_long: dict[str, pd.DataFrame], out_path: str,
                 ncol=3, width=3.2, height=3.4):
    """Faceted grid with species name as strip; small stats in subtitle-like line not shown (facets don’t support subtitle).
       So we keep facets clean; stars/brackets are not drawn in facets for readability."""
    if not species_to_long:
        return

    panels = []
    for sp, m in species_to_long.items():
        tmp = m.copy()
        tmp["Facet"] = sp
        panels.append(tmp)
    long_all = pd.concat(panels, ignore_index=True)

    p = (
        ggplot(long_all, aes(x="ConditionLabel", y="Value", fill="Condition"))
        + geom_boxplot(outlier_shape=None, alpha=0.95, width=0.7)
        + geom_jitter(width=0.12, size=1.8, alpha=0.65)
        + scale_x_discrete(limits=[condition_labels[c] for c in condition_order])
        + scale_fill_manual(values=condition_colors)
        + facet_wrap("~Facet", ncol=ncol, scales="free_y")
        + labs(x="", y="Normalized intensity")
        + theme_minimal(base_size=11)
        + theme(
            panel_grid=element_blank(),
            axis_text_x=element_text(size=9, rotation=20, ha="right"),
            axis_title_y=element_text(size=11, weight="bold"),
            strip_text=element_text(size=10, weight="bold"),
            legend_position="none"
        )
    )

    ensure_dir(os.path.dirname(out_path))
    n = len(species_to_long)
    cols = ncol
    rows = int(np.ceil(n/cols))
    fig_w = max(6, cols*width)
    fig_h = max(5, rows*height)
    p.save(out_path, width=fig_w, height=fig_h, verbose=False)

# -------------------------
# Data I/O + prep
# -------------------------
def read_and_normalize(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\t")
    df.columns = df.columns.str.strip()
    if "Metabolite name" not in df.columns:
        raise ValueError("Missing 'Metabolite name' column in input file.")
    df = df[df["Metabolite name"].notna()].copy()
    df["Species"] = df["Metabolite name"].astype(str)

    # Optional: add inferred headgroup (not used, harmless)
    hg_col = detect_headgroup_column(df)
    if hg_col and hg_col not in df.columns:
        df[hg_col] = df["Metabolite name"].str.extract(r"^([A-Za-z0-9\-]+)", expand=False)

    for c in sample_cols:
        if c not in df.columns:
            raise ValueError(f"Missing sample column: {c}")

    keep = ["Species"] + sample_cols
    collapsed = (
        df[keep]
        .groupby(["Species"], as_index=False)[sample_cols]
        .sum()
        .copy()
    )

    col_sums = collapsed[sample_cols].sum(axis=0).replace(0, np.nan)
    norm = collapsed.copy()
    for c in sample_cols:
        norm[c] = norm[c] / col_sums[c]

    cs = norm[sample_cols].sum(axis=0)
    print("Column sums after normalization (≈1.0 each):")
    for c in sample_cols:
        print(f"  {c}: {cs[c]:.6f}")
    return norm

def pick_species(df_norm: pd.DataFrame, species: list[str]|None,
                 pattern: str|None, species_file: str|None) -> list[str]:
    chosen = set()
    all_species = df_norm["Species"].astype(str).unique().tolist()

    if species:
        for s in species:
            s = s.strip()
            if s in all_species:
                chosen.add(s)
            else:
                print(f"  [warn] species not found: {s}")

    if species_file and os.path.exists(species_file):
        with open(species_file, "r") as fh:
            for line in fh:
                s = line.strip()
                if s and s in all_species:
                    chosen.add(s)
                elif s:
                    print(f"  [warn] species not found (file): {s}")

    if pattern:
        rx = re.compile(pattern)
        for s in all_species:
            if rx.search(s):
                chosen.add(s)

    if not chosen:
        print("No species matched your selection. Exiting.", file=sys.stderr)
        sys.exit(2)

    return sorted(chosen)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Box plots for lipid species with FC, p-values, and Prism-style brackets.")
    ap.add_argument("--file", required=True, help="Path to the combined lipidomics TSV file.")
    ap.add_argument("--species", default=None, help="Comma-separated species list.")
    ap.add_argument("--species-file", default=None, help="Text file: one species per line.")
    ap.add_argument("--pattern", default=None, help="Regex to match species (applied to 'Species').")
    ap.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    ap.add_argument("--format", default="pdf", choices=["pdf","png"], help="Output format.")
    ap.add_argument("--ylims", default=None, help="Y-limits as 'min,max' (optional).")
    ap.add_argument("--facet-ncol", type=int, default=3, help="Columns for automatic subplot when >1 species.")
    args = ap.parse_args()

    df_norm = read_and_normalize(args.file)

    species_list = None
    if args.species:
        species_list = [s for s in args.species.split(",") if s.strip()]
    chosen_species = pick_species(df_norm, species_list, args.pattern, args.species_file)

    out_dir = args.outdir
    ensure_dir(out_dir)

    y_limits = None
    if args.ylims:
        try:
            lo, hi = map(float, args.ylims.split(","))
            y_limits = (lo, hi)
        except Exception:
            print("[warn] Could not parse --ylims; ignoring.")

    # Per-species plots
    species_to_long = {}
    for sp in chosen_species:
        sub = df_norm[df_norm["Species"] == sp][["Species"] + sample_cols].copy()
        if sub.empty:
            print(f"[skip] No data for {sp}")
            continue
        m_long = make_long(sub)
        species_to_long[sp] = m_long

        out_path = os.path.join(out_dir, f"{sanitize_filename(sp)}_boxplot.{args.format}")
        plot_species_box(m_long, sp, out_path, y_limits=y_limits)
        print(f"Saved: {out_path}")

    # Automatic subplot (faceted) if multiple species provided
    if len(species_to_long) > 1:
        out_path = os.path.join(out_dir, f"species_boxplots_faceted.{args.format}")
        plot_faceted(species_to_long, out_path, ncol=args.facet_ncol)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
