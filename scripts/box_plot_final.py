#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse
from typing import List, Dict
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']

from pathlib import Path

# Script lives in: <repo>/scripts/...
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]              # -> /Users/vikash/git_hub/lipidomics-metabolomics

# Defaults relative to the repo layout you showed
DEFAULT_RAW = REPO_ROOT / "data" / "Final_Combined_Lipidomics.txt"
DEFAULT_OUTDIR = REPO_ROOT / "PLmode2_allinone"


from plotnine import (
    ggplot, aes, geom_boxplot, geom_point, position_dodge,
    scale_fill_manual, scale_color_manual, scale_x_discrete,
    scale_y_continuous, scale_y_log10, labs,
    theme, theme_bw, element_text, element_blank, element_line,
    guides, guide_legend,theme_minimal
)
from plotnine.ggplot import save_as_pdf_pages

# ------------------ CONFIG ------------------
# DEFAULT_RAW = "/Users/vikash/Documents/Thorey and Martina_Umea/Analysis/Lipid_scripts/thoreylipid/Final_Combined_Lipidomics.txt"
# DEFAULT_OUTDIR = "PLmode2_allinone"

sample_cols = [
    'Ctrl_1','Ctrl_2','Ctrl_3','Ctrl_4',
    'POI7_1','POI7_2','POI7_3','POI7_4',
    'POI8_1','POI8_2','POI8_3','POI8_4'
]
condition_order = ["Ctrl","POI7","POI8"]
condition_labels = {"Ctrl":"Control (mCherry_Luc)","POI7":"RhoSH KO (POI7)","POI8":"MAP1 KO (POI8)"}
condition_colors = {"Ctrl":"#C1C0C0","POI7":"#010101","POI8":"#EE377A"}

PL_DEFAULT_ORDER = ["PL","lysoPL","MG","DG","TG"]
ALIASES_PL = {
    "PL_SET":["PL"], "LYSO_SET":["lysoPL"], "NEUTRAL":["DG","TG","MG"],
    "ALL_DEFAULT":PL_DEFAULT_ORDER, "AUTO":PL_DEFAULT_ORDER
}

# ------------------ CLI ------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compact, multi-page PL/headgroup boxplots with one-time scaling.")
    p.add_argument("--raw", default=DEFAULT_RAW)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--domain", choices=["pl","headgroup"], default="pl")
    p.add_argument("--groups", default="auto", help="Commas=same page; semicolons=new page. e.g. 'PL,lysoPL;MG'")
    p.add_argument("--xtick-rot", type=int, choices=[0,45,90], default=45)

    # scaling (computed ONCE)
    p.add_argument("--scale-mode", choices=["fraction","none"], default="none",
                   help="'none' shows raw values on log10 y by default (see --no-log).")
    p.add_argument("--denominator", choices=["selected","all"], default="selected")
    p.add_argument("--as-percent", action="store_true", default=False)

    # display
    p.add_argument("--no-log", action="store_true",
                   help="If given and --scale-mode none, use linear y instead of log10.")
    p.add_argument("--xtick-size", type=int, default=20)
    p.add_argument("--ytick-size", type=int, default=25)
    p.add_argument("--axis-title-size", type=int, default=18)
    return p.parse_args()

# ------------------ IO & grouping ------------------
def ensure_outdir(path: str): os.makedirs(path, exist_ok=True)

def read_raw_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()
    df = df[df["Metabolite name"].notna()].copy()
    if df["Metabolite name"].duplicated().any():
        keep = ["Metabolite name"] + sample_cols
        df = df[keep].groupby("Metabolite name", as_index=False, sort=False).sum()
    return df

def group_pl(df: pd.DataFrame) -> Dict[str, list]:
    from utils import group_lipids
    gl, _ = group_lipids(df, mode="PL")
    return {k:v for k,v in gl.items() if k in {"PL","lysoPL","DG","TG","MG"} and v}

def group_headgroup(df: pd.DataFrame) -> Dict[str, list]:
    from utils import group_lipids
    hg, _ = group_lipids(df, mode="Headgroup")
    return {k:v for k,v in hg.items() if isinstance(k,str) and v}

# ------------------ scaling (once) ------------------
def build_domain_tables(df: pd.DataFrame, members: Dict[str,list],
                        scale_mode: str, denominator: str, as_percent: bool,
                        log10_for_raw: bool, denom_selected_groups: List[str],
                        domain_tag: str):
    EPS = 1e-9  # small positive to make log-scale safe

    # raw sums (group × sample)
    rows = []
    order = list(members.keys())
    for g in order:
        gdf = df[df["Metabolite name"].isin(members[g])]
        s = (gdf[sample_cols].sum(axis=0) if not gdf.empty else pd.Series(0.0, index=sample_cols))
        rows.append(s.rename(g))
    group_sum = pd.DataFrame(rows, index=order)

    if scale_mode == "fraction":
        # one-time fraction scaling
        if denominator == "selected":
            present = [g for g in denom_selected_groups if g in group_sum.index]
            if not present:
                raise SystemExit("No groups available for 'selected' denominator.")
            denom = group_sum.loc[present].sum(axis=0).replace(0, np.nan)
        elif denominator == "all":
            denom = df[sample_cols].sum(axis=0).replace(0, np.nan)
        else:
            raise ValueError("denominator must be 'selected' or 'all'")
        group_sum = group_sum.divide(denom, axis=1).fillna(0.0)
        # linear y for fractions
        use_log10_scale = True
        y_label = "Fraction (%)" if as_percent else "Fraction"
        if as_percent:
            group_sum = group_sum * 100.0
            y_limits = (0, 100)
        else:
            y_limits = (0, 1)

    else:
        # RAW numbers — DO NOT pre-log. Just make strictly positive for log scale.
        group_sum = (group_sum + EPS).clip(lower=EPS)
        use_log10_scale = bool(log10_for_raw)  # True ⇒ plot with scale_y_log10()
        y_label = "Summed Intensity (log10)" if use_log10_scale else "Summed Intensity"
        y_limits = None  # let log scale choose breaks

    tidy = (group_sum.reset_index(names="Group")
            .melt(id_vars="Group", var_name="Sample", value_name="Value"))
    tidy["Condition"] = tidy["Sample"].str.extract(r"(^[A-Za-z]+\d*)")[0]
    tidy = tidy[tidy["Condition"].isin(condition_order)].copy()
    tidy["ConditionLabel"] = tidy["Condition"].map(condition_labels)
    tidy["Group"] = tidy["Group"].astype(str)
    tidy["Condition"] = pd.Categorical(tidy["Condition"], categories=condition_order, ordered=True)
    tidy["ConditionLabel"] = pd.Categorical(
        tidy["ConditionLabel"], categories=[condition_labels[c] for c in condition_order], ordered=True
    )

    return tidy, (y_label, y_limits, use_log10_scale)


# ------------------ panels ------------------
def expand_alias_pl(tok: str) -> List[str]:
    key = tok.strip().upper().replace("-","_")
    if key in ALIASES_PL: return ALIASES_PL[key][:]
    if key in {"PLSET","PL"}: return ["PL"]
    if key in {"LYSOSET","LYSOPL"}: return ["lysoPL"]
    if key in {"NEUTRALS"}: return ["DG","TG","MG"]
    return [tok.strip()]

def spec_to_panels(spec: str, domain: str, present: List[str]) -> List[List[str]]:
    spec = spec.strip()
    if domain == "pl" and (spec.lower() == "auto" or spec.upper() == "ALL_DEFAULT"):
        wanted = [g for g in PL_DEFAULT_ORDER if g in present]
        if not wanted: sys.exit("No PL groups present for auto.")
        return [wanted]
    pages = []
    for raw_page in spec.split(";"):
        if not raw_page.strip(): continue
        tokens = [t for t in raw_page.split(",") if t.strip()]
        expanded = []
        for t in tokens:
            expanded += (expand_alias_pl(t) if domain == "pl" else [t.strip()])
        seen = set(); page=[]
        for g in expanded:
            if g in present and g not in seen:
                page.append(g); seen.add(g)
        if page: pages.append(page)
    if not pages: sys.exit("Requested groups are not present.")
    return pages

# ------------------ plotting (compact, thick axis, big ticks) ------------------
from plotnine import (
    ggplot, aes, geom_boxplot, geom_point, position_dodge,
    scale_fill_manual, scale_color_manual, scale_x_discrete,
    scale_y_continuous, scale_y_log10, labs,
    theme_bw, theme, element_text, element_blank, element_line
)

def build_page_plot(tidy, groups, yconf, rotation, fig_width, xtick_size, ytick_size, axis_title_size):
    y_label, y_limits, use_log10_scale = yconf
    y_limits = None
    present = [g for g in groups if g in set(tidy["Group"])]
    if not present:
        raise ValueError(f"No requested groups present: {groups}")

    dfp = tidy[tidy["Group"].isin(present)].copy()
    dfp["Group"] = pd.Categorical(dfp["Group"], categories=present, ordered=True)

    dodge_w = 0.66
    box_w   = 0.60

    p = (
        ggplot(dfp, aes(x="Group", y="Value", fill="ConditionLabel"))
        + geom_boxplot(position=position_dodge(width=dodge_w),
                       outlier_shape=None, width=box_w, color="#000000")
        + geom_point(aes(color="ConditionLabel"),
                     position=position_dodge(width=dodge_w), size=2.0, stroke=0.3)
        + scale_fill_manual(values=[condition_colors[c] for c in condition_order])
        + scale_color_manual(values=[condition_colors[c] for c in condition_order])
        + scale_x_discrete(limits=present, expand=(0.025, 0.025))      # no edge padding
        + labs(x="Headgroup", y='Summed relative intensity')
        + theme_minimal()
        + theme(
            figure_size=(fig_width, 5),                       # fixed height 6"
            panel_grid=element_blank(),
            panel_border=None,
            legend_position="none",                              # no legend
            axis_line=element_line(color="black", size=3),
            axis_ticks_major=element_line(color="black", size=3),
            axis_text_x=element_text(size=xtick_size, rotation=rotation, ha=("right" if rotation else "center")),
            axis_text_y=element_text(size=ytick_size),
            axis_title_x=element_text(size=axis_title_size, weight="bold"),
            axis_title_y=element_text(size=axis_title_size, weight="bold"),
            subplots_adjust={'left': 0.05, 'right': 0.995, 'bottom': 0.16, 'top': 0.985},
        )
    )

    # Choose ONE y-scale (no trailing commas!)
    if use_log10_scale:
        p = p + scale_y_log10()
    elif y_limits is not None:
        p = p + scale_y_continuous(limits=y_limits)

    return p  # <- no comma here




# ------------------ main ------------------
def main():
    a = parse_args()
    ensure_outdir(a.outdir)

    print(f"📥 Loading: {a.raw}")
    df = read_raw_table(a.raw)

    if a.domain == "pl":
        members = group_pl(df); domain_tag = "pl"
        denom_sel = [g for g in PL_DEFAULT_ORDER if g in members]
        if not denom_sel: sys.exit("No PL-domain groups found.")
        present_groups = list(members.keys())
    else:
        members = group_headgroup(df); domain_tag = "headgroup"
        denom_sel = list(members.keys())
        present_groups = list(members.keys())
        if not present_groups: sys.exit("No headgroups found.")

    tidy, yconf = build_domain_tables(
        df=df, members=members,
        scale_mode=a.scale_mode, denominator=a.denominator,
        as_percent=a.as_percent,
        log10_for_raw=(a.scale_mode=="none" and not a.no_log),
        denom_selected_groups=denom_sel, domain_tag=domain_tag
    )

    panels = spec_to_panels(a.groups, a.domain, present_groups)
    print(f"🧩 Pages: {panels}")

    out_pdf = os.path.join(a.outdir, f"{domain_tag}_panels_{a.scale_mode}_{a.denominator}{'_pct' if a.as_percent else ''}{'_log10' if (a.scale_mode=='none' and not a.no_log) else ''}.pdf")
    print(f"🖨️  PDF: {out_pdf}")

    plots = []
    for page in panels:
       
        width_inches = 1 * len(page)  # compact width ≈ 1.0" per group
        try:
            plots.append(build_page_plot(
                tidy, page, yconf=yconf, rotation=a.xtick_rot,
                fig_width=width_inches, xtick_size=a.xtick_size,
                ytick_size=a.ytick_size, axis_title_size=a.axis_title_size
            ))
            print(f"  • page: {page}")
        except Exception as e:
            print(f"  ! skipped {page}: {e}")

    if not plots: sys.exit("Nothing to plot.")
    save_as_pdf_pages(plots, filename=out_pdf, verbose=False)
    print("🎉 Done.")

if __name__ == "__main__":
    main()

# python box_plot_final.py --domain headgroup --groups "PC,PE,PI,SM,HexCer" --xtick-rot 90