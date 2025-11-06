import os
import re
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind  # Welch's t-test by default (equal_var=False)
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from plotnine import annotate

from plotnine import (
    ggplot, aes, geom_point, geom_errorbar, stat_summary, geom_text,
    facet_wrap, labs, theme_minimal, theme, element_text, scale_color_manual,
    geom_hline, geom_vline, coord_cartesian, scale_y_continuous, geom_col,
    guides, position_jitter,element_blank,element_line
)
# imports – add this and (optionally) remove ttest_ind
from scipy.stats import f_oneway
 
from utils import group_lipids  # Ensure this returns: grouped_lipids, _

# -------------------------
# CONFIG
# # -------------------------
# file_path = "/Users/vikash/Documents/Thorey and Martina_Umea/Analysis/Lipid_scripts/thoreylipid/Final_Combined_Lipidomics.txt"
# output_dir = "headgroup_plots_sum"

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repo root

file_path = ROOT / "data" / "Final_Combined_Lipidomics.txt"
output_dir = ROOT / "headgroup_plots_sum"

grouping_modes = ["Headgroup"]
sample_cols = [
    'Ctrl_1', 'Ctrl_2', 'Ctrl_3', 'Ctrl_4',
    'POI7_1', 'POI7_2', 'POI7_3', 'POI7_4',
    'POI8_1', 'POI8_2', 'POI8_3', 'POI8_4'
]
LOG2_FC_CUTOFF = np.log2(3 / 2)
EPS = 1e-9

headgroup_full_names = {
    "PA": "Phosphatidic Acid", "PC": "Phosphatidylcholine", "PE": "Phosphatidylethanolamine",
    "PG": "Phosphatidylglycerol", "PI": "Phosphatidylinositol", "PS": "Phosphatidylserine",
    "CL": "Cardiolipin", "MLCL": "Monolysocardiolipin", "HBMP": "Hydroxybutyl Monophosphate",
    "LPC": "Lysophosphatidylcholine", "LPE": "Lysophosphatidylethanolamine", "LPI": "Lysophosphatidylinositol",
    "LPS": "Lysophosphatidylserine", "DG": "Diglyceride", "TG": "Triglyceride", "MG": "Monoglyceride",
    "Dioctyl": "Dioctylglycerol", "Cer": "Ceramide", "HexCer": "Hexosylceramide",
    "Hex2Cer": "Dihexosylceramide", "SM": "Sphingomyelin", "SPB": "Sphingoid Base",
    "FA": "Fatty Acid", "FAHFA": "Fatty Acid Ester of Hydroxy Fatty Acid", "CAR": "Acylcarnitine",
    "Adenosine": "Nucleoside-derived Lipid", "ST": "Sterol", "NAGly": "N-Arachidonoyl Glycine",
    "NAGlySer": "N-Arachidonoyl Glycylserine", "SL": "Sulfolipid", "PE-Cer": "Phosphoethanolamine Ceramide"
}

# -------------------------
# Helpers
# -------------------------
def sanitize_filename(name):
    return re.sub(r"[^\w\-]", "_", str(name))

# calc_stats — make it match script 1 exactly
def calc_stats(ctrl, poi):
    ctrl = pd.Series(ctrl).dropna()
    poi = pd.Series(poi).dropna()
    if ctrl.empty or poi.empty:
        return None, None
    # SAME FC formula/epsilon placement as script 1
    fc = np.log2(np.median(poi) / np.median(ctrl) + 1e-9)
    # SAME test as script 1
    pval = f_oneway(ctrl, poi)[1]
    return fc, float(pval)


def summarize_mean_sd(y):
    y = np.asarray(y, dtype=float)
    m = np.nanmean(y)
    s = np.nanstd(y, ddof=0)
    # plotnine's stat_summary(fun_data=...) must return a DataFrame
    return pd.DataFrame({"y": [m], "ymin": [m - s], "ymax": [m + s]})


def make_volcano_plot(stats_df, title, filename, annotations=None):
    """
    Volcano with fixed x-axis [-2, 3], p=0.05 line, ±log2(1.5) dashed, +log2(8) dotted.
    Coloring:
      - Significant Up   (log2FC>0 & significant)  -> #3D88C8
      - Significant Down (log2FC<0 & significant)  -> #212D75
      - Not significant                            -> #8E8E8E
    Significance:
      (abs(log2FC) > LOG2_FC_CUTOFF & p < 0.062) OR (log2FC >= log2(8))
    Labels: full names for all significant points (black).
    annotations: list of {"group": "<GroupCode>", "dx": float, "dy": float, ["size": float, "color": str]}
                 Places text = full headgroup name at (x+dx, y+dy) based on that group's point.
    """
    if stats_df.empty:
        return

    df = stats_df.copy()
    df["nlog10p"] = -np.log10(df["pval"])

    alpha = 0.05
    fc15 = np.log2(3/2)  # ~0.585 (1.5×)

    # EXACTLY like script 1
    sig_rule = (df["log2FC"].abs() > fc15) & (df["pval"] < alpha)
    df["is_sig"] = sig_rule


    # Color categories
    df["Category"] = np.where(
        df["is_sig"] & (df["log2FC"] > 0), "Significant Up",
        np.where(df["is_sig"] & (df["log2FC"] < 0), "Significant Down", "Not significant")
    )

    # Full labels for significant points + precomputed label positions
    df["Label"] = df["Group"].map(lambda g: headgroup_full_names.get(g, g))
    df_lab = df[df["is_sig"]].copy()
    if not df_lab.empty:
        df_lab["label_x"] = np.where(df_lab["log2FC"] >= 0,
                                     df_lab["log2FC"] + 0.05,
                                     df_lab["log2FC"] - 0.05)
        df_lab["label_y"] = df_lab["nlog10p"] + 0.05
    
    p = (
        ggplot(df, aes("log2FC", "nlog10p", color="Category"))
        + geom_point(alpha=0.9, size=4)
        + scale_color_manual(values={
            "Significant Up": "#3D88C8",
            "Significant Down": "#212D75",
            "Not significant": "#8E8E8E",
        })
        + geom_hline(yintercept=-np.log10(0.05), linetype="dashed",alpha=0.2)     # p=0.05
        + geom_vline(xintercept=[-fc15, fc15], linetype="dashed",alpha=0.2)       # ±1.5×
        # + geom_vline(xintercept=[fc8], linetype="dotted",alpha=0.2)               # +8× (positive)
        + labs(title='', x="log2 Fold Change", y="-log10(p-value)", color="")
        + theme_minimal(base_size=12)
        + theme(
            panel_grid=element_blank(),
            panel_background=element_blank(),
            axis_line=element_line(color="black", size=4),
            axis_ticks_major=element_line(color="black", size=4),
            axis_text_x=element_text(size=21),
            axis_text_y=element_text(size=25),
            axis_title_x=element_text(size=21, weight="bold"),
            axis_title_y=element_text(size=21, weight="bold"),
            legend_position="none",
        )
        + coord_cartesian(xlim=(-2, 5))
    )
    text_size=12
    # Black labels for significant points
    ann_groups = set([a.get("group") for a in (annotations or []) if a.get("group")])
    if not df_lab.empty and ann_groups:
        df_lab = df_lab[~df_lab["Group"].isin(ann_groups)]
    if not df_lab.empty:
        p = p + geom_text(
            aes(x="label_x", y="label_y", label="Label"),
            data=df_lab,
            size=text_size,
            color="black"
        )

    # Optional custom annotations: [{"group": "LPE", "dx": 0.10, "dy": 0.15}, ...]
    if annotations:
        for note in annotations:
            grp = note.get("group")
            if not grp:
                continue
            base = df[df["Group"] == grp]
            if base.empty:
                continue
            row = base.iloc[0]
            base_x = float(row["log2FC"])
            base_y = float(row["nlog10p"])
            dx = float(note.get("dx", 0.0))
            dy = float(note.get("dy", 0.0))
            txt = headgroup_full_names.get(grp, grp)
            p = p + annotate(
                "text",
                x=base_x + dx,
                y=base_y + dy,
                label=txt,
                size=float(note.get("size", text_size)),
                color=note.get("color", "black")
            )

    p.save(filename=filename, width=6, height=6, verbose=False)

# -------------------------
# Load Data
# -------------------------
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(file_path, sep="\t")
df.columns = df.columns.str.strip()
df = df[df["Metabolite name"].notna()].copy()

# -------------------------
# Main Loop
# -------------------------
for mode in grouping_modes:
    grouped_lipids, _ = group_lipids(df, mode)
    print(f"\n🔹 Grouping Mode: {mode} | Total Groups: {len(grouped_lipids)}")

    stats_list_poi7 = []
    stats_list_poi8 = []

    red_pdf_path = os.path.join(output_dir, sanitize_filename(mode) + "_significant_groups.pdf")
    gray_pdf_path = os.path.join(output_dir, sanitize_filename(mode) + "_nonsignificant_groups.pdf")

    red_pdf = PdfPages(red_pdf_path)
    gray_pdf = PdfPages(gray_pdf_path)

    # Iterate groups
    for group_name, lipids in grouped_lipids.items():
        group_df = df[df["Metabolite name"].isin(lipids)].copy()
        if group_df.empty:
            continue

        summed = group_df[sample_cols].sum()
        samples_df = summed.reset_index()
        samples_df.columns = ["Sample", "SummedValue"]
        # Extract "Ctrl", "POI7", "POI8"
        samples_df["Condition"] = samples_df["Sample"].str.extract(r"(^[A-Za-z]+\d*)")[0]
        samples_df["Log2Value"] = np.log2(samples_df["SummedValue"] + EPS)

        ctrl = samples_df[samples_df["Condition"] == "Ctrl"]["SummedValue"]
        poi7 = samples_df[samples_df["Condition"] == "POI7"]["SummedValue"]
        poi8 = samples_df[samples_df["Condition"] == "POI8"]["SummedValue"]

        fc7, pval7 = calc_stats(ctrl, poi7)
        fc8, pval8 = calc_stats(ctrl, poi8)

        if None in (fc7, pval7, fc8, pval8):
            continue

        stats_list_poi7.append({"Group": group_name, "log2FC": fc7, "pval": pval7})
        stats_list_poi8.append({"Group": group_name, "log2FC": fc8, "pval": pval8})

        highlight = any([
            abs(fc7) > LOG2_FC_CUTOFF and pval7 < 0.05,
            abs(fc8) > LOG2_FC_CUTOFF and pval8 < 0.05
        ])

        title = (
            f"{group_name} | {mode} | {len(lipids)} lipids\n"
            f"POI7 vs Ctrl: log2FC={fc7:.2f}, p={pval7:.1e} | "
            f"POI8 vs Ctrl: log2FC={fc8:.2f}, p={pval8:.1e}"
        )

        # Long format for two-panel facet (Raw vs Log2)
        long_dat = pd.concat([
            samples_df.assign(Scale="Raw Intensity", Value=samples_df["SummedValue"]),
            samples_df.assign(Scale="Log2 Intensity", Value=samples_df["Log2Value"])
        ], ignore_index=True)

        # Compute center for Log2 panel to mimic y-range ±2 (with small padding)
        center = np.median(samples_df["Log2Value"].dropna()) if not samples_df["Log2Value"].dropna().empty else 0.0
        y_limits = (center - 2.2, center + 2.2)

        # Colors per condition
        color_map = {"Ctrl": "#1f77b4", "POI7": "#2ca02c", "POI8": "#d62728"}

        p = (
            ggplot(long_dat, aes("Condition", "Value", color="Condition"))
            + geom_point(position=position_jitter(width=0.1, height=0), alpha=0.8, size=2)
            + stat_summary(
                fun_y=np.mean, geom="point", size=2, color="black"
            )
            + stat_summary(
                fun_data=summarize_mean_sd, geom="errorbar", width=0.2, color="black"
            )
            + scale_color_manual(values=color_map)
            + facet_wrap("~Scale", ncol=2, scales="free_y")
            + labs(title=title, y="", x="")
            + theme_minimal(base_size=12)
            + theme(
                plot_title=element_text(color="red" if highlight else "black", size=11)
            )
        )

        # For the Raw panel ensure non-negative y; for Log2 panel, clamp to ±~2.2 around median
        # We approximate using coord_cartesian for both panels; free_y allows raw to auto-scale from 0.
        # (plotnine facets share coord limits; to enforce raw>=0, set scale_y_continuous with expand and let data >=0)
        p = p + scale_y_continuous(expand=(0, 0.05))
        # Save each page to the appropriate multipage PDF by drawing with matplotlib backend
        fig = p.draw()  # returns a Matplotlib figure
        # Tight layout is already handled; but ensure log2 panel window roughly centered via annotation—skip for simplicity
        (red_pdf if highlight else gray_pdf).savefig(fig, bbox_inches="tight")
        plt.close(fig)

    red_pdf.close()
    gray_pdf.close()

    # Volcano plots
    stats_df7 = pd.DataFrame(stats_list_poi7)
    stats_df8 = pd.DataFrame(stats_list_poi8)
  


    make_volcano_plot(
    stats_df7,
    f"{mode} | POI7 vs Ctrl Volcano",
    os.path.join(output_dir, f"{mode}_POI7_volcano.pdf"),
    annotations=[{"group": "MG",  "dx":0, "dy": 0.08},
                 {"group": "HBMP",  "dx":0.8, "dy": 0.08},
                 {"group": "FAHFA",  "dx":0, "dy": -0.08}]
    )

    make_volcano_plot(
        stats_df8,
        f"{mode} | POI8 vs Ctrl Volcano",
        os.path.join(output_dir, f"{mode}_POI8_volcano.pdf"),
        annotations=[{"group": "DG", "dx": 0, "dy": 0.08},
                    {"group": "MG",  "dx":0, "dy": 0.08},
                    {"group": "HBMP",  "dx":1.5, "dy": -0.08},
                     {"group": "PE-Cer",  "dx":0.4, "dy": 0.08},
                     {"group": "NAGlySer",  "dx":0, "dy": 0.08},
                     {"group": "FAHFA",  "dx":1.8, "dy": -0.08},
                      {"group": "LPC",  "dx":-1.3, "dy": 0.08},
                       {"group": "LPE",  "dx":1.8, "dy": 0.08},
                       {"group": "LPS",  "dx":0.8, "dy": -0.08},
                       {"group": "CL",  "dx":-0.5, "dy": 0.08},
                       {"group": "PG",  "dx":0, "dy": 0.08}
                     ]
    )

    # make_volcano_plot(
    #     stats_df8,
    #     f"{mode} | POI8 vs Ctrl Volcano",
    #     os.path.join(output_dir, f"{mode}_POI8_volcano.pdf"),
    #     annotations=[]
    # )

    



   # --- Bar plot: ranked, heavy axes, big ticks, no frame/legend ---
group_counts = {group_name: len(lipids) for group_name, lipids in grouped_lipids.items()}
gc_df = pd.DataFrame({"Group": list(group_counts.keys()), "n": list(group_counts.values())})

# Sort descending & lock the display order
gc_df = gc_df.sort_values("n", ascending=False).reset_index(drop=True)
gc_df["Group"] = pd.Categorical(gc_df["Group"], categories=gc_df["Group"], ordered=True)

from plotnine import scale_x_discrete, scale_y_continuous

bp = (
    ggplot(gc_df, aes("Group", "n"))
    + geom_col(fill="#3D88C8")
    + labs(y="Number of Species", x="Headgroups")
    + scale_x_discrete(expand=(0.005, 0))  
    # + scale_x_discrete(expand=(0, 0))
    + scale_y_continuous(expand=(0, 0))
    + theme_minimal(base_size=12)
    + theme(
        panel_grid=element_blank(),
        panel_background=element_blank(),
        panel_border=None,
        axis_line=element_line(color="black", size=4),
        axis_ticks_major=element_line(color="black", size=4),
        axis_text_x=element_text(size=21, rotation=45, ha="right", va="top"),
        axis_text_y=element_text(size=25),
        axis_title_x=element_text(size=21, weight="bold",margin={'t': 0}),
        axis_title_y=element_text(size=21, weight="bold",margin={'r': 0}),
        legend_position="none",
    )
)



bp.save(os.path.join(output_dir, f"{mode}_group_counts.pdf"), width=12, height=4.5, verbose=False)



