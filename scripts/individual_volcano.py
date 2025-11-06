# individual_volcano.py
import os
import re
import sys
import pandas as pd
import numpy as np
from scipy.stats import f_oneway

# plotting
from plotnine import (
    ggplot, aes, geom_point, geom_text, labs, theme_minimal, theme, element_text,
    scale_color_manual, geom_hline, geom_vline, coord_cartesian,
    element_blank, element_line
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist


from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repo root

file_path = ROOT / "data" / "Final_Combined_Lipidomics.txt"
output_dir = ROOT / "headgroup_species_plots"


# -------------------------
# CONFIG
# -------------------------
# file_path = "/Users/vikash/Documents/Thorey and Martina_Umea/Analysis/Lipid_scripts/thoreylipid/Final_Combined_Lipidomics.txt"
# output_dir = "headgroup_species_plots"        # outputs live here

sample_cols = [
    'Ctrl_1', 'Ctrl_2', 'Ctrl_3', 'Ctrl_4',
    'POI7_1', 'POI7_2', 'POI7_3', 'POI7_4',
    'POI8_1', 'POI8_2', 'POI8_3', 'POI8_4'
]
CTRL_PREFIX = "Ctrl"

# Volcano thresholds
LOG2_FC_CUTOFF = np.log2(3/2)
ALPHA = 0.05
ANNOTATE_P = 0.05
EPS = 1e-9

# Clustermap transform
#   "raw"   -> normalized values (each sample column sums to 1 across species)
#   "log10" -> log10(value) with zeros masked to NaN (not drawn)
CLUSTERMAP_MODE = "log10"

# Shared colors
CAT_COLORS = {
    "Significant Up":   "#3D88C8",
    "Significant Down": "#212D75",
    "Not significant":  "#8E8E8E",
}

# -------------------------
# Utils
# -------------------------
def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", str(name))

def _to_1d_numeric(arr_like) -> np.ndarray:
    if isinstance(arr_like, pd.DataFrame):
        vec = arr_like.astype(float).sum(axis=0).to_numpy(dtype=float)
    elif isinstance(arr_like, pd.Series):
        vec = arr_like.to_numpy(dtype=float)
    else:
        vec = np.asarray(arr_like, dtype=float)
    vec = np.ravel(vec).astype(float)
    vec = vec[~np.isnan(vec)]
    return vec

def calc_stats(ctrl, poi):
    ctrl_vec = _to_1d_numeric(ctrl)
    poi_vec  = _to_1d_numeric(poi)
    if ctrl_vec.size == 0 or poi_vec.size == 0:
        return None, None
    fc = np.log2(np.median(poi_vec) / np.median(ctrl_vec) + EPS)
    pval = float(f_oneway(ctrl_vec, poi_vec)[1])
    return fc, pval

def detect_headgroup_column(df: pd.DataFrame) -> str:
    if "Headgroup" in df.columns:
        return "Headgroup"
    inferred = df["Metabolite name"].str.extract(r"^([A-Za-z0-9\-]+)", expand=False)
    return "Headgroup_inferred" if inferred.notna().any() else None

def make_species_volcano(df_cond, headgroup, condition_label, out_path):
    if df_cond.empty:
        return
    d = df_cond.copy()
    d["nlog10p"] = -np.log10(d["pval"].clip(lower=1e-300))
    d["is_sig"] = (np.abs(d["log2FC"]) > LOG2_FC_CUTOFF) & (d["pval"] < ALPHA)
    d["Category"] = np.where(
        d["is_sig"] & (d["log2FC"] > 0), "Significant Up",
        np.where(d["is_sig"] & (d["log2FC"] < 0), "Significant Down", "Not significant")
    )
    d_lab = d[d["pval"] < ANNOTATE_P].copy()
    if not d_lab.empty:
        d_lab["label_x"] = np.where(d_lab["log2FC"] >= 0, d_lab["log2FC"] + 0.05, d_lab["log2FC"] - 0.05)
        d_lab["label_y"] = d_lab["nlog10p"] + 0.05

    p = (
        ggplot(d, aes("log2FC", "nlog10p", color="Category"))
        + geom_point(alpha=0.9, size=3)
        + scale_color_manual(values=CAT_COLORS)
        + geom_hline(yintercept=-np.log10(ALPHA), linetype="dashed", alpha=0.25)
        + geom_vline(xintercept=[-LOG2_FC_CUTOFF, LOG2_FC_CUTOFF], linetype="dashed", alpha=0.25)
        + labs(title=f"{headgroup} | {condition_label} vs Ctrl (species-level)",
               x="log2 Fold Change", y="-log10(p-value)", color="")
        + theme_minimal(base_size=12)
        + theme(
            panel_grid=element_blank(),
            panel_background=element_blank(),
            axis_line=element_line(color="black"),
            axis_ticks_major=element_line(color="black"),
            axis_text_x=element_text(size=12),
            axis_text_y=element_text(size=12),
            axis_title_x=element_text(size=13, weight="bold"),
            axis_title_y=element_text(size=13, weight="bold"),
            legend_position="none",
            plot_title=element_text(size=14, weight="bold")
        )
        + coord_cartesian(xlim=(-4, 6))
    )
    if not d_lab.empty:
        p = p + geom_text(aes(x="label_x", y="label_y", label="Species", color="Category"),
                          data=d_lab, size=7)
    ensure_parent_dir(out_path)
    p.save(out_path, width=6, height=6, verbose=False)

def adaptive_fig_height(n_species: int) -> float:
    # base=8 inches, +0.15 inch per row, capped at 120"
    return float(np.clip(8 + 0.15 * n_species, 8, 120))

# -------------------------
# I/O + Data prep
# -------------------------
os.makedirs(output_dir, exist_ok=True)
volcano_dir = os.path.join(output_dir, "volcano_per_headgroup")
heatmap_dir = os.path.join(output_dir, "clustermaps_per_headgroup")
os.makedirs(volcano_dir, exist_ok=True)
os.makedirs(heatmap_dir, exist_ok=True)

df = pd.read_csv(file_path, sep="\t")
df.columns = df.columns.str.strip()
df = df[df["Metabolite name"].notna()].copy()
df["Species"] = df["Metabolite name"].astype(str)

hg_col = detect_headgroup_column(df)
if hg_col is None:
    print("ERROR: No Headgroup column and could not infer headgroup from names.", file=sys.stderr)
    sys.exit(1)
if hg_col not in df.columns:
    df[hg_col] = df["Metabolite name"].str.extract(r"^([A-Za-z0-9\-]+)", expand=False)

keep_cols = ["Species", "Metabolite name", hg_col] + sample_cols
missing = [c for c in keep_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

collapsed = (
    df[keep_cols]
    .groupby(["Species", "Metabolite name", hg_col], as_index=False)[sample_cols]
    .sum()
    .copy()
)
collapsed.rename(columns={hg_col: "Group"}, inplace=True)

# Normalize each sample to sum = 1
col_sums = collapsed[sample_cols].sum(axis=0).replace(0, np.nan)
norm = collapsed.copy()
for c in sample_cols:
    norm[c] = norm[c] / col_sums[c]

check_sums = norm[sample_cols].sum(axis=0)
print("🔹 Column sums after normalization (should be 1.0):")
for c in sample_cols:
    print(f"   {c}: {check_sums[c]:.6f}")

headgroups = sorted(norm["Group"].dropna().unique().tolist())
print(f"\n🔹 Total headgroups detected: {len(headgroups)}")

# -------------------------
# Stats (for volcano + significance categories)
# -------------------------
def _vec_from_prefix(row_like: pd.Series, prefix: str) -> np.ndarray:
    cols = [c for c in sample_cols if c.startswith(prefix)]
    return row_like[cols].astype(float).to_numpy(dtype=float)

records_7, records_8 = [], []
for _, row in norm.iterrows():
    species = row["Species"]
    group   = row["Group"]
    ctrl_vals = _vec_from_prefix(row, CTRL_PREFIX)
    poi7_vals = _vec_from_prefix(row, "POI7")
    poi8_vals = _vec_from_prefix(row, "POI8")
    fc7, p7 = calc_stats(ctrl_vals, poi7_vals)
    fc8, p8 = calc_stats(ctrl_vals, poi8_vals)
    if (fc7 is not None) and (p7 is not None):
        records_7.append({"Species": species, "Group": group, "log2FC": fc7, "pval": p7})
    if (fc8 is not None) and (p8 is not None):
        records_8.append({"Species": species, "Group": group, "log2FC": fc8, "pval": p8})

stats_df7 = pd.DataFrame(records_7)
stats_df8 = pd.DataFrame(records_8)

def _fc_lookup(stats_df: pd.DataFrame):
    if stats_df.empty:
        return {}
    return dict(zip(stats_df["Species"], stats_df["log2FC"]))

fc7_map = _fc_lookup(stats_df7)
fc8_map = _fc_lookup(stats_df8)

def species_category_map(stats_df: pd.DataFrame):
    if stats_df.empty:
        return {}
    d = stats_df.copy()
    d["is_sig"] = (np.abs(d["log2FC"]) > LOG2_FC_CUTOFF) & (d["pval"] < ALPHA)
    d["Category"] = np.where(
        d["is_sig"] & (d["log2FC"] > 0), "Significant Up",
        np.where(d["is_sig"] & (d["log2FC"] < 0), "Significant Down", "Not significant")
    )
    return dict(zip(d["Species"], d["Category"]))

cat_map_p7 = species_category_map(stats_df7)
cat_map_p8 = species_category_map(stats_df8)

# -------------------------
# Volcano plots
# -------------------------
for grp in headgroups:
    d7 = stats_df7[stats_df7["Group"] == grp].copy()
    d8 = stats_df8[stats_df8["Group"] == grp].copy()
    if not d7.empty:
        out7 = os.path.join(volcano_dir, f"{sanitize_filename(grp)}_POI7_volcano.pdf")
        make_species_volcano(d7, grp, "POI7", out7)
    if not d8.empty:
        out8 = os.path.join(volcano_dir, f"{sanitize_filename(grp)}_POI8_volcano.pdf")
        make_species_volcano(d8, grp, "POI8", out8)

# -------------------------
# Heatmap prep
# -------------------------
def transform_for_heatmap(df_sub: pd.DataFrame) -> pd.DataFrame:
    mat = df_sub.set_index("Species")[sample_cols].astype(float)
    if CLUSTERMAP_MODE.lower() == "raw":
        return mat
    elif CLUSTERMAP_MODE.lower() == "log10":
        mat = mat.where(mat > 0, np.nan)
        mat = np.log10(mat)
        return mat
    else:
        raise ValueError("CLUSTERMAP_MODE must be 'raw' or 'log10'")

def build_row_colors(index_species: pd.Index) -> pd.DataFrame:
    rows = []
    for sp in index_species:
        cat7 = cat_map_p7.get(sp, "Not significant")
        cat8 = cat_map_p8.get(sp, "Not significant")
        rows.append([CAT_COLORS[cat7], CAT_COLORS[cat8]])
    rc = pd.DataFrame(rows, columns=["POI7", "POI8"], index=index_species)
    return rc

# helper: find x-index for 'POI7'/'POI8' in SHOWN columns
def _find_group_x(cols_shown, group_name):
    """
    Prefer exact 'POI7'/'POI8' columns if present.
    Else first column starting with that prefix (e.g., POI7_1).
    Else fallback to column index 0 or 1 respectively.
    """
    # exact match
    try:
        return cols_shown.index(group_name)
    except ValueError:
        pass
    # prefix match
    for i, c in enumerate(cols_shown):
        if str(c).startswith(group_name):
            return i
    # fallback
    return 0 if group_name == "POI7" else 1

# -------------------------
# Save clustermap with in-cell FC annotations (significant only)
# -------------------------
def save_clustermap(matrix: pd.DataFrame, title: str, out_path: str):
    """
    Draw clustermap and annotate log2FC values *inside* heatmap cells:
      - Only annotate significant species (per cat_map_p7 / cat_map_p8).
      - FC text is placed in the 'POI7' and 'POI8' columns if present,
        otherwise in the first column starting with those prefixes (e.g., POI7_1/POI8_1),
        else fallback to x=0/x=1.
      - Row color bars (POI7/POI8) are preserved.
    """
    if matrix.empty:
        return

    # keep NaNs; drop all-NaN rows
    mat_disp = matrix.dropna(how='all', axis=0)
    if mat_disp.empty:
        print(f"   (skip {title}: all rows NaN after transform)")
        return

    n_rows = mat_disp.shape[0]
    height = adaptive_fig_height(n_rows)

    # row colors aligned to display index
    row_colors = build_row_colors(mat_disp.index)

    # --- small heatmap if <2 rows (no clustering)
    if n_rows < 2:
        plt.figure(figsize=(10, height))
        ax = sns.heatmap(mat_disp, cmap="viridis", yticklabels=True, xticklabels=True)
        ax.set_title(title, fontsize=12, pad=16)

        # figure out shown columns from tick labels
        cols_shown = [t.get_text() for t in ax.get_xticklabels()]
        poi7_x = _find_group_x(cols_shown, "POI7")
        poi8_x = _find_group_x(cols_shown, "POI8")

        # annotate
        species_order = [t.get_text() for t in ax.get_yticklabels()]
        for y, sp in enumerate(species_order):
            if cat_map_p7.get(sp, "Not significant") != "Not significant":
                fc7 = fc7_map.get(sp, np.nan)
                if pd.notna(fc7):
                    ax.text(poi7_x + 0.5, y + 0.5, f"{fc7:.2f}",
                            ha="center", va="center", fontsize=7, fontweight="bold")
            if cat_map_p8.get(sp, "Not significant") != "Not significant":
                fc8 = fc8_map.get(sp, np.nan)
                if pd.notna(fc8):
                    ax.text(poi8_x + 0.5, y + 0.5, f"{fc8:.2f}",
                            ha="center", va="center", fontsize=7, fontweight="bold")

        ensure_parent_dir(out_path)
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return

    # build a filled copy ONLY for distance computation
    col_medians = np.nanmedian(mat_disp.values, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    mat_fill = mat_disp.copy()
    for j, col in enumerate(mat_fill.columns):
        mat_fill[col] = mat_fill[col].fillna(col_medians[j])

    try:
        dists = pdist(mat_fill.values, metric="euclidean")
        if dists.size == 0:
            raise ValueError("Empty distance vector")
        row_link = sch.linkage(dists, method="average")

        cg = sns.clustermap(
            mat_disp,
            row_linkage=row_link,
            col_cluster=False,  # keep column order
            cmap="viridis",
            linewidths=0,
            figsize=(10, height),
            xticklabels=True,
            yticklabels=True,
            row_colors=row_colors
        )
        plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha="right")
        cg.ax_heatmap.set_title(title, fontsize=12, pad=16)

        # columns as actually displayed by seaborn
        cols_shown = list(cg.data2d.columns)
        # poi7_x = _find_group_x(cols_shown, "POI7")
        # poi8_x = _find_group_x(cols_shown, "POI8")
        poi7_x = -1
        poi8_x = 0
      
        # annotate FC inside the heatmap
        ax = cg.ax_heatmap
        species_order = [t.get_text() for t in ax.get_yticklabels()]
        for y, sp in enumerate(species_order):
            if cat_map_p7.get(sp, "Not significant") != "Not significant":
                fc7 = fc7_map.get(sp, np.nan)
                if pd.notna(fc7):
                    ax.text(poi7_x + 0.25, y + 0.5, f"{fc7:.1f}",
                            ha="center", va="center", fontsize=7, fontweight="bold",color="white")
            if cat_map_p8.get(sp, "Not significant") != "Not significant":
                fc8 = fc8_map.get(sp, np.nan)
                if pd.notna(fc8):
                    ax.text(poi8_x - 0.25, y + 0.5, f"{fc8:.1f}",
                            ha="center", va="center", fontsize=7, fontweight="bold",color="white")

        ensure_parent_dir(out_path)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"   (fallback heatmap for '{title}' due to clustering error: {e})")
        plt.figure(figsize=(10, height))
        ax = sns.heatmap(mat_disp, cmap="viridis", yticklabels=True, xticklabels=True)
        ax.set_title(title, fontsize=12, pad=16)

        cols_shown = [t.get_text() for t in ax.get_xticklabels()]
        poi7_x = _find_group_x(cols_shown, "POI7")
        poi8_x = _find_group_x(cols_shown, "POI8")

        species_order = [t.get_text() for t in ax.get_yticklabels()]
        for y, sp in enumerate(species_order):
            if cat_map_p7.get(sp, "Not significant") != "Not significant":
                fc7 = fc7_map.get(sp, np.nan)
                if pd.notna(fc7):
                    ax.text(poi7_x + 0.5, y + 0.5, f"{fc7:.2f}",
                            ha="center", va="center", fontsize=7, fontweight="bold")
            if cat_map_p8.get(sp, "Not significant") != "Not significant":
                fc8 = fc8_map.get(sp, np.nan)
                if pd.notna(fc8):
                    ax.text(poi8_x + 0.5, y + 0.5, f"{fc8:.2f}",
                            ha="center", va="center", fontsize=7, fontweight="bold")

        ensure_parent_dir(out_path)
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

# -------------------------
# Run per headgroup
# -------------------------
# Volcanoes
for grp in headgroups:
    d7 = stats_df7[stats_df7["Group"] == grp].copy()
    d8 = stats_df8[stats_df8["Group"] == grp].copy()
    if not d7.empty:
        out7 = os.path.join(volcano_dir, f"{sanitize_filename(grp)}_POI7_volcano.pdf")
        make_species_volcano(d7, grp, "POI7", out7)
    if not d8.empty:
        out8 = os.path.join(volcano_dir, f"{sanitize_filename(grp)}_POI8_volcano.pdf")
        make_species_volcano(d8, grp, "POI8", out8)

# Clustermaps with row color bars + in-cell FC annotations
for grp in headgroups:
    sub = norm[norm["Group"] == grp].copy()
    if sub.empty:
        continue
    mat = transform_for_heatmap(sub)
    ttl = f"{grp} | normalized intensities ({CLUSTERMAP_MODE})"
    out = os.path.join(heatmap_dir, f"{sanitize_filename(grp)}_clustermap_{CLUSTERMAP_MODE}.pdf")
    save_clustermap(mat, ttl, out)

print("\n✅ Done. Clustermaps include POI7/POI8 row color bars and in-cell FC labels at 'POI7'/'POI8' (or their first sample columns) for significant species.")
