# -*- coding: utf-8 -*-
import warnings
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from plotnine import *


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

FC15 = np.log2(2)

class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """
    PLS-DA implemented via PLSRegression + one-hot Y.
    - fit(X, y): y are class labels (strings/ints)
    - predict(X): returns class labels
    - predict_proba(X): softmax-like over reconstructed Y-hat rows
    Attributes after fit:
      - pls_: fitted PLSRegression
      - classes_: np.array of class labels in the order used for one-hot
      - x_scores_: (n_samples, n_components) sample scores (LVs)
    """
    def __init__(self, n_components=2, scale=True, with_mean=True, random_state=0):
        self.n_components = n_components
        self.scale = scale
        self.with_mean = with_mean
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        # One-hot encode y in the stable class order
        ohe = OneHotEncoder(categories=[self.classes_], sparse_output=False, handle_unknown="ignore")
        Y = ohe.fit_transform(y.reshape(-1, 1))
        self._ohe = ohe

        steps = []
        steps.append(("imputer", SimpleImputer(strategy="median")))
        if self.scale or self.with_mean:
            steps.append(("scaler", StandardScaler(with_mean=self.with_mean, with_std=self.scale)))
        steps.append(("pls", PLSRegression(n_components=self.n_components, scale=False, copy=True)))
        self.pipe_ = Pipeline(steps)
        self.pipe_.fit(X, Y)

        # Expose scores from underlying PLS
        self.pls_ = self.pipe_.named_steps["pls"]
        # sklearn stores scores on the final fitted estimator, not pipeline
        self.x_scores_ = self.pls_.x_scores_
        return self

    def decision_function(self, X):
        # Continuous Y-hat (one column per class)
        Yhat = self.pipe_.predict(X)
        return Yhat

    def predict(self, X):
        Yhat = self.decision_function(X)
        idx = np.argmax(Yhat, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        # Row-wise normalize positive predictions to sum to 1
        Yhat = np.maximum(self.decision_function(X), 0)
        row_sum = Yhat.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            P = np.divide(Yhat, row_sum, out=np.zeros_like(Yhat), where=row_sum>0)
        return P



def _prepare_xy_from_wide(
    wide_df,
    sample_cols,
    log_transform=True,
    pseudo=1e-12
):
    # features x samples -> transpose to samples x features
    X = wide_df[sample_cols].apply(pd.to_numeric, errors="coerce").copy()
    X = X.dropna(how="all")

    if log_transform:
        X = X.clip(lower=0)
        X = np.log2(X + pseudo)

    Xs = X.T.values  # (n_samples, n_features)
    y = np.array([s.split("_", 1)[0] for s in sample_cols])  # labels from "Ctrl_1" etc.
    return Xs, y


def run_plsda_from_wide(
    wide_df,
    sample_cols,
    n_components=2,
    log_transform=True,
    pseudo=1e-12,
    scale=True,
    with_mean=True,
    cv_folds=5,
    random_state=0
):
    """
    Returns:
      scores_df: [Sample, ConditionLabel, LV1, LV2]
      metrics:   dict with 'cv_accuracy_mean', 'cv_accuracy_std', 'train_accuracy', 'confusion_matrix'
      model:     fitted PLSDAClassifier
    """
    Xs, y = _prepare_xy_from_wide(wide_df, sample_cols, log_transform, pseudo)

    # CV accuracy using the classifier wrapper
    clf = PLSDAClassifier(n_components=n_components, scale=scale, with_mean=with_mean, random_state=random_state)

    def _cv_scorer(estimator, X, y_true):
        y_pred = estimator.predict(X)
        return accuracy_score(y_true, y_pred)

    skf = StratifiedKFold(n_splits=min(cv_folds, len(np.unique(y))), shuffle=True, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_val_score(clf, Xs, y, scoring=_cv_scorer, cv=skf)

    # Fit on full data for plotting/diagnostics
    clf.fit(Xs, y)
    y_pred_train = clf.predict(Xs)
    train_acc = accuracy_score(y, y_pred_train)
    cm = confusion_matrix(y, y_pred_train, labels=condition_order)  # keeps your order

    # Build scores_df for LV1/LV2 plot
    lv = clf.x_scores_[:, :2]
    scores_df = pd.DataFrame(lv, columns=["LV1", "LV2"])
    scores_df["Sample"] = sample_cols
    scores_df["ConditionLabel"] = pd.Categorical(y, categories=condition_order, ordered=True)

    metrics = {
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
        "train_accuracy": float(train_acc),
        "confusion_matrix": cm  # numpy array; print or save as needed
    }
    return scores_df[["Sample", "ConditionLabel", "LV1", "LV2"]], metrics, clf

def make_plsda_plot(scores_df, metrics, filename):
    if scores_df is None or scores_df.empty:
        return

    title_txt = f"PLS-DA (CV acc: {metrics['cv_accuracy_mean']*100:.1f}% ± {metrics['cv_accuracy_std']*100:.1f}%)"

    p = (
        ggplot(scores_df, aes(x="LV1", y="LV2", color="ConditionLabel"))
        + geom_point(size=3.8, alpha=0.9)
        + scale_color_manual(values=condition_colors)
        + labs(title=title_txt, x="Latent Variable 1 (LV1)", y="Latent Variable 2 (LV2)")
        + theme_minimal(base_size=16)
        + theme(
            panel_grid=element_blank(),
            panel_background=element_blank(),
            plot_background=element_blank(),
            legend_position="right",
            axis_line=element_line(color="black", size=1.5),
            axis_ticks_major=element_line(color="black", size=1.5),
            axis_text_x=element_text(size=16),
            axis_text_y=element_text(size=16),
            axis_title_x=element_text(size=18, weight="bold"),
            axis_title_y=element_text(size=18, weight="bold"),
        )
    )
    p.save(filename=filename, width=6.5, height=6.0, verbose=False)

# ============================================================
# Global config & helpers
# ============================================================


y_label_map = {
    "Valine*": "Valine",
}

def map_labels(mapping):
    # plotnine passes a list-like of tick values; return same-length list
    return lambda ticks: [mapping.get(t, t) for t in ticks]

def significant_mask(df, alpha=0.05, include_fc8=False):
    base = (df["log2FC"].abs() > FC15) & (df["pval"] < alpha)
    if include_fc8:
        return base | (df["log2FC"] >= np.log2(8.0))
    return base

# ------------------------------------------------------------
# Amino-acid utilities for box plot
# ------------------------------------------------------------
condition_order  = ["Ctrl", "POI7", "POI8"]
condition_colors = {"Ctrl":"#C1C0C0","POI7":"#010101","POI8":"#EE377A"}

ESSENTIAL_AA = [
    "Phenylalanine","Histidine","Tryptophan","Valine",
    "Alanine","Serine","Threonine","Lysine","Arginine",
    "Isoleucine","Leucine","Methionine"
]


Core_metabolites = [
    "Fructose","Glyceraldehyde 3-phosphate","Xylose 5-phosphate","Ribose 5-phosphate","Cystine*",
    "Uridine*","Adenine", "Glutathione (GSSG)","Glutathione (GSH)", "NAD",
]


def wide_to_tidy_eaa(wide_df, id_col, sample_cols,
                     keep_only_eaa=True, rename_map=None):
    """
    Returns tidy dataframe with columns [Group, ConditionLabel, Value]
    """
    missing = [c for c in sample_cols if c not in wide_df.columns]
    if missing:
        raise ValueError(f"Missing sample columns: {missing}")
    if id_col not in wide_df.columns:
        raise ValueError(f"id_col '{id_col}' not found in dataframe")

    tidy = (
        wide_df[[id_col] + sample_cols]
        .melt(id_vars=id_col, var_name="Sample", value_name="Value")
        .dropna(subset=["Value"])
        .copy()
    )
    tidy["ConditionLabel"] = tidy["Sample"].str.split("_", n=1, expand=True)[0]
    tidy = tidy[tidy["ConditionLabel"].isin(condition_order)].copy()

    tidy["Group"] = tidy[id_col].astype(str)
    if rename_map:
        tidy["Group"] = tidy["Group"].map(lambda x: rename_map.get(x, x))

    if keep_only_eaa:
        tidy = tidy[tidy["Group"].isin(ESSENTIAL_AA)].copy()

    tidy["ConditionLabel"] = pd.Categorical(
        tidy["ConditionLabel"], categories=condition_order, ordered=True
    )
    tidy["Value"] = pd.to_numeric(tidy["Value"], errors="coerce")
    # Ensure strictly positive values for log-scale plotting later
    tidy = tidy.dropna(subset=["Value"])
    tidy = tidy[tidy["Value"] > 0].copy()

    return tidy[["Group", "ConditionLabel", "Value"]]


def wide_to_tidy_core(wide_df, id_col, sample_cols,
                     keep_only_eaa=True, rename_map=None):
    """
    Returns tidy dataframe with columns [Group, ConditionLabel, Value]
    """
    missing = [c for c in sample_cols if c not in wide_df.columns]
    if missing:
        raise ValueError(f"Missing sample columns: {missing}")
    if id_col not in wide_df.columns:
        raise ValueError(f"id_col '{id_col}' not found in dataframe")

    tidy = (
        wide_df[[id_col] + sample_cols]
        .melt(id_vars=id_col, var_name="Sample", value_name="Value")
        .dropna(subset=["Value"])
        .copy()
    )
    tidy["ConditionLabel"] = tidy["Sample"].str.split("_", n=1, expand=True)[0]
    tidy = tidy[tidy["ConditionLabel"].isin(condition_order)].copy()

    tidy["Group"] = tidy[id_col].astype(str)
    if rename_map:
        tidy["Group"] = tidy["Group"].map(lambda x: rename_map.get(x, x))

    if keep_only_eaa:
        tidy = tidy[tidy["Group"].isin(Core_metabolites)].copy()

    tidy["ConditionLabel"] = pd.Categorical(
        tidy["ConditionLabel"], categories=condition_order, ordered=True
    )
    tidy["Value"] = pd.to_numeric(tidy["Value"], errors="coerce")
    # Ensure strictly positive values for log-scale plotting later
    tidy = tidy.dropna(subset=["Value"])
    tidy = tidy[tidy["Value"] > 0].copy()

    return tidy[["Group", "ConditionLabel", "Value"]]
# ============================================================
# Box plot (true log-scale axis; no negative values shown)
# ============================================================
from plotnine import (
    ggplot, aes, geom_boxplot, geom_jitter, labs, theme_minimal, theme,
    element_blank, element_line, element_text, scale_fill_manual, scale_color_manual,
    scale_x_discrete, scale_y_log10
)
from plotnine.positions import position_jitterdodge

def make_eaa_grouped_boxplot_onepanel(
    tidy_eaa_df,
    filename,
    rotation=45,
    control_label="Ctrl",
    dodge_width=0.75
):
    if tidy_eaa_df is None or tidy_eaa_df.empty:
        return

    df = tidy_eaa_df.copy()
    df["Value"] = df["Value"].astype(float)
    # Keep only positive (already ensured upstream, but double-guard)
    df = df[df["Value"] > 0].copy()
    if df.empty:
        return

    # Order amino acids by Control mean (desc) if present, else overall mean
    if control_label in df["ConditionLabel"].unique():
        order_idx = (df.loc[df["ConditionLabel"] == control_label]
                       .groupby("Group")["Value"]
                       .mean()
                       .sort_values(ascending=False)
                       .index.tolist())
    else:
        order_idx = (df.groupby("Group")["Value"]
                       .mean()
                       .sort_values(ascending=False)
                       .index.tolist())
    df["Group"] = pd.Categorical(df["Group"], categories=order_idx, ordered=True)

    # Pretty log10 tick labels like 1e3, 1e4, ...
    def sci_labels(vals):
        out = []
        for v in vals:
            try:
                v = float(v)
                if v <= 0 or not np.isfinite(v):
                    out.append("")
                else:
                    e = int(np.floor(np.log10(v)))
                    m = v / (10**e)
                    # Use 1eN style when mantissa ~1
                    if np.isclose(m, 1.0, atol=1e-6):
                        out.append(f"1e{e}")
                    else:
                        out.append(f"{m:.1f}e{e}")
            except Exception:
                out.append("")
        return out

    p = (
        ggplot(df, aes(x="Group", y="Value"))
        + geom_boxplot(
            aes(fill="ConditionLabel"),
            position=position_dodge(width=dodge_width),
            outlier_alpha=0,
            width=0.7
        )
        + geom_jitter(
            aes(color="ConditionLabel"),
            position=position_jitterdodge(
                jitter_width=0.18, jitter_height=0, dodge_width=dodge_width
            ),
            alpha=0.6, size=1.8
        )
        + scale_y_log10(labels=sci_labels)
        + scale_fill_manual(values=condition_colors)
        + scale_color_manual(values=condition_colors)
        + scale_x_discrete(expand=(0.02, 0.02))
        + labs(title="", x="", y="Intensity (log10 scale)")
        + theme_minimal(base_size=16)
        + theme(
            panel_grid=element_blank(),
            panel_background=element_blank(),
            plot_background=element_blank(),
            legend_position="none",
            axis_line=element_line(color="black", size=1.8),
            axis_ticks_major=element_line(color="black", size=1.8),
            axis_text_x=element_text(size=20, angle=rotation, ha="right"),
            axis_text_y=element_text(size=25),
            axis_title_y=element_text(size=20, weight="bold"),
        )
    )

    # width = 0.5 per amino acid (min 6 so labels don’t clip)
    n_groups = df["Group"].nunique()
    fig_width = max(6, 0.5 * n_groups)
    fig_height = 6.5

    p.save(filename=filename, width=fig_width, height=fig_height, verbose=False)

# ============================================================
# Differential analysis (one-way ANOVA: POI7 vs Ctrl, POI8 vs Ctrl)
# ============================================================
def compute_anova_stats_from_wide(
    wide_df,
    id_col,
    ctrl_cols,
    case_cols,
    small_eps=1e-12
):
    """
    For each metabolite row:
      - FC = mean(case) / mean(ctrl)
      - log2FC = log2(FC)  (with epsilon to avoid division by zero)
      - pval = one-way ANOVA (f_oneway) comparing case vs ctrl
    Returns: DataFrame with ['Group','log2FC','pval']
    """
    cols_needed = [id_col] + ctrl_cols + case_cols
    missing = [c for c in cols_needed if c not in wide_df.columns]
    if missing:
        raise ValueError(f"Missing columns for ANOVA: {missing}")

    sub = wide_df[cols_needed].copy()
    sub = sub.dropna(how="all", subset=ctrl_cols + case_cols)

    records = []
    for _, row in sub.iterrows():
        name = str(row[id_col])

        ctrl_vals = pd.to_numeric(row[ctrl_cols], errors="coerce").dropna().values
        case_vals = pd.to_numeric(row[case_cols], errors="coerce").dropna().values

        # Keep strictly positive where appropriate for FC stability
        ctrl_vals = ctrl_vals[np.isfinite(ctrl_vals)]
        case_vals = case_vals[np.isfinite(case_vals)]
        if ctrl_vals.size == 0 or case_vals.size == 0:
            continue  # skip if one group is empty

        m_ctrl = float(np.mean(ctrl_vals))
        m_case = float(np.mean(case_vals))
        # avoid zero divide; if both zero, skip
        if m_ctrl <= 0 and m_case <= 0:
            continue

        fc = (m_case + small_eps) / (m_ctrl + small_eps)
        log2fc = np.log2(fc)

        try:
            stat, pval = f_oneway(case_vals, ctrl_vals)
            if not np.isfinite(pval):
                continue
        except Exception:
            continue

        records.append((name, log2fc, float(pval)))

    out = pd.DataFrame(records, columns=["Group", "log2FC", "pval"])
    return out

# ============================================================
# Volcano plot (same styling & logic)
# ============================================================
def make_volcano_plot(stats_df, title, filename):
    if stats_df.empty:
        return
    df = stats_df.copy()
    df["nlog10p"] = -np.log10(df["pval"])

    alpha = 0.05
    fc8  = np.log2(8.0)

    df["is_sig"] = significant_mask(df, alpha=alpha, include_fc8=False)
    df["Category"] = np.where(
        df["is_sig"] & (df["log2FC"] > 0), "Significant Up",
        np.where(df["is_sig"] & (df["log2FC"] < 0), "Significant Down", "Not significant")
    )

    p = (
        ggplot(df, aes("log2FC", "nlog10p", color="Category"))
        + geom_point(alpha=0.9, size=3.5)
        + scale_color_manual(values={
            "Significant Up": "#3D88C8",
            "Significant Down": "#212D75",
            "Not significant": "#8E8E8E",
        })
        + geom_hline(yintercept=-np.log10(0.05), linetype="dashed", alpha=0.5)
        + geom_vline(xintercept=[-FC15, FC15], linetype="dashed", alpha=0.5)
        # + geom_vline(xintercept=[fc8], linetype="dotted", alpha=0.7)
        + labs(title='', x="log2(Fold Change)", y="-log10(p-value)", color="")
        + theme_minimal(base_size=14)
        + theme(
            panel_grid=element_blank(),
            panel_background=element_blank(),
            plot_background=element_blank(),
            legend_position="none",
            axis_line=element_line(color="black", size=1.5),
            axis_ticks_major=element_line(color="black", size=1.5),
            axis_text_x=element_text(size=25),
            axis_text_y=element_text(size=25),
            axis_title_x=element_text(size=22, weight="bold"),
            axis_title_y=element_text(size=22, weight="bold"),
        )
        + coord_cartesian(xlim=(-8, 4))
    )
    p.save(filename=filename, width=6, height=6, verbose=False)

# ============================================================
# Lollipop plot (width depends linearly on log2FC span)
# ============================================================
def make_lollipop_plot(
    stats_df,
    title,
    filename,
    pad=0.2,
    base_width=5.0,
    pixels_per_log2_unit=0.9,   # linear factor (width grows linearly with x-range)
    min_width=5.0,
    max_width=14.0,
    height=6.0
):
    if stats_df.empty:
        return

    df = stats_df.copy()
    df = df[significant_mask(df, alpha=0.05, include_fc8=False)].copy()
    if df.empty:
        return

    xmin = float(df["log2FC"].min())
    xmax = float(df["log2FC"].max())
    if np.isclose(xmin, xmax):
        xmin -= 0.5
        xmax += 0.5
    xmin -= pad
    xmax += pad

    # Linear dependence: fig_width = base + factor * (xmax - xmin)
    x_range = max(xmax - xmin, 1e-6)
    fig_width = np.clip(base_width + pixels_per_log2_unit * x_range, min_width, max_width)

    df["absfc"] = df["log2FC"].abs()
    df = df.sort_values("absfc", ascending=False)
    df["Category"] = np.where(df["log2FC"] > 0, "Significant Up", "Significant Down")
    df["name"] = pd.Categorical(df["Group"], categories=df["Group"].tolist(), ordered=True)

    p = (
        ggplot(df, aes(x="log2FC", y="name", color="Category"))
        + geom_segment(aes(x=0, xend="log2FC", y="name", yend="name"), size=1.2)
        + geom_point(size=3.8)
        + geom_vline(xintercept=0, linetype="dashed", alpha=0.6)
        + scale_color_manual(values={
            "Significant Up": "#3D88C8",
            "Significant Down": "#212D75",
        })
        + scale_y_discrete(labels=lambda labs: [str(l).replace("*", "") for l in labs])
        + labs(title='', x="log2(Fold Change)", y="")
        + theme_minimal(base_size=14)
        + theme(
            panel_grid=element_blank(),
            panel_background=element_blank(),
            plot_background=element_blank(),
            legend_position="none",
            axis_line=element_line(color="black", size=3),
            axis_ticks_major=element_line(color="black", size=3),
            axis_text_x=element_text(size=25),
            axis_text_y=element_text(size=22),
            axis_title_x=element_text(size=22, weight="bold"),
        )
        + coord_cartesian(xlim=(xmin, xmax))
    )

    p.save(filename=filename, width=fig_width, height=height, verbose=False)

# ============================================================
# PCA (PC1/PC2) with condition colors and % variance in labels
# ============================================================
def compute_pca_scores_from_wide(
    wide_df,
    sample_cols,
    log_transform=True,
    pseudo=1e-12,
    center=True,
    scale_unit_variance=False
):
    """
    Returns:
      - scores_df: columns [Sample, ConditionLabel, PC1, PC2]
      - evr: explained variance ratios [pc1, pc2]
    """
    # Grab numeric matrix features x samples
    X = wide_df[sample_cols].apply(pd.to_numeric, errors="coerce").copy()

    # Drop features (rows) that are all NaN across samples
    X = X.dropna(how="all")

    # Optional log2 transform
    if log_transform:
        X = X.clip(lower=0)
        X = np.log2(X + pseudo)

    # Transpose to samples x features for sklearn
    Xs = X.T.values  # shape: (n_samples, n_features)

    # Impute per-feature median
    imp = SimpleImputer(strategy="median")
    Xs = imp.fit_transform(Xs)

    # Center (and optionally scale to unit variance) across features
    if center or scale_unit_variance:
        scaler = StandardScaler(with_mean=center, with_std=scale_unit_variance)
        Xs = scaler.fit_transform(Xs)

    # PCA -> first two PCs
    pca = PCA(n_components=2, svd_solver="full", random_state=0)
    scores = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_
    pc1_var = float(evr[0])
    pc2_var = float(evr[1])

    scores_df = pd.DataFrame(scores, columns=["PC1", "PC2"])
    scores_df["Sample"] = sample_cols  # order preserved by X.T
    scores_df["ConditionLabel"] = [s.split("_", 1)[0] for s in sample_cols]
    scores_df["ConditionLabel"] = pd.Categorical(
        scores_df["ConditionLabel"], categories=condition_order, ordered=True
    )
    return scores_df[["Sample", "ConditionLabel", "PC1", "PC2"]], (pc1_var, pc2_var)

def make_pca_plot(scores_df, evr, filename):
    if scores_df is None or scores_df.empty:
        return
    pc1_var, pc2_var = evr
    xlab = f"PC1 ({pc1_var*100:.1f}%)"
    ylab = f"PC2 ({pc2_var*100:.1f}%)"

    p = (
        ggplot(scores_df, aes(x="PC1", y="PC2", color="ConditionLabel"))
        + geom_point(size=3.8, alpha=0.9)
        + scale_color_manual(values=condition_colors)
        + labs(title="", x=xlab, y=ylab)
        + theme_minimal(base_size=16)
        + theme(
            panel_grid=element_blank(),
            panel_background=element_blank(),
            plot_background=element_blank(),
            legend_position="right",
            axis_line=element_line(color="black", size=1.5),
            axis_ticks_major=element_line(color="black", size=1.5),
            axis_text_x=element_text(size=16),
            axis_text_y=element_text(size=16),
            axis_title_x=element_text(size=18, weight="bold"),
            axis_title_y=element_text(size=18, weight="bold"),
        )
    )
    p.save(filename=filename, width=6.5, height=6.0, verbose=False)

# ============================================================
# Main workflow
#   1) Box plot (true log-scale)
#   2) One-way ANOVA: POI7 vs Ctrl, POI8 vs Ctrl
#   3) Volcano plots
#   4) Lollipop plots (width linear in log2FC range)
#   5) PCA (PC1 vs PC2)
# ============================================================
def main():
    # --- I/O paths ---
    # filepath = "/Users/vikash/Documents/Thorey and Martina_Umea/Analysis/plots/changed metabolites.xlsx"
    # output_dir = "/Users/vikash/Documents/Thorey and Martina_Umea/Analysis/metabolomics/newfig/"
    from pathlib import Path

# Resolve repo root no matter where you run from
    ROOT = Path(__file__).resolve().parents[1]

    # Input file now lives in: <repo>/data/changed metabolites.xlsx
    filepath = ROOT / "data" / "changed metabolites.xlsx"

    # Put outputs inside the repo (adjust if you prefer another folder)
    output_dir = ROOT / "metabolomics" / "newfig"

    # Make sure the output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load the wide raw metabolite table ---
    raw_wide = pd.read_excel(filepath)

    # Pick which column has the metabolite names for EAA.
    id_col = "Identification" if "Identification" in raw_wide.columns else "Processed Metabolite"
    rename_map = {"Valine*": "Valine"}

    # Your sample columns
    ctrl_cols = ['Ctrl_1','Ctrl_2','Ctrl_3','Ctrl_4']
    poi7_cols = ['POI7_1','POI7_2','POI7_3','POI7_4']
    poi8_cols = ['POI8_1','POI8_2','POI8_3','POI8_4']
    sample_cols = ctrl_cols + poi7_cols + poi8_cols

    # -----------------------------
    # 1) Box plot (true log-scale axis)
    # -----------------------------
    tidy_eaa = wide_to_tidy_eaa(
        wide_df=raw_wide,
        id_col=id_col,
        sample_cols=sample_cols,
        keep_only_eaa=True,
        rename_map=rename_map
    )
    make_eaa_grouped_boxplot_onepanel(
        tidy_eaa_df=tidy_eaa,
        filename=f"{output_dir}eaa_boxplot.pdf",
        rotation=45
    )

    tidy_core = wide_to_tidy_core(
        wide_df=raw_wide,
        id_col=id_col,
        sample_cols=sample_cols,
        keep_only_eaa=True,
        rename_map=rename_map
    )

    make_eaa_grouped_boxplot_onepanel(
        tidy_eaa_df=tidy_core,
        filename=f"{output_dir}core_boxplot.pdf",
        rotation=45
    )

    # -----------------------------
    # 2) Differential analysis
    #    One-way ANOVA: POI7 vs Ctrl, POI8 vs Ctrl
    # -----------------------------
    stats_poi7 = compute_anova_stats_from_wide(
        wide_df=raw_wide,
        id_col=id_col,
        ctrl_cols=ctrl_cols,
        case_cols=poi7_cols
    )
    stats_poi8 = compute_anova_stats_from_wide(
        wide_df=raw_wide,
        id_col=id_col,
        ctrl_cols=ctrl_cols,
        case_cols=poi8_cols
    )

    # -----------------------------
    # 3) Volcano plots
    # -----------------------------
    make_volcano_plot(
        stats_poi7,
        title="Volcano: Ctrl vs POI7",
        filename=f"{output_dir}volcano_ctrl_vs_poi7_fixed.pdf"
    )
    make_volcano_plot(
        stats_poi8,
        title="Volcano: Ctrl vs POI8",
        filename=f"{output_dir}volcano_ctrl_vs_poi8_fixed.pdf"
    )

    # -----------------------------
    # 4) Lollipop plots (width ~ linear in log2FC span)
    # -----------------------------
    make_lollipop_plot(
        stats_poi7,
        "Top Changes: Ctrl vs POI7",
        f"{output_dir}lollipop_ctrl_vs_poi7.pdf"
    )
    make_lollipop_plot(
        stats_poi8,
        "Top Changes: Ctrl vs POI8",
        f"{output_dir}lollipop_ctrl_vs_poi8.pdf"
    )

    # -----------------------------
    # 5) PCA (PC1 vs PC2) with condition colors + variance in labels
    # -----------------------------
    pca_scores, evr = compute_pca_scores_from_wide(
        wide_df=raw_wide,
        sample_cols=sample_cols,
        log_transform=True,          # standard for metabolomics
        pseudo=1e-9,
        center=True,
        scale_unit_variance=False    # toggle True if you prefer autoscaling
    )
    make_pca_plot(
        pca_scores,
        evr,
        filename=f"{output_dir}pca_pc1_pc2.pdf"
    )

        # -----------------------------
    # 6) PLS-DA (LV1 vs LV2)
    # -----------------------------
    pls_scores, pls_metrics, pls_model = run_plsda_from_wide(
        wide_df=raw_wide,
        sample_cols=sample_cols,
        n_components=2,          # increase to 3+ if you plan more LVs
        log_transform=True,
        pseudo=1e-9,
        scale=True,
        with_mean=True,
        cv_folds=5,
        random_state=0
    )
    make_plsda_plot(
        scores_df=pls_scores,
        metrics=pls_metrics,
        filename=f"{output_dir}plsda_lv1_lv2.pdf"
    )

    # (Optional) Print quick diagnostics
    print("PLS-DA CV accuracy: "
          f"{pls_metrics['cv_accuracy_mean']*100:.1f}% ± {pls_metrics['cv_accuracy_std']*100:.1f}%")
    print("PLS-DA train accuracy: "
          f"{pls_metrics['train_accuracy']*100:.1f}%")
    print("Confusion matrix (rows=true, cols=pred) in order:", condition_order)
    print(pls_metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
