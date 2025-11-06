# app.py
import re
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# =========================================================
# Config
# =========================================================
file_path = "Final_Combined_Lipidomics.txt"

sample_cols = [
    'Ctrl_1','Ctrl_2','Ctrl_3','Ctrl_4',
    'POI7_1','POI7_2','POI7_3','POI7_4',
    'POI8_1','POI8_2','POI8_3','POI8_4'
]
condition_order  = ["Ctrl","POI7","POI8"]
condition_labels = {"Ctrl":"Control (mCherry_Luc)","POI7":"RhoSH KO (POI7)","POI8":"MAP1 KO (POI8)"}
condition_colors = {"Ctrl":"#C1C0C0","POI7":"#010101","POI8":"#EE377A"}

headgroup_full_names = {
    "PA": "Phosphatidic Acid",
    "PC": "Phosphatidylcholine",
    "PE": "Phosphatidylethanolamine",
    "PG": "Phosphatidylglycerol",
    "PI": "Phosphatidylinositol",
    "PS": "Phosphatidylserine",
    "CL": "Cardiolipin",
    "MLCL": "Monolysocardiolipin",
    "HBMP": "Hydroxybutyl Monophosphate",
    "LPC": "Lysophosphatidylcholine",
    "LPE": "Lysophosphatidylethanolamine",
    "LPI": "Lysophosphatidylinositol",
    "LPS": "Lysophosphatidylserine",
    "DG": "Diglyceride",
    "TG": "Triglyceride",
    "MG": "Monoglyceride",
    "Dioctyl": "Dioctylglycerol",
    "Cer": "Ceramide",
    "HexCer": "Hexosylceramide",
    "Hex2Cer": "Dihexosylceramide",
    "SM": "Sphingomyelin",
    "SPB": "Sphingoid Base",
    "FA": "Fatty Acid",
    "FAHFA": "Fatty Acid Ester of Hydroxy Fatty Acid",
    "CAR": "Acylcarnitine",
    "Adenosine": "Nucleoside-derived Lipid",
    "ST": "Sterol",
    "NAGly": "N-Arachidonoyl Glycine",
    "NAGlySer": "N-Arachidonoyl Glycylserine",
    "SL": "Sulfolipid"
}

EPS = 1e-9

# =========================================================
# Data loading & parsing
# =========================================================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()
    df = df[df["Metabolite name"].notna()].copy()
    if df["Metabolite name"].duplicated().any():
        keep = ["Metabolite name"] + [c for c in sample_cols if c in df.columns]
        df = df[keep].groupby("Metabolite name", as_index=False, sort=False).sum()
    keep_cols = ["Metabolite name"] + [c for c in sample_cols if c in df.columns]
    df = df[keep_cols].copy()
    return df

HEADGROUP_PATTERN = r"^[A-Za-z0-9]+"
CARBON_PATTERN    = r"(\d+):(\d+)"

def parse_headgroup(name: str) -> str:
    m = re.match(HEADGROUP_PATTERN, str(name).strip())
    return m.group(0) if m else None

def parse_carbon(name: str) -> str:
    m = re.search(CARBON_PATTERN, str(name))
    return f"{m.group(1)}:{m.group(2)}" if m else None

def to_long(df: pd.DataFrame) -> pd.DataFrame:
    m = df.melt(id_vars=["Metabolite name"], value_vars=sample_cols,
                var_name="Sample", value_name="Value")
    m["Condition"] = m["Sample"].str.split("_").str[0]
    m["Headgroup"] = m["Metabolite name"].apply(parse_headgroup)
    m["Carbon"]    = m["Metabolite name"].apply(parse_carbon)
    m = m[m["Condition"].isin(condition_order)].copy()
    m["Value"] = pd.to_numeric(m["Value"], errors="coerce").fillna(0.0)
    m = m.dropna(subset=["Headgroup"])
    return m

# =========================================================
# Aggregations
# =========================================================
def headgroup_fractions(long_df: pd.DataFrame) -> pd.DataFrame:
    """Sum by headgroup within each sample; normalize so per-sample sum across headgroups == 1."""
    gsum = (long_df.groupby(["Sample","Condition","Headgroup"], as_index=False)["Value"]
            .sum().rename(columns={"Value":"Sum"}))
    totals = gsum.groupby("Sample", as_index=False)["Sum"].sum().rename(columns={"Sum":"Total"})
    gsum = gsum.merge(totals, on="Sample", how="left")
    gsum["Fraction"] = np.where(gsum["Total"] > 0, gsum["Sum"] / gsum["Total"], 0.0)
    return gsum[["Sample","Condition","Headgroup","Fraction"]]

def species_values_for_headgroup(long_df: pd.DataFrame, headgroup: str) -> pd.DataFrame:
    """Within headgroup, aggregate species defined only by carbon (e.g., 34:1)."""
    sub = long_df[(long_df["Headgroup"] == headgroup)].copy()
    sub = sub.dropna(subset=["Carbon"]).copy()
    sp = (sub.groupby(["Sample","Condition","Carbon"], as_index=False)["Value"]
          .sum().rename(columns={"Carbon":"Species"}))
    return sp[["Sample","Condition","Species","Value"]]

# -------- sorting helpers for species like "34:1" → numeric sort --------
def species_sort_key(s: str) -> tuple:
    if isinstance(s, str):
        m = re.match(r"(\d+):(\d+)", s)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    return (10**9, 10**9)  # push unknowns to end

# =========================================================
# Plot helpers
# =========================================================
def compute_figure_size(n_groups: int, orientation_xy: bool) -> tuple[int, int]:
    """Return (width_px, height_px) scaled for readability."""
    if orientation_xy:
        width = max(1100, 160 + int(n_groups * 90))
        height = 600
    else:
        width  = 1100
        height = max(600, 220 + n_groups * 28)
    return width, height

def make_grouped_box_figure(
    df_grp: pd.DataFrame,
    value_col: str,
    group_col: str,
    title: str,
    swap_axes: bool = False,
    scale: str = "linear"  # "linear" | "log10"
) -> tuple[go.Figure, list]:
    """
    Draw grouped boxes: one box per (group, condition) side-by-side.
    Fix overlap by enforcing shared category order, categorical dtype per trace,
    and unique offsetgroup per condition.
    Returns (figure, ordered_groups_used)
    """
    assert "Condition" in df_grp.columns, "Expected a 'Condition' column."

    # --- 0) normalize group labels to strings (critical for category matching) ---
    df_grp = df_grp.copy()
    df_grp[group_col] = df_grp[group_col].astype(str)

    groups = [g for g in df_grp[group_col].dropna().unique().tolist()]
    if not groups:
        return empty_fig("No data"), []

    # --- 1) Shared category order for the group axis ---
    if group_col.lower() == "species":
        # Numeric sort for "34:1" style labels
        ordered_groups = sorted(groups, key=species_sort_key)
    else:
        ordered_groups = sorted(groups)  # alpha

    # --- 2) Log scale handling ---
    is_log = (scale == "log10")
    if is_log:
        df_grp[value_col] = np.where(df_grp[value_col] > 0, df_grp[value_col], EPS)

    # --- 3) Dynamic canvas ---
    width_px, height_px = compute_figure_size(len(ordered_groups), orientation_xy=(not swap_axes))
    fig = go.Figure()

    # --- 4) Apply category array BEFORE adding traces ---
    if not swap_axes:
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=ordered_groups)
    else:
        fig.update_yaxes(type="category", categoryorder="array", categoryarray=ordered_groups)

    # --- 5) One trace per condition; enforce same categorical dtype & unique offsetgroup ---
    for cond in condition_order:
        d = df_grp[df_grp["Condition"] == cond].copy()
        if d.empty:
            continue

        # Force same categorical order for this trace
        d[group_col] = pd.Categorical(d[group_col].astype(str), categories=ordered_groups, ordered=True)

        common_kwargs = dict(
            name=condition_labels[cond],
            marker_color=condition_colors[cond],
            boxmean=False,
            opacity=0.95,
            boxpoints="all",     # show points
            jitter=0.5,          # jitter those points
            pointpos=0.0,        # centered
            whiskerwidth=0.8,
            notched=False,
            offsetgroup=cond,    # <- distinct per condition (prevents overlap)
            alignmentgroup="grp",# <- same across conditions (aligns widths)
            legendgroup=cond,
            showlegend=True
        )

        if not swap_axes:
            fig.add_box(x=d[group_col], y=d[value_col], **common_kwargs)
        else:
            fig.add_box(x=d[value_col], y=d[group_col], orientation="h", **common_kwargs)

    axis_label_val = (
        "Fraction (sums to 1 within each sample)"
        if value_col.lower().startswith("fraction")
        else ("Value (log10)" if is_log else "Value")
    )

    if not swap_axes:
        fig.update_xaxes(title=group_col, tickangle=0, automargin=True)
        fig.update_yaxes(title=axis_label_val, type=("log" if is_log else "linear"), rangemode="tozero")
    else:
        fig.update_xaxes(title=axis_label_val, type=("log" if is_log else "linear"), rangemode="tozero")
        fig.update_yaxes(title=group_col, automargin=True)

    # --- 6) Grouped layout ---
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", y=0.98, yanchor="top", pad=dict(b=20)),
        template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=90, r=60, t=96, b=96),
        width=width_px,
        height=height_px,
        boxmode="group",   # side-by-side boxes
        boxgap=0.18,       # gap between condition boxes within a group
        boxgroupgap=0.06   # gap between groups
    )
    return fig, ordered_groups


def empty_fig(title="No Data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="simple_white",
        title=dict(text=title, x=0.01, xanchor="left", y=0.98, yanchor="top", pad=dict(b=20)),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[{
            "text": title, "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5, "showarrow": False, "font": {"size": 16}
        }],
        margin=dict(l=60, r=60, t=96, b=78),
        width=1100, height=450
    )
    return fig

# =========================================================
# App
# =========================================================
df_raw  = load_data(file_path)
df_long = to_long(df_raw)
ALL_HEADGROUPS = sorted(df_long["Headgroup"].dropna().unique().tolist())

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = "Lipidomics — Headgroup Fractions & Carbon Species"

controls_card = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Label("Axis orientation", className="mb-1"),
                dbc.RadioItems(
                    id="axis-orientation",
                    options=[
                        {"label":"x = Group, y = Value", "value":"xy"},
                        {"label":"x = Value, y = Group", "value":"yx"},
                    ],
                    value="xy",
                    inline=True,
                ),
            ], md=4, sm=12, className="mb-3"),
            dbc.Col([
                dbc.Label("Scale", className="mb-1"),
                dbc.RadioItems(
                    id="axis-scale",
                    options=[{"label":"Linear", "value":"linear"},
                             {"label":"Log10","value":"log10"}],
                    value="linear",
                    inline=True,
                ),
            ], md=3, sm=12, className="mb-3"),
            dbc.Col([
                dbc.Label("Headgroup (default: All)", className="mb-1"),
                dcc.Dropdown(
                    id="headgroup-select",
                    options=[{"label":"All (headgroup fractions)", "value":"__ALL__"}] +
                            [{"label":hg, "value":hg} for hg in ALL_HEADGROUPS],
                    value="__ALL__",
                    clearable=False,
                ),
            ], md=3, sm=12, className="mb-3"),
            dbc.Col([
                dbc.Label("Species (carbon only)", className="mb-1"),
                dcc.Dropdown(
                    id="species-select",
                    clearable=True,
                    placeholder="Pick a headgroup first",
                ),
            ], md=2, sm=12, className="mb-3"),
        ])
    ]),
    className="shadow-sm"
)

app.layout = dbc.Container([
    html.H2(
        "Lipidomics Explorer — Headgroup Fractions (sum=1)",
        className="mt-3 mb-2"
    ),
    html.Div(
        "Tip: The plot groups the three conditions per species/headgroup. Use the controls to adjust axes and scale.",
        className="text-muted mb-4"
    ),

    controls_card,
    html.Div(className="mb-3"),  # gap before figure

    dbc.Card(
        dbc.CardBody([
            html.Div(
                dcc.Loading(
                    dcc.Graph(
                        id="main-plot",
                        config={"displayModeBar": True, "responsive": True}
                    ),
                    type="circle"
                ),
                style={"overflowX": "auto", "overflowY": "hidden", "paddingBottom": "8px"},
                className="w-100"
            ),
            html.Div(id="helper-text", className="text-muted mt-2"),
            html.Hr(className="my-3"),
            html.Div(id="headgroup-ref", className="text-muted small")
        ]),
        className="shadow-sm"
    ),

    html.Div(className="mb-4")
], fluid=True)

# ---------------------------------------------------------
# Species options update
# ---------------------------------------------------------
@app.callback(
    Output("species-select", "options"),
    Output("species-select", "value"),
    Input("headgroup-select", "value")
)
def update_species_options(headgroup_value):
    if headgroup_value is None or headgroup_value == "__ALL__":
        return [], None
    sp = species_values_for_headgroup(df_long, headgroup_value)
    species = sorted(sp["Species"].dropna().unique().tolist(), key=species_sort_key)
    opts = [{"label": s, "value": s} for s in species]
    return opts, None

# ---------------------------------------------------------
# Main figure + headgroup full-name reference
# ---------------------------------------------------------
@app.callback(
    Output("main-plot","figure"),
    Output("helper-text","children"),
    Output("headgroup-ref","children"),
    Input("headgroup-select","value"),
    Input("species-select","value"),
    Input("axis-orientation","value"),
    Input("axis-scale","value"),
)
def render_plot(headgroup_value, species_value, orientation, axis_scale):
    swap_axes = (orientation == "yx")

    # 1) All headgroups as fractions
    if headgroup_value is None or headgroup_value == "__ALL__":
        frac = headgroup_fractions(df_long)
        if frac.empty:
            return empty_fig("No data for headgroup fractions"), "", ""
        title = "Headgroup Fractions (sum = 1 within each sample)"
        fig, ordered_groups = make_grouped_box_figure(
            df_grp=frac, value_col="Fraction", group_col="Headgroup",
            title=title, swap_axes=swap_axes, scale=axis_scale
        )
        msg = "Fractions are computed per sample: each headgroup’s sum divided by the total across headgroups."
        # build full-name legend for visible headgroups
        ref_spans = []
        for abbr in ordered_groups:
            full = headgroup_full_names.get(abbr)
            if full:
                ref_spans.append(html.Span([html.B(f"{abbr}"), f" — {full}"], className="me-3"))
        ref_div = html.Div([html.Span("Headgroup reference: ", className="me-2")] + ref_spans) if ref_spans else ""
        return fig, msg, ref_div

    # 2) Headgroup + species selected → condition-wise boxes
    if species_value:
        sp = species_values_for_headgroup(df_long, headgroup_value)
        sp = sp[sp["Species"] == species_value].copy()
        if sp.empty:
            return empty_fig(f"No data for {headgroup_value} {species_value}"), "", ""
        title = f"{headgroup_value} {species_value} — Species Box Plot (sums of intensities)"
        sp_disp = sp.copy()
        sp_disp["DisplayGroup"] = sp_disp["Condition"]
        fig, ordered_groups = make_grouped_box_figure(
            df_grp=sp_disp, value_col="Value", group_col="DisplayGroup",
            title=title, swap_axes=swap_axes, scale=axis_scale
        )
        if not swap_axes:
            fig.update_xaxes(title="Condition")
        else:
            fig.update_yaxes(title="Condition")
        msg = "Species defined by carbon (e.g., 34:1). Values are summed intensities."
        # reference: full name of selected headgroup
        full = headgroup_full_names.get(headgroup_value)
        ref_div = html.Div([html.Span("Headgroup reference: ", className="me-2"),
                            html.Span([html.B(headgroup_value), f" — {full}"] if full else headgroup_value)])
        return fig, msg, ref_div

    # 3) Headgroup only → all species (by carbon)
    sp = species_values_for_headgroup(df_long, headgroup_value)
    if sp.empty:
        return empty_fig(f"No species data for {headgroup_value}"), "", ""
    title = f"{headgroup_value} — Species (by Carbon) Box Plot (sums of intensities)"
    fig, ordered_groups = make_grouped_box_figure(
        df_grp=sp, value_col="Value", group_col="Species",
        title=title, swap_axes=swap_axes, scale=axis_scale
    )
    # reference
    full = headgroup_full_names.get(headgroup_value)
    ref_items = [html.Span([html.B(headgroup_value), f" — {full}"] if full else headgroup_value,
                           className="me-3"),
                 html.Span("(Species sorted: small → big by carbon)")]
    ref_div = html.Div([html.Span("Reference: ", className="me-2")] + ref_items)
    msg = "Select a species (carbon) from the dropdown to focus on a single one."
    return fig, msg, ref_div

# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
