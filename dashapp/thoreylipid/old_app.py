# app/app.py

import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from utility import group_lipids

# ----------------------------------------------
# Load and clean data
# ----------------------------------------------
file_path = "Final_Combined_Lipidomics.txt"
df = pd.read_csv(file_path, sep='\t')
df.columns = df.columns.str.strip()

# Define sample columns
sample_cols = [
    'Ctrl_1', 'Ctrl_2', 'Ctrl_3', 'Ctrl_4',
    'POI7_1', 'POI7_2', 'POI7_3', 'POI7_4',
    'POI8_1', 'POI8_2', 'POI8_3', 'POI8_4'
]

# ----------------------------------------------
# Group lipids
# ----------------------------------------------
grouped_lipids, unparsed_lipids = group_lipids(df)

# ----------------------------------------------
# Dash App setup
# ----------------------------------------------
app = Dash(__name__)
app.title = "Lipidomics Viewer"

server = app.server  # THIS IS IMPORTANT



# Dropdown group options
group_keys = sorted(set((k[1], k[2], k[3]) for k in grouped_lipids))
group_options = [
    {"label": f"{cb} | {sat} | {ox}", "value": f"{cb}|{sat}|{ox}"}
    for cb, sat, ox in group_keys
]

# App layout
app.layout = html.Div([
    html.H1("Lipidomics Grouped Visualization"),
    html.Label("Select Lipid Groups:"),
    dcc.Dropdown(
        options=group_options,
        multi=True,
        id='group-selector'
    ),
    dcc.Graph(id='boxplot-normal'),
    dcc.Graph(id='boxplot-log10')
])

# ----------------------------------------------
# Empty placeholder plot
# ----------------------------------------------
def empty_fig(title="No Data"):
    return px.scatter(title=title).update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{
            "text": title,
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 18}
        }]
    )

# ----------------------------------------------
# Callback for updating plots
# ----------------------------------------------
@app.callback(
    [Output('boxplot-normal', 'figure'),
     Output('boxplot-log10', 'figure')],
    [Input('group-selector', 'value')]
)
def update_boxplots(selected_groups):
    if not selected_groups:
        return empty_fig("No group selected"), empty_fig("No group selected")

    lipid_subset = []
    for key, lipids in grouped_lipids.items():
        label = f"{key[1]}|{key[2]}|{key[3]}"
        if label in selected_groups:
            for lipid in lipids:
                lipid_subset.append((key[0], lipid))

    if not lipid_subset:
        return empty_fig("No matching lipids"), empty_fig("No matching lipids")

    sub_df = df.set_index("Metabolite name")
    melt_data = []

    for headgroup, lipid in lipid_subset:
        if lipid not in sub_df.index:
            continue
        row = sub_df.loc[lipid]
        for col in sample_cols:
            try:
                value = row[col]
                condition = col.split('_')[0]
                # Small epsilon to avoid log10(0)
                epsilon = 1e-9
                melt_data.append({
                    "Headgroup": headgroup,
                    "Condition": condition,
                    "Sample": col,
                    "Lipid": lipid,  # Add short name
                    "Value": value,
                    "Log10Value": np.log10(value + epsilon)
                })
            except KeyError:
                continue

    plot_df = pd.DataFrame(melt_data)

    if plot_df.empty:
        return empty_fig("No data to display"), empty_fig("No data to display")

    fig_normal = px.box(
        plot_df,
        x="Headgroup",
        y="Value",
        color="Condition",
        title="Lipid Abundance (Normal Scale)",
        points="all",
        hover_data=["Lipid"],
    )

    fig_log = px.box(
        plot_df,
        x="Headgroup",
        y="Log10Value",
        color="Condition",
        title="Lipid Abundance (Log10 Scale)",
        points="all",
        hover_data=["Lipid"],
    )

    return fig_normal, fig_log

# ----------------------------------------------
# Run the app
# ----------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
