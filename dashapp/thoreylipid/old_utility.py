import re
from collections import defaultdict
import pandas as pd

# ----------------------------------------------
# Regex patterns
# ----------------------------------------------
# Matches things like "PC 34:2", "Cer_42:1;O3"
# base_pattern = re.compile(r'([A-Za-z]+)[ _]?(\d+):(\d+)(?:;([A-Za-z0-9]+))?')
base_pattern = re.compile(r'([A-Za-z0-9]+)[ _]?(\d+):(\d+)(?:;([A-Za-z0-9]+))?')

# Matches side chains like "18:1(OH)"
fatty_acyl_pattern = re.compile(r'(\d+):(\d+)(?:\(([^)]+)\))?')

# ----------------------------------------------
# Utility functions
# ----------------------------------------------
def bin_carbons(c):
    if c < 15:
        return "<15C"
    elif 15 <= c <= 17:
        return "15–17C"
    elif 18 <= c <= 20:
        return "18–20C"
    elif 21 <= c <= 23:
        return "21–23C"
    elif 24 <= c <= 26:
        return "24–26C"
    elif 27 <= c <= 29:
        return "27–29C"
    elif 30 <= c <= 32:
        return "30–32C"
    elif 33 <= c <= 35:
        return "33–35C"
    elif 36 <= c <= 38:
        return "36–38C"
    elif 39 <= c <= 41:
        return "39–41C"
    elif 42 <= c <= 44:
        return "42–44C"
    elif 45 <= c <= 47:
        return "45–47C"
    elif c >= 48:
        return "48+C"
    else:
        return "Misclassified"

def get_saturation_class(db):
    if db == 0:
        return "Saturated"
    elif db == 1:
        return "Monounsaturated"
    else:
        return "Polyunsaturated"

def is_oxidized(mods):
    ox_keywords = ['OH', 'OOH', 'oxo', 'epoxy', 'O2', 'O3']
    return any(any(ox in m.upper() for ox in ox_keywords) for m in mods)

# ----------------------------------------------
# Lipid grouping function
# ----------------------------------------------
def group_lipids(df):
    lipid_names = df["Metabolite name"].dropna().unique()
    grouped_lipids = defaultdict(list)
    unparsed_lipids = []

    for lipid in lipid_names:
        parts = re.split(r'[|/]', lipid)
        headgroup = None
        total_carbons = 0
        total_double_bonds = 0
        modifications = []
        parsed = False

        # Prefer main lipid spec (e.g., PC 34:1) if available
        for part in parts:
            part = part.strip()
            match = base_pattern.fullmatch(part)  # match entire part only
            if match:
                parsed = True
                hg, c, db, mod = match.groups()
                headgroup = hg
                total_carbons = int(c)
                total_double_bonds = int(db)
                if mod:
                    modifications.append(mod)
                break  # Don't double count with FA chains

        # If no full match found, try parsing fatty acid chains
        if not parsed:
            for part in parts:
                part = part.strip()
                fa_match = fatty_acyl_pattern.fullmatch(part)
                if fa_match:
                    parsed = True
                    c, db, mod = fa_match.groups()
                    total_carbons += int(c)
                    total_double_bonds += int(db)
                    if mod:
                        modifications.append(mod)

        if not parsed:
            unparsed_lipids.append(lipid)
            continue

        carbon_bin = bin_carbons(total_carbons)
        saturation = get_saturation_class(total_double_bonds)
        oxidation = "Oxidized" if is_oxidized(modifications) else "Not Oxidized"
        group_key = (headgroup if headgroup else "FA", carbon_bin, saturation, oxidation)

        grouped_lipids[group_key].append(lipid)

    # 🔍 Only keep groups that actually have lipids
    filtered_grouped_lipids = {k: v for k, v in grouped_lipids.items() if v}

    return filtered_grouped_lipids, unparsed_lipids
