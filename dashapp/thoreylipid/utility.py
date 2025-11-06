import re
from collections import defaultdict
import pandas as pd
import numpy as np

# ----------------------------------------------
# Regex patterns
# ----------------------------------------------
base_pattern = re.compile(r'([A-Za-z0-9\-]+)[ _]?(\d+):(\d+)(?:;([A-Za-z0-9]+))?')
fatty_acyl_pattern = re.compile(r'(\d+):(\d+)(?:\(([^)]+)\))?')
ether_lipid_pattern = re.compile(r'([A-Za-z0-9\-]+)[ _]?O-(\d+):(\d+)')
lpen_fa_pattern = re.compile(r'\(?FA\)? ?(\d+):(\d+)')
plasmalogen_pattern = re.compile(r'([A-Za-z0-9\-]+) P-(\d+):(\d+)')
sterol_pattern = re.compile(r'(ST) (\d+):(\d+);([A-Za-z0-9;]+)')
isotope_cleanup = re.compile(r'\(d\d+\)')

# ----------------------------------------------
# Utility functions
# ----------------------------------------------

def bin_carbons_dynamic(carbon_counts):
    values = list(carbon_counts.values())
    if len(values) < 3:
        print("⚠️ Not enough values to compute dynamic bins. Returning 'Unbinned'.")
        return {lipid: f"Unbinned ({c}C)" for lipid, c in carbon_counts.items()}

    q33, q66 = np.percentile(values, [33, 66])
    q33 = int(np.floor(q33))
    q66 = int(np.floor(q66))

    # print(f"📊 Dynamic Carbon Length Binning:")
    # print(f"   Short  : ≤ {q33}")
    # print(f"   Medium : > {q33} and ≤ {q66}")
    # print(f"   Long   : > {q66}\n")

    bins = {}
    for lipid, c in carbon_counts.items():
        if c <= q33:
            bins[lipid] = f"Short (≤{q33})"
        elif c <= q66:
            bins[lipid] = f"Medium ({q33+1}–{q66})"
        else:
            bins[lipid] = f"Long (>{q66})"
    return bins




def bin_carbons(c):
    if c <= 5:
        return "SCFA (≤5C)"
    elif 6 <= c <= 12:
        return "MCFA (6–12C)"
    elif 13 <= c <= 21:
        return "LCFA (13–21C)"
    else:
        return "VLCFA (≥22C)"

def bin_carbons2(c):
    if c < 15: return "<15C"
    elif 15 <= c < 17: return "15–16C"
    elif 17 <= c < 19: return "17–18C"
    elif 19 <= c < 21: return "19–20C"
    elif 21 <= c < 23: return "21–22C"
    elif 23 <= c < 25: return "23–24C"
    elif 25 <= c < 27: return "25–26C"
    elif 27 <= c < 29: return "27–28C"
    elif 29 <= c < 31: return "29–30C"
    elif 31 <= c < 33: return "31–32C"
    elif 33 <= c < 35: return "33–34C"
    elif 35 <= c < 37: return "35–36C"
    elif 37 <= c < 39: return "37–38C"
    elif 39 <= c < 41: return "39–40C"
    elif 41 <= c < 43: return "41–42C"
    elif 43 <= c < 45: return "43–44C"
    elif 45 <= c < 47: return "45–46C"
    elif 47 <= c < 49: return "47–48C"
    else: return "49+C"

def get_saturation_class(db):
    if db == 0: return "Saturated"
    elif db == 1: return "Monounsaturated"
    else: return "Polyunsaturated"

def is_oxidized(mods):
    ox_keywords = ['OH', 'OOH', 'oxo', 'epoxy', 'O2', 'O3']
    return any(any(ox in m.upper() for ox in ox_keywords) for m in mods)

def assign_functional_group(headgroup):
    if headgroup in ['PC', 'PE', 'PS', 'PI', 'LPE']:
        return 'Structural'
    elif headgroup in ['Cer', 'HexCer', 'DAG', 'PA']:
        return 'Signaling'
    elif headgroup in ['TG', 'DG']:
        return 'Storage'
    else:
        return 'Other'

# ----------------------------------------------
# Lipid grouping function
# ----------------------------------------------
def group_lipids(df, mode='Headgroup'):
    # if mode == 'Default':  # 👈 Full Profile mode
    #     grouped = {lipid: [lipid] for lipid in df["Metabolite name"].dropna().unique()}
    #     return grouped, []
    # After lipid parsing loop

    
    
    if mode == 'Ontology' and 'Ontology' in df.columns:
        grouped_lipids = defaultdict(list)
        for _, row in df.iterrows():
            key = row['Ontology'] if pd.notna(row['Ontology']) else 'Unknown'
            grouped_lipids[key].append(row['Metabolite name'])
        return grouped_lipids, []

    lipid_names = df["Metabolite name"].dropna().unique()
    grouped_lipids = defaultdict(list)
    unparsed_lipids = []
    lipid_carbons = {}
    lipid_meta = {}  # <-- Add this before the for-loop over lipid_names
    
    for lipid in lipid_names:
        parts = re.split(r'[|/]', lipid)
        headgroup = None
        total_carbons = 0
        total_double_bonds = 0
        modifications = []
        parsed = False

        for part in parts:
            part = part.strip()

            match = base_pattern.fullmatch(part)
            if match:
                parsed = True
                hg, c, db, mod = match.groups()
                headgroup = hg
                total_carbons = int(c)
                total_double_bonds = int(db)
                if mod: modifications.append(mod)
                break
           
            ether_match = ether_lipid_pattern.match(part)
            if ether_match:
                parsed = True
                hg, c, db = ether_match.groups()
                headgroup = hg
                total_carbons = int(c)
                total_double_bonds = int(db)
                if ";" in part:
                    mods = part.split(";")[1:]
                    modifications.extend(mods)
                modifications.append("O-ether")
                break

            plasmalogen_match = plasmalogen_pattern.match(part)
            if plasmalogen_match:
                parsed = True
                hg, c, db = plasmalogen_match.groups()
                headgroup = hg
                total_carbons = int(c)
                total_double_bonds = int(db)
                modifications.append("P-vinyl")
                break

            sterol_match = sterol_pattern.match(part)
            if sterol_match:
                parsed = True
                hg, c, db, mods = sterol_match.groups()
                headgroup = hg
                total_carbons = int(c)
                total_double_bonds = int(db)
                modifications.extend(mods.split(";"))
                break

            cleaned = isotope_cleanup.sub('', part)
            match = base_pattern.fullmatch(cleaned)
            if match:
                parsed = True
                hg, c, db, mod = match.groups()
                headgroup = hg
                total_carbons = int(c)
                total_double_bonds = int(db)
                if mod: modifications.append(mod)
                break

        if not parsed:
            for part in parts:
                part = part.strip()
                fa_matches = lpen_fa_pattern.findall(part)
                if fa_matches:
                    parsed = True
                    for c, db in fa_matches:
                        total_carbons += int(c)
                        total_double_bonds += int(db)
                    if 'LPE' in part or 'LPE-N' in part:
                        headgroup = 'LPE'
                    break

                fa_match = fatty_acyl_pattern.fullmatch(part)
                if fa_match:
                    parsed = True
                    c, db, mod = fa_match.groups()
                    total_carbons += int(c)
                    total_double_bonds += int(db)
                    if mod: modifications.append(mod)

        if not parsed:
            unparsed_lipids.append(lipid)
            continue
        
        lipid_meta[lipid] = {
            'headgroup': headgroup,
            'carbons': total_carbons,
            'db': total_double_bonds,
            'mods': modifications
        }
        
        
        carbon_map = {lipid: meta['carbons'] for lipid, meta in lipid_meta.items()}
        carbon_bins = bin_carbons_dynamic(carbon_map)
        

        # carbon_bins2 = bin_carbons(total_carbons)
        # print(carbon_bins2)
        
        # Assign group keys
    for lipid, meta in lipid_meta.items():
        headgroup = meta['headgroup']
        total_carbons = meta['carbons']
        total_double_bonds = meta['db']
        modifications = meta['mods']
        carbon_bin = carbon_bins.get(lipid, "Unknown")
        saturation = get_saturation_class(total_double_bonds)
        oxidation = "Oxidized" if is_oxidized(modifications) else "Not Oxidized"
        functional_group = assign_functional_group(headgroup if headgroup else "FA")
        carbon_class = bin_carbons(total_carbons)      # SCFA / MCFA / LCFA / VLCFA
        carbon_bin2 = bin_carbons2(total_carbons)  

        if mode == 'Headgroup':
            group_key = headgroup
        elif mode == 'Headgroup + Carbon':
            group_key = f"{headgroup}_{carbon_bin}"
        elif mode == 'Headgroup + Carbon2':
            group_key = f"{headgroup}_{carbon_bin2}"
        elif mode == 'Headgroup + CarbonClass':
            group_key = f"{headgroup}_{carbon_class}"
        elif mode == 'Headgroup + Saturation':
            group_key = f"{headgroup}_DB{saturation}"
        elif mode == 'Headgroup + Oxidation':
            group_key = f"{headgroup}_OX{oxidation}"
        elif mode == 'Functional':
            group_key = functional_group
        elif mode == 'Headgroup + CarbonClass + Saturation':
            group_key = f"{headgroup}_{carbon_class}_{saturation}"
        elif mode == 'Headgroup + CarbonClass + Oxidation':
            group_key = f"{headgroup}_{carbon_class}_OX{oxidation}"
        
        else:
            group_key = f"({headgroup if headgroup else 'FA'}, {carbon_bin}, {saturation}, {oxidation})"

        grouped_lipids[group_key].append(lipid)
    
    #     saturation = get_saturation_class(total_double_bonds)
    #     oxidation = "Oxidized" if is_oxidized(modifications) else "Not Oxidized"
    #     functional_group = assign_functional_group(headgroup if headgroup else "FA")

    #     if mode == 'Headgroup':
    #         group_key = headgroup
    #     elif mode == 'Headgroup + Carbon':
    #         group_key = f"{headgroup}_{carbon_bin}"
    #         import pdb;pdb.set_trace()
    #     elif mode == 'Headgroup + DoubleBonds':
    #         group_key = f"{headgroup}_DB{total_double_bonds}"
    #     elif mode == 'Headgroup + Oxidation':
    #         group_key = f"{headgroup}_OX{oxidation}"
    #     elif mode == 'Functional':
    #         group_key = functional_group
    #     else:
    #         #group_key = (headgroup if headgroup else "FA", carbon_bin, saturation, oxidation)
    #         group_key = f"({headgroup if headgroup else 'FA'}, {carbon_bin}, {saturation}, {oxidation})"


    #     grouped_lipids[group_key].append(lipid)

    # print("\n--- SUMMARY ---")
    # print("Unparsed lipids:", len(unparsed_lipids))
    # print("Groups created:", len(grouped_lipids))
    # print("\n=== UNPARSED LIPIDS ONLY ===")
    # for lipid in unparsed_lipids:
    #     print(lipid)

    # filtered_grouped_lipids = {k: v for k, v in grouped_lipids.items() if v}
    return dict(grouped_lipids), unparsed_lipids

# ----------------------------------------------
# Optional filter example
# ----------------------------------------------
def filter_custom_rule(df):
    return df[df['Metabolite name'].str.contains('Cer') & (df['Metabolite name'].str.count('OH') >= 2)]