import pandas as pd
import re

# Load the dataset
file_path = "Final_Combined_Lipidomics.txt"
df = pd.read_csv(file_path, sep="\t")
df.columns = df.columns.str.strip()

# Updated function to extract headgroup using improved base_pattern
def extract_headgroup(name):
    if pd.isna(name):
        return None
    base_pattern = re.compile(r'([A-Za-z0-9\-]+)[ _]?(\d+):(\d+)(?:;([A-Za-z0-9]+))?')
    match = base_pattern.match(name)
    if match:
        return match.group(1)
    return None

# Apply extraction
df["Extracted_Headgroup"] = df["Metabolite name"].apply(extract_headgroup)

# Get unique headgroups and ontology values
unique_headgroups = sorted(df["Extracted_Headgroup"].dropna().unique())
unique_ontology = sorted(df["Ontology"].dropna().unique())

# Save to text files
headgroup_path = "/mnt/data/unique_headgroups.txt"
ontology_path = "/mnt/data/unique_ontology.txt"

with open(headgroup_path, "w") as f:
    for hg in unique_headgroups:
        f.write(f"{hg}\n")

with open(ontology_path, "w") as f:
    for ont in unique_ontology:
        f.write(f"{ont}\n")


