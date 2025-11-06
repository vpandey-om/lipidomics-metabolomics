# **Lipidomics & Metabolomics Analysis Pipeline**

This repository contains scripts and workflows for processing lipidomics
and metabolomics datasets, generating statistical summaries, boxplots,
volcano plots, and species-level visualizations used in downstream
biological interpretation.

## **Repository Structure**

    lipidomics-metabolomics/
    │
    ├── data/                     # Input datasets
    ├── scripts/                  # All analysis scripts
    ├── headgroup_plots_sum/      # Output: summed headgroup plots
    ├── headgroup_species_plots/  # Output: individual lipid species plots
    ├── PLmode2_allinone/         # Output: combined PL mode figures
    ├── metabolomics/             # Output: metabolomics visualizations
    └── README.md

## **Running the Analysis**

All commands assume you are inside the repository folder:

``` bash
cd lipidomics-metabolomics/scripts
```

### **1. Main lipidomics + metabolomics analysis**

``` bash
python met_analysis_again.py
```

### **2. Generate headgroup-level boxplots**

``` bash
python box_plot_final.py   --domain headgroup   --groups "PC,PE,PI,SM,HexCer"   --xtick-rot 90
```

### **3. Plot selected individual lipid species**

``` bash
python individual_box_plot.py   --file "Final_Combined_Lipidomics.txt"   --species "LPE 20:4,LPE 18:0,PE 38:4,LPE O-18:0"
```

### **4. Individual volcano plots**

``` bash
python individual_volcano.py
```

### **5. Summarized volcano plots**

``` bash
python sum_volcano_lipidomics.py
```

## **Requirements**

``` bash
pip install pandas numpy matplotlib seaborn scipy
```
