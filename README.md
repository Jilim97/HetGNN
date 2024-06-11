# HetGNN Project Files Overview

This document provides an overview of the main scripts and environment settings used in the HetGNN project.

## 1. Python Scripts

### HetGNN
- **hetgnn_main.py**
  - Main script for HetGNN. Layer configurations can be modified in `gat_dependency/GAT_model.py`.

### DLP (Dependency Learning Protocol)
- **DLP_main.py**
  - Implements DLP with the Whole split method. This script is not the main script for DLP operations.
- **DLP_cellsplit.py**
  - Implements DLP with the Cell Only split method. This is the main script for DLP operations.

### Graph Construction
- **construct_heterogeneous_graph_jihwan.py**
  - Script for constructing a heterogeneous graph.
- **construct_heterogeneous_graph_pyG_jihwan.py**
  - Script for constructing a heterogeneous graph using PyTorch Geometric.

## 2. Environment Setup

- **Conda list**
  - Provides the conda environment settings required for running the project.

## 3. Results Generation

- **Final_Result**
  - Contains code for generating figures, tables, and lists of reprioritized genes.
  - Contains generated figures by pptx

### 3.1 WandB (Weights & Biases)

- **DLP:** [View DLP Project](https://wandb.ai/jilim97/Final_DLP?nw=nwuserjilim97)
- **HetGNN:** [View HetGNN Project](https://wandb.ai/jilim97/Final_HetGNN?nw=nwuserjilim97)
- **HetGNN-Lin:** [View HetGNN-Lin Project](https://wandb.ai/jilim97/Final_HetGNN_Lin?nw=nwuserjilim97)

## 4. How to Run the Code

You can run the scripts with the following format, adjusting arguments using `--arg`:

```bash
python hetgnn_main.py --epochs 30 --lr 0.0001 --batch_size 256 --emb_dim 512 --seed 42 --exp_name example_test
```

## 5. Resources and Data Sources

This README provides detailed links to the resources and data sources used in the project.

### 1. Reactome

- **Protein-protein network:** [Download FIsInGene with annotations (Zip file)](https://reactome.org/download/tools/ReatomeFIs/FIsInGene_070323_with_annotations.txt.zip)

### 2. Dependency Map

- **DepMap Public 23Q2 Files:** [View all data](https://depmap.org/portal/data_page/?tab=allData)
  - **DepMap Score:** CRISPRGeneEffect.csv
  - **Cell Line Name:** Model.csv
  - **Expression:** OmicsExpressionProteinCodingGenesTPMLogp1.csv
  - **Copy Number Variation:** OmicsCNGene.csv
  - **Control:** AchillesCommonEssentialControls.csv

### 3. MsigDB

- **CGP:** [Download CGP Symbols (GMT)](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/c2.cgp.v2023.2.Hs.symbols.gmt)
- **CP:** [Download CP Symbols (GMT)](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/c2.cp.v2023.2.Hs.symbols.gmt)
- **GO:** [Download GO Symbols (GMT)](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/c5.go.v2023.2.Hs.symbols.gmt)
- **BP:** [Download BP Symbols (GMT)](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/c5.go.bp.v2023.2.Hs.symbols.gmt)


