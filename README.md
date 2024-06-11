# HetGNN

## hetgnn_main.py

For HetGNN, the layers can be modified in gat_dependency/GAT_model.py

## DLP_main.py

DLP with Whole split method; not main for DLP

## DLP_cellsplit.py

DLP with Cell Only split method; main for DLP

## Graph Construction

1. construct_heterogeneous_graph_jihwan.py
2. construct_heterogeneous_graph_pyG_jihwan.py

## Conda list

Conda environment setting

## Final_Result

Code for generating figures, tables, and reprioritized gene lists

## Model run example:

python hetgnn_main.py --epochs 30 --lr 0.0001 --batch_size 256 --emb_dim 512 --seed 42 --exp_name example_test 

codes can be run like this format, adjusting arguement using --arg

# Resources and Data Sources

This README provides detailed links to the resources and data sources used in the project.

## 1. Reactome

- **Protein-protein network:** [Download FIsInGene with annotations (Zip file)](https://reactome.org/download/tools/ReatomeFIs/FIsInGene_070323_with_annotations.txt.zip)

## 2. Dependency Map

- **DepMap Public 23Q2 Files:** [View all data](https://depmap.org/portal/data_page/?tab=allData)
  - **DepMap Score:** CRISPRGeneEffect.csv
  - **Cell Line Name:** Model.csv
  - **Expression:** OmicsExpressionProteinCodingGenesTPMLogp1.csv
  - **Copy Number Variation:** OmicsCNGene.csv
  - **Control:** AchillesCommonEssentialControls.csv

## 3. MsigDB

- **CGP:** [Download CGP Symbols (GMT)](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/c2.cgp.v2023.2.Hs.symbols.gmt)
- **CP:** [Download CP Symbols (GMT)](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/c2.cp.v2023.2.Hs.symbols.gmt)
- **GO:** [Download GO Symbols (GMT)](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/c5.go.v2023.2.Hs.symbols.gmt)
- **BP:** [Download BP Symbols (GMT)](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/c5.go.bp.v2023.2.Hs.symbols.gmt)

## 4. WandB (Weights & Biases)

- **DLP:** [View DLP Project](https://wandb.ai/jilim97/Final_DLP?nw=nwuserjilim97)
- **HetGNN:** [View HetGNN Project](https://wandb.ai/jilim97/Final_HetGNN?nw=nwuserjilim97)
- **HetGNN-Lin:** [View HetGNN-Lin Project](https://wandb.ai/jilim97/Final_HetGNN_Lin?nw=nwuserjilim97)
