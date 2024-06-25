# Master's Dissertation: Gene Reprioritization in Cancer Cell Lines through Heterogeneous GNNs

## 1 Python Scripts

### 1.1 HetGNN
- **hetgnn_main.py**
  - Main script for HetGNN. Layer configurations can be modified in `gat_dependency/GAT_model.py`.

### 1.2 DLP 
- **DLP_main.py**
  - Implements DLP with the Whole split method. 
- **DLP_cellsplit.py**
  - Implements DLP with the Cell Only split method. 

### 1.3 Graph Construction
- **construct_heterogeneous_graph_jihwan.py**
  - Script for constructing a heterogeneous graph.
- **construct_heterogeneous_graph_pyG_jihwan.py**
  - Script for constructing a heterogeneous graph with biological features using Pytorch Geometric.

## 2 Environment Setup

- **Conda list**
  - Conda environment settings required for running the project.
 
- Build conda virtual environment for running codes\
  - yml file contains all necessary packages
```
conda env create --name envname --file=envrironment.yml
```

## 3 Results Generation

- **Final_Result**
  - Contains code for generating figures, tables, and lists of reprioritized genes. (See Jupyter Notebook in folder)
  - Contains generated figures by pptxm can be checked in `Final_Result/Fugures made.pptx`.

## 4 How to Run the Code

You can run the scripts with the following format, adjusting arguments using `--arg`:

```bash
python hetgnn_main.py --epochs 30 --lr 0.0001 --batch_size 256 --emb_dim 512 --seed 42 --exp_name example_test
```

## 5 Resources and Data Sources

All resources and data sources mentioned in this thesis are publicly accessible.


### 5.1 Reactome

- **Protein-protein network:** [Download FIsInGene with annotations](https://reactome.org/download/tools/ReatomeFIs/FIsInGene_070323_with_annotations.txt.zip)

### 5.2 Dependency Map

- **DepMap Public 23Q2 Files:** [View all data](https://depmap.org/portal/data_page/?tab=allData)
  - **DepMap Score:** CRISPRGeneEffect.csv
  - **Cell Line Name:** Model.csv
  - **Expression:** OmicsExpressionProteinCodingGenesTPMLogp1.csv
  - **Copy Number Variation:** OmicsCNGene.csv
  - **Control:** AchillesCommonEssentialControls.csv

### 5.3 MsigDB

- **CGP:** [Download CGP Symbols (GMT)](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/c2.cgp.v2023.2.Hs.symbols.gmt)
- **CP:** [Download CP Symbols (GMT)](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/c2.cp.v2023.2.Hs.symbols.gmt)
- **GO:** [Download GO Symbols (GMT)](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/c5.go.v2023.2.Hs.symbols.gmt)
- **BP:** [Download BP Symbols (GMT)](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/c5.go.bp.v2023.2.Hs.symbols.gmt)


## 6 WandB (Weights & Biases)

All results can be found in WandB and check comments in overview first!

- **DLP:** [View DLP Project](https://wandb.ai/jilim97/Final_DLP?nw=nwuserjilim97)
- **HetGNN:** [View HetGNN Project](https://wandb.ai/jilim97/Final_HetGNN?nw=nwuserjilim97)
- **HetGNN-Lin:** [View HetGNN-Lin Project](https://wandb.ai/jilim97/Final_HetGNN_Lin?nw=nwuserjilim97)
