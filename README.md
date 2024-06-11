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

python hetgnn_main.py --epochs 30 --lr 0.00001 --batch_size 256 --emb_dim 512 --seed 42 --exp_name example_test \

cods can be run like this format, adjusting arguement using --arg
