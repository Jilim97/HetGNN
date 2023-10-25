import sys
sys.path.append('/Users/jihwanlim/Documents/GitHub/NetworkAnalysis/')

from NetworkAnalysis.UndirectedInteractionNetwork import UndirectedInteractionNetwork

#from gat_dependency.utils import read_gmt_file
import torch_geometric.transforms as T
from torch_geometric.transforms import to_undirected
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
import gseapy as gp
import random
import pickle
import torch

cancer_type = 'Neuroblastoma'
BASE_PATH = "/Users/jihwanlim/Desktop/"
ppi = "Reactome"
remove_rpl = "_noRPL"
remove_commonE = ""
useSTD = "STD"
crispr_threshold_pos = -1.5
drugtarget_nw = "_drugtarget"
cell_feat_name = "CNV"

with open(BASE_PATH+f"multigraphs/{cancer_type.replace(' ', '_')}_{ppi}{remove_rpl}_{useSTD}{remove_commonE}_crispr{str(crispr_threshold_pos).replace('.','_')}.pickle", 'rb') as handle:
    mg_obj = pickle.load(handle)
all_genes_int = mg_obj.type2nodes['gene']
all_genes_name = [mg_obj.int2gene[i] for i in all_genes_int]

# PPI obj
ppi_obj = mg_obj.getEdgeType_subset(edge_type='scaffold')
ppi_obj_new_gene2int = {n:i for i, n in enumerate(all_genes_name)}
ppi_obj_new_int2gene = {v:k for k, v in ppi_obj_new_gene2int.items()}
ppi_interactions = ppi_obj.getInteractionNamed()
ppi_interactions = ppi_interactions.applymap(lambda x: ppi_obj_new_gene2int[x])

# DEP obj
dep_obj = mg_obj.getEdgeType_subset(edge_type='depmap')
cells = [k for k, v in mg_obj.node_type_names.items() if v == 'cell']
cell2int = {c:i for i, c in enumerate(cells)}
int2cell = {v:k for k, v in cell2int.items()}
dep_interactions = dep_obj.getInteractionNamed()
dep_genes = [dep_obj.int2gene[i] for i in dep_obj.type2nodes['gene']]

dep_interactions.loc[~dep_interactions.Gene_A.isin(cells), ['Gene_A', 'Gene_B']] = \
    dep_interactions.loc[~dep_interactions.Gene_A.isin(cells), ['Gene_B', 'Gene_A']].values

assert dep_interactions.Gene_A.isin(cells).sum() == dep_interactions.shape[0]
dep_interactions = dep_interactions.applymap(lambda x: cell2int[x] if x in cell2int else ppi_obj_new_gene2int[x])
dep_interactions = dep_interactions[['Gene_B', 'Gene_A']]
print(dep_interactions.shape)

# Drug target network
if drugtarget_nw:
    with open(BASE_PATH+f'multigraphs/drug_target_network.pickle', 'rb') as handle:
        drugtarget_nw_obj = pickle.load(handle)
    drugtarget_interactions = drugtarget_nw_obj.getInteractionNamed()
    drugs = list(set(drugtarget_nw_obj.getInteractionNamed().Gene_A))
    drug2id = {d:i for i, d in enumerate(drugs)}
    drugtarget_interactions.Gene_A = drugtarget_interactions.Gene_A.apply(lambda x: drug2id[x])
    drugtarget_interactions.Gene_B = drugtarget_interactions.Gene_B.apply(lambda x: ppi_obj_new_gene2int[x])

# Oversample low pos -------------------------------------------------------------------------------------------------------------------------------------
# crispr_neurobl = pd.read_csv(BASE_PATH+f"data/crispr_{cancer_type}_{ppi}.csv", index_col=0)
# crispr_neurobl_int = crispr_neurobl.copy(deep=True)
# crispr_neurobl_int.index = [cell2int[i] for i in crispr_neurobl.index]
# crispr_neurobl_int.columns = [ppi_obj_new_gene2int[i] for i in crispr_neurobl.columns]
# dep_genes = [ppi_obj_new_gene2int[i] for i in dep_obj.node_names if i not in cells]
# crispr_neurobl_int = crispr_neurobl_int.loc[:, dep_genes]
# crispr_neurobl_bin = crispr_neurobl_int.applymap(lambda x: int(x < crispr_threshold_pos))

# to_sample = len(cells)-crispr_neurobl_bin.sum()
# for gi, tosample in to_sample.iteritems():
#     possible_edges = list(map(tuple, dep_interactions[dep_interactions.Gene_B == gi].values))
#     to_concat = pd.DataFrame(random.choices(population=possible_edges, k=tosample), columns=['Gene_B', 'Gene_A'], dtype=int)
#     dep_interactions = pd.concat([dep_interactions, to_concat])
# print(dep_interactions.shape)

# -------------------------------------------------------------------------------------------------------------------------------------

# Gene features
cgn = read_gmt_file(BASE_PATH+"data/c2.cgp.v2023.1.Hs.symbols.gmt", ppi_obj)
cgn_df = pd.DataFrame(np.zeros((len(all_genes_name), len(cgn))), index=all_genes_name, columns=list(cgn.keys()))
for k, v in cgn.items():
    cgn_df.loc[list(v), k] = 1
zero_gene_feat = cgn_df.index[cgn_df.sum(axis=1) == 0] # dit mag niet want alle genes moeten features hebbem
# Check hoeveel van de dep genes er in die all 0 zitten want anders basically heeft dit geen nut
zero_depgenes = set(zero_gene_feat) & set(dep_genes)
len(zero_depgenes)

gene_feat = torch.from_numpy(cgn_df.values).to(torch.float)

# Cell features
if cell_feat_name == "expression":
    path = BASE_PATH+'data/raw/OmicsExpressionProteinCodingGenesTPMLogp1.csv'
    ccle_expression = pd.read_csv(path, header=0, index_col=0)
    ccle_expression.columns = [i.split(' ')[0] for i in ccle_expression.columns]
    # subset_nodes = list(set(ccle_expression.columns) & set(all_genes_name))
    cancer_expression = ccle_expression.loc[list(set(cells) & set(ccle_expression.index))]

    hvg_q = cancer_expression.std().quantile(q=0.95)
    hvg_final = cancer_expression.std()[cancer_expression.std() >= hvg_q].index

    cancer_expression_hvg = cancer_expression[hvg_final]
    # cancer_expression_full = pd.concat([cancer_expression,
    #                                     pd.DataFrame(np.tile(cancer_expression.mean().values, (len(set(cells) - set(cancer_expression.index)), 1)),
    #                                                  index=list(set(cells) - set(cancer_expression.index)), columns=cancer_expression.columns)])
    cancer_expression_full = pd.concat([cancer_expression_hvg,
                                        pd.DataFrame(np.tile(cancer_expression_hvg.mean().values, (len(set(cells) - set(cancer_expression_hvg.index)), 1)),
                                                    index=list(set(cells) - set(cancer_expression_hvg.index)), columns=cancer_expression_hvg.columns)])
    cell_feat = torch.from_numpy(cancer_expression_full.loc[cell2int.keys()].values).to(torch.float)

elif cell_feat_name == "CNV":
    path = BASE_PATH+'data/raw/OmicsCNGene.csv'
    ccle_cnv = pd.read_csv(path, header=0, index_col=0)
    ccle_cnv.columns = [i.split(' ')[0] for i in ccle_cnv.columns]
    ccle_cnv = ccle_cnv[ccle_cnv.columns[ccle_cnv.isna().sum() == 0]]
    ccle_cnv = ccle_cnv.loc[list(set(cells) & set(ccle_cnv.index))]

    hvg_q = ccle_cnv.std().quantile(q=0.99)
    hvg_final = ccle_cnv.std()[ccle_cnv.std() >= hvg_q].index

    ccle_cnv_hvg = ccle_cnv[hvg_final]
    cell_feat = torch.from_numpy(ccle_cnv_hvg.loc[cell2int.keys()].values).to(torch.float)
    
# Drug features 
if drugtarget_nw:
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import Chem
    primary_screen_info = pd.read_csv('data/primary-screen-replicate-collapsed-treatment-info.csv', header=0, index_col=2)
    primary_screen_info = primary_screen_info[~primary_screen_info.index.duplicated(keep='first')]
    primary_screen_info.index = [i+'_drug' if isinstance(i, str) else i for i in primary_screen_info.index]
    primary_screen_info.smiles.fillna('', inplace=True)
    primary_screen_info.smiles = primary_screen_info.smiles.apply(lambda x: x.split(',')[0])
    smiles_drugs = primary_screen_info.loc[drugs].smiles.values
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    bit_morganfp = pd.DataFrame([list(mfpgen.GetFingerprint(Chem.MolFromSmiles(i))) if len(i) > 0 else [0]*1024 for i in smiles_drugs],
                                index=smiles_drugs)

# Construction of the PyTroch geometric heterogeneous graph
data = HeteroData()

# First construt the node ids, easy from the MG obj
data['gene'].node_id = torch.tensor(list(ppi_obj_new_gene2int.values()))
data['gene'].names = list(ppi_obj_new_gene2int.keys())
data['cell'].node_id = torch.tensor(list(cell2int.values()))
data['cell'].names = list(cell2int.keys())
if drugtarget_nw:
    data['drug'].node_id = torch.tensor([drug2id[i] for i in drugs])
    data['drug'].names = drugs

# Add the node features and edge indices
data['gene'].x = gene_feat
data['cell'].x = cell_feat
if drugtarget_nw:
    data['drug'].x = torch.from_numpy(bit_morganfp.values).to(torch.float)


data['gene', 'interacts_with', 'gene'].edge_index = torch.tensor(ppi_interactions.values.transpose(), dtype=torch.long)
data['gene', 'dependency_of', 'cell'].edge_index = torch.tensor(dep_interactions.values.transpose(), dtype=torch.long)
if drugtarget_nw:
    data['drug', 'has_target', 'gene'].edge_index = torch.tensor(drugtarget_interactions.values.transpose(), dtype=torch.long)
# Convert to undirected graph
data = T.ToUndirected(merge=False)(data)
assert data.validate()

print(data)

torch.save(obj=data, f=BASE_PATH+f"multigraphs/heteroData_gene_cell_{cancer_type.replace(' ', '_')}_{ppi}"\
          f"_crispr{str(crispr_threshold_pos).replace('.','_')}{drugtarget_nw}_cgp_cnv.pt")
