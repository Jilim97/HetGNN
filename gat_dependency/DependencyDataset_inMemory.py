from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.transforms import to_undirected, normalize_features
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os.path as osp
import pandas as pd
import numpy as np
import torch
import glob

class DependencyDataset_inMemory(InMemoryDataset):
    def __init__(self, root, ppi_obj, sensitivity_threshold, cancer_type=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.cancer_type = cancer_type
        self.focus_genes = set(ppi_obj.node_names)
        self.ppi_obj = ppi_obj
        self.sensitivity_threshold = sensitivity_threshold
        super(DependencyDataset_inMemory, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(osp.join(self.processed_dir, f'{self.cancer_type}.pt'))

    @property
    def raw_file_names(self):
        return ['CRISPRGeneEffect.csv', 'Model.csv', 'OmicsExpressionProteinCodingGenesTPMLogp1.csv',
                'OmicsSomaticMutations.csv']

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        # TODO: this is bad practice
        return [f"{self.cancer_type}.pt"]

    def download(self):
        pass

    def process(self):
        path = osp.join(self.raw_dir, 'Model.csv')
        ccles = pd.read_csv(path, header=0, index_col=0)
        if self.cancer_type != "all":
            grouped_ccle = ccles.groupby('OncotreeLineage').groups
            focus_cls = set(grouped_ccle[self.cancer_type])
        else:
            focus_cls = set(ccles.index)

        path = osp.join(self.raw_dir, 'OmicsExpressionProteinCodingGenesTPMLogp1.csv')
        ccle_expression = pd.read_csv(path, header=0, index_col=0)
        ccle_expression.columns = [i.split(' ')[0] for i in ccle_expression.columns]

        path = osp.join(self.raw_dir, 'OmicsSomaticMutations.csv')
        ccle_mutation = pd.read_csv(path, header=0, index_col=0)
        ccle_mutation = ccle_mutation[ccle_mutation['AF'] > 0.1]

        path = osp.join(self.raw_dir, 'CRISPRGeneEffect.csv')
        crispr_effect = pd.read_csv(path, header=0, index_col=0)
        crispr_effect.columns = [i.split(' ')[0] for i in crispr_effect.columns]

        focus_cls &= set(ccle_expression.index) & set(crispr_effect.index)
        data_list = []
        for cl in tqdm(focus_cls):
            x_expression = ccle_expression.loc[cl, self.ppi_obj.node_names]
            x_expression = x_expression[~x_expression.isna()]

            x_mutation = pd.Series(np.zeros(self.ppi_obj.N_nodes), index=self.ppi_obj.node_names).astype(int)
            mutated_genes = set(ccle_mutation.loc[ccle_mutation['DepMap_ID'] == cl].HugoSymbol)
            x_mutation.loc[list(mutated_genes & set(self.ppi_obj.node_names))] = 1

            # Normalize features! https://www.youtube.com/watch?v=FDCfw-YqWTE
            assert (x_expression.index == x_mutation.index).all(), "Feature index mismatch"
            feature_matrix = pd.concat([x_expression, x_mutation], axis=1)

            scaler_ = StandardScaler()
            feature_matrix_scaled = scaler_.fit_transform(feature_matrix.values)
            feature_matrix_t = torch.tensor(feature_matrix_scaled, dtype=torch.float)

            edge_index = torch.tensor(self.ppi_obj.interactions.values.transpose(),
                                      dtype=torch.long)
            edge_index = to_undirected.to_undirected(edge_index=edge_index)

            labels = (crispr_effect.loc[cl, self.ppi_obj.node_names] < -1).astype(float)

            graph_tensor = Data(x=feature_matrix_t, edge_index=edge_index, y=torch.tensor(labels).view(-1, 1).float(),
                                name=cl)
            assert graph_tensor.validate(raise_on_error=True) and not graph_tensor.is_directed(), "error"
            data_list.append(graph_tensor)

        data, slices = self.collate(data_list)
        torch.save((data, slices), osp.join(self.processed_dir, f'{self.cancer_type}.pt'))
