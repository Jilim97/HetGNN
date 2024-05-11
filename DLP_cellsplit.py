from gat_dependency.utils import read_h5py
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch.functional as F
import torch
import pickle 
from gat_dependency.GAT_model import DLP_model
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
import os
import argparse
import wandb
from datetime import datetime
from torch_geometric import seed_everything
from copy import deepcopy



def construct_complete_predMatrix(total_predictions: np.array,
                                  edge_index: torch.Tensor,
                                  index: list, columns: list) -> pd.DataFrame:
    total_preds_df = pd.DataFrame({"gene":edge_index[0], "cell":edge_index[1], "prob": total_predictions})
    # total_preds_df['gene'] = total_preds_df.gene.apply(lambda x: ppi_int2gene[x])
    # total_preds_df['cell'] = total_preds_df.cell.apply(lambda x: int2cell[x])

    dep_df = pd.DataFrame(index=index, columns=columns, dtype=float)
    for i in range(dep_df.shape[0]):
        tmp = total_preds_df.iloc[i*dep_df.shape[1]:(i+1)*dep_df.shape[1]]
        dep_df.loc[tmp.cell.iloc[0], tmp.gene.values] = tmp['prob'].values

    return dep_df

def to_homo(data):
    
    # Delete reverse direction (DLP doesn't consider directionality)
    del data['gene', 'rev_interacts_with', 'gene']
    del data['cell', 'rev_dependency_of', 'gene']

    # Convert to homogeneous data object
    homo_data = data.to_homogeneous()
    # During conversion, nan values are added due to gene-cell interaction edge
    nan_msk = ~torch.isnan(homo_data.edge_label)

    homo_data.edge_label = homo_data.edge_label[nan_msk]

    return homo_data

def merge_Data(data_gene, data_cell, mother_Data, cell_feat, gene_feat):

    # Generate homogeneous data
    data = mother_Data.clone()

    data_gene = to_homo(data_gene)
    data_cell = to_homo(data_cell)

    # Merge into separate tensor containing both information
    merged_label_index = torch.cat([data_gene.edge_label_index,data_cell.edge_label_index],dim=1)
    merged_index = torch.cat([data_gene.edge_index,data_cell.edge_index],dim=1)
    merged_label = torch.cat([data_gene.edge_label,data_cell.edge_label],dim=0)
    merged_edge_type = torch.cat([data_gene.edge_type,data_cell.edge_type],dim=0)

    # Shuffle, if not labels will be  [1111,,,,00000]
    shuffled_index = torch.randperm(merged_label_index.size(1))
    shuffled_label_index = merged_label_index[:, shuffled_index]
    shuffled_label = merged_label[shuffled_index]

    # Complete homogeneous grpah data object
    data.edge_index = merged_index
    data.edge_label = shuffled_label
    data.edge_label_index = shuffled_label_index
    #data.edge_label = merged_label
    #data.edge_label_index = merged_label_index
    data.edge_type = merged_edge_type
    data.gene_feat = gene_feat
    data.cell_feat = cell_feat

    #gene_feat = gene_feat[:,:cell_feat.size(1)]
    #cell_feat = torch.nn.functional.pad(cell_feat, (0, gene_feat.size(1)-cell_feat.size(1)), "constant", 0)
    #x = torch.cat((gene_feat,cell_feat),dim=0)
    
    #data.x = x
    return data



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='args')

    # general args
    parser.add_argument('--save_path', type=str, default='/project_ghent/GAT_pred/', help='path to save results')
    parser.add_argument('--log', type=int, default=1, help='Whether to log to wandb')
    parser.add_argument('--wandb_user', type=str, default='jilim97', help='wandb username')

    # model and train args
    parser.add_argument('--cancer_type', type=str, default="Neuroblastoma", help='Cancer type to train for')
    parser.add_argument('--drugs', type=int, default="0", help='Use the intergrated graph with drugs and targets')
    parser.add_argument('--ppi', type=str, default="Reactome", help='Which ppi to use as scaffold')
    parser.add_argument('-crp_pos', type=float, default=-1.5, help='crispr threshold for positives')
    parser.add_argument('--epochs', type=int, default=6, help='num epochs')
    parser.add_argument('--npr', type=float, default=3.0, help='Negatiev sampling ratio')
    parser.add_argument('--emb_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--train_neg_sampling', type=int, default=1, help='If 1(true) negatives will be sampled BEFORE training')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='percentage training data')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='percentage test data')
    parser.add_argument('--disjoint_train_ratio', type=float, default=0.0, help='percentage disjoint train data')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--remove_rpl', type=int, default=1, help='removing RPL genes')
    parser.add_argument('--remove_commonE', type=int, default=0, help='removing common essentials')
    parser.add_argument('--useSTD', type=int, default=1, help='removing common essentials')
    parser.add_argument('--save_full_pred', type=int, default=1, help='If you want to save the full (all genes in scaffold) perdiction df')
    parser.add_argument('--plot_cell_embeddings', type=int, default=0, help='If you want to plot the cell embeddings colored by subtype')
    parser.add_argument('--cell_feat', type=str, default='cnv', help='Cell feature name')
    parser.add_argument('--gene_feat', type=str, default='cgp', help='Gene feature name')
    parser.add_argument('--emb_method', type=str, default='emb', help='Embedding method')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    parser.add_argument('--exp_name', type=str, default='emb', help='Experiment Name')

    args = parser.parse_args()
    now = datetime.now()
    args.remove_rpl = "_noRPL" if args.remove_rpl else ""        
    args.remove_commonE = "commonE" if args.remove_commonE else ""
    args.useSTD = "STD" if args.useSTD else "NOSTD"
    args.drugs = "_drugtarget" if args.drugs else ""
    
    seed_everything(args.seed)

    experiment_name = f"{args.exp_name}"
    group_name = 'DLP'

    if args.log:
        run = wandb.init(project="CKPT_Loss", entity=args.wandb_user,  config=args, name=experiment_name, group=group_name) 

    BASE_PATH = "/kyukon/data/gent/vo/000/gvo00095/vsc45456/"

    gpu_available = torch.cuda.is_available()
    print(f"GPU Available: {gpu_available}")
    if gpu_available:
        device = 'cuda:1'
    else:
        device = 'cpu'

    # Read in Heterogeneous graph and ori screening data
    
    heterodata_obj = torch.load(BASE_PATH+f"multigraphs/"\
                                f"heteroData_gene_cell_{args.cancer_type.replace(' ', '_')}_{args.ppi}"\
                                    f"_crispr{str(args.crp_pos).replace('.','_')}{args.drugs}_{args.gene_feat}_{args.cell_feat}.pt")
    
    cells, genes = heterodata_obj['cell'].names, heterodata_obj['gene'].names
    if args.drugs:
        drugs = heterodata_obj['drug'].names
        del heterodata_obj['drug'].names

    cell2int = dict(zip(heterodata_obj['cell'].names, heterodata_obj['cell'].node_id.numpy()))
    gene2int = dict(zip(heterodata_obj['gene'].names, heterodata_obj['gene'].node_id.numpy()))
    dep_genes = list(set(heterodata_obj['gene', 'dependency_of', 'cell'].edge_index[0].numpy()))

    crispr_neurobl = pd.read_csv(BASE_PATH+f"data/crispr_{args.cancer_type.replace(' ', '_')}_{args.ppi}.csv", index_col=0)
    crispr_neurobl_int = crispr_neurobl.copy(deep=True)
    crispr_neurobl_int.index = [cell2int[i] for i in crispr_neurobl.index]
    crispr_neurobl_int.columns = [gene2int[i] for i in crispr_neurobl.columns]
    crispr_neurobl_int = crispr_neurobl_int.loc[:, dep_genes]

    crispr_neurobl_bin = crispr_neurobl_int.applymap(lambda x: int(x < args.crp_pos))
    del heterodata_obj['gene'].names, heterodata_obj['cell'].names

    emb_dim = args.emb_dim
    heterodata_obj_to_homo = heterodata_obj.clone()
    
    del heterodata_obj_to_homo['gene', 'rev_interacts_with', 'gene']
    del heterodata_obj_to_homo['cell', 'rev_dependency_of', 'gene'] # Delete Reverse

    homodata_obj = heterodata_obj_to_homo.to_homogeneous() # Convert to homogeneous graph
    
    # Count numbers of unique node types to filter out cell node 
    node_types = homodata_obj.node_type
    
    unique_node_types, node_type_counts = node_types.unique(return_counts=True)

    gene_num = node_type_counts[0]
    cell_num = node_type_counts[1]

    # Extract node features
    cell_feat = heterodata_obj['cell'].x
    gene_feat = heterodata_obj['gene'].x
    #cell_feat = torch.nn.functional.pad(cell_feat, (0, gene_feat.size(1)-cell_feat.size(1)), "constant", 0)
    #x = torch.cat((gene_feat,cell_feat),dim=0)

    features_dim = {'cell': heterodata_obj['cell'].x.shape[1],
                    'gene': heterodata_obj['gene'].x.shape[1]
                    }

    # node_ids = homodata_obj.node_id
    # gnes_id = node_ids[:gene_num]
    # cell_id = node_ids[gene_num:] + gene_num

    # whole_genes = list(set(gnes_id.numpy()))
    # whole_cells = list(set(cell_id.numpy()))

    # whole_edges = torch.zeros((2, len(whole_cells)*len(whole_genes)), dtype=torch.long)

    # for i, cl in enumerate(whole_cells):
    #     # cl = 20
    #     x_ = torch.stack((torch.tensor(whole_genes),
    #                     torch.tensor([cl]*len(whole_genes))), dim=0)
                        
    #     whole_edges[:, i*len(whole_genes):(i+1)*len(whole_genes)] = x_

    DLP_model = DLP_model(data=homodata_obj,
                                  embedding_dim=emb_dim,
                                  features_dim=features_dim,
                                  embedding_method=args.emb_method,
                                  )

    DLP_model.to(device)
    print(DLP_model)
    # Define the full probability matrix for validation
    cls_int = heterodata_obj['cell'].node_id
    cl_probs = torch.zeros((2, len(cls_int)*len(dep_genes)), dtype=torch.long)

    for i, cl in enumerate(cls_int):
        # cl = 20
        x_ = torch.stack((torch.tensor(dep_genes),
                        torch.tensor([cl]*len(dep_genes))), dim=0)
                        
        cl_probs[:, i*len(dep_genes):(i+1)*len(dep_genes)] = x_
    full_pred_data = homodata_obj.clone()
    full_pred_data.edge_label_index = cl_probs

    full_pred_data.cell_feat = cell_feat
    full_pred_data.gene_feat = gene_feat
    #full_pred_data.x = x

    # Define training parameters
    optimizer = torch.optim.Adam(DLP_model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_loss = np.inf
    epoch_since_best = 0
    n_epochs = args.epochs
    
    # Define parameters for checkpoint
    best_ap = 0 
    best_ap_model = None
    lowest_loss = np.inf
    best_loss_model = None

    #Split graph in train/validation
    # transform_traintest = T.RandomLinkSplit(num_val=args.val_ratio,
    #                                         num_test=args.test_ratio,
    #                                         disjoint_train_ratio=args.disjoint_train_ratio,
    #                                         neg_sampling_ratio=args.npr,
    #                                         add_negative_train_samples=args.train_neg_sampling,
    #                                         is_undirected=False)

    # train_data, val_data, test_data = transform_traintest(homodata_obj)
    # direct_modifier(train_data)
    # direct_modifier(val_data)
    # direct_modifier(test_data)

    transform_traintest_cell = T.RandomLinkSplit(num_val=args.val_ratio,
                                            num_test=args.test_ratio,
                                            disjoint_train_ratio=args.disjoint_train_ratio,
                                            neg_sampling_ratio=3.0,
                                            add_negative_train_samples=args.train_neg_sampling,
                                            edge_types=('gene', 'dependency_of', 'cell'),
                                            rev_edge_types=('cell', 'rev_dependency_of', 'gene'),
                                            is_undirected=True)

    train_data_cell, val_data_cell, test_data_cell = transform_traintest_cell(heterodata_obj)

    # Generate graph obeject to conserve full connection of gene-gene interaction
    train_data_gene = heterodata_obj.clone()

    # Since all edges are connected, make tensor with same size of edge index containing 1 (positive)
    label_len = heterodata_obj['gene', 'interacts_with', 'gene'].edge_index.size(1)
    labels = torch.ones(label_len)

    # Add gene-gene interaction edge label index and its labels for training
    train_data_gene['gene', 'interacts_with', 'gene'].edge_label_index = heterodata_obj['gene', 'interacts_with', 'gene'].edge_index
    train_data_gene['gene', 'interacts_with', 'gene'].edge_label = labels
    
    # Train data will be converted to homogeneous afeter merging (gene-gene + cell-gene)
    # train_data_cell = to_homo(train_data_cell)
    # train_data_gene = to_homo(train_data_gene)
    val_data = to_homo(val_data_cell)
    # val_data_gene = to_homo(val_data_gene)
    test_data = to_homo(test_data_cell)
    # test_data_gene = to_homo(test_data_gene)

    # Merge train data containing each edge type
    train_data = merge_Data(train_data_gene, train_data_cell, homodata_obj, cell_feat, gene_feat)
    #val_data = merge_Data(val_data_gene, val_data_cell, homodata_obj, cell_feat, gene_feat)
    #test_data = merge_Data(test_data_gene, test_data_cell, homodata_obj, cell_feat, gene_feat)

    # Add node features
    # train_data.cell_feat = cell_feat
    # train_data.gene_feat = gene_feat
    val_data.cell_feat = cell_feat
    val_data.gene_feat = gene_feat
    test_data.cell_feat = cell_feat
    test_data.gene_feat = gene_feat

    # Define the loaders
    if args.train_neg_sampling:
        train_loader = LinkNeighborLoader(data=train_data,
                                        num_neighbors=[-1]*2, 
                                        edge_label_index=train_data.edge_label_index,
                                            edge_label=train_data.edge_label,
                                            batch_size=args.batch_size, # how many pos per batch -> actual batch_size is (npr+1)*batch_size
                                            directed=True, # undirected het graphs not yet supported -> that is why the reverse type is added
                                            shuffle=True,
                                            num_workers=10)
    else:
        train_loader = LinkNeighborLoader(data=train_data,
                                        num_neighbors=[-1]*2, 
                                        neg_sampling_ratio=args.npr,
                                        edge_label_index=train_data.edge_label_index,
                                            edge_label=train_data.edge_label,
                                            batch_size=args.batch_size, # how many pos per batch -> actual batch_size is (npr+1)*batch_size
                                            directed=True, # undirected het graphs not yet supported -> that is why the reverse type is added
                                            shuffle=True,
                                            num_workers=10)

    # val_loader = LinkNeighborLoader(data=val_data,
    #                                 num_neighbors={et: [-1]*2 for et in heterodata_obj.edge_types},
    #                                 edge_label_index=(("gene", "dependency_of", "cell"), 
    #                                                   val_data["gene", "dependency_of", "cell"].edge_label_index),
    #                                 edge_label=val_data["gene", "dependency_of", "cell"].edge_label,
    #                                 batch_size=args.batch_size,
    #                                 shuffle=True,) 

    # Train the model but first delete the names 
    assay_ap_total, gene_ap_total = [], []
    for epoch in range(n_epochs):
        # epoch = 0
        total_train_loss = 0
        DLP_model.train()
        for sampled_data in train_loader:
            # sampled_data = next(iter(train_loader))
            optimizer.zero_grad()
            sampled_data.to(device)
            
            out = DLP_model(sampled_data)

            ground_truth = sampled_data.edge_label
            loss = loss_fn(out, ground_truth) 
            total_train_loss += loss
            loss.backward()
            optimizer.step()

        ap_val, auc_val = 0, 0
        DLP_model.eval()
        with torch.no_grad():
            if args.val_ratio != 0.0:
                val_data.to(device)

                out = DLP_model(val_data)
                pred = torch.sigmoid(out)

                ground_truth = val_data.edge_label
                val_loss = loss_fn(out, ground_truth)

                auc_val = roc_auc_score(ground_truth.cpu(), pred.cpu())
                ap_val = average_precision_score(ground_truth.cpu(), pred.cpu())
                
                # CKPT for best loss
                if val_loss < lowest_loss:
                    lowest_loss = val_loss
                    best_loss_model = deepcopy(DLP_model.state_dict())
                    final_epoch = epoch
                
                # CKPT for best AP
                # if ap_val > best_ap:
                #     best_ap = ap_val
                #     best_ap_model = deepcopy(DLP_model.state_dict())
                #     final_epoch = epoch

            full_pred_data.to(device)
            total_preds = DLP_model(data=full_pred_data)
            total_preds_out = torch.sigmoid(total_preds).cpu().numpy()
            tot_pred_deps = construct_complete_predMatrix(total_predictions=total_preds_out,
                                                        edge_index=cl_probs,
                                                        index=cls_int.numpy(),
                                                        columns=dep_genes)

            assay_corr = tot_pred_deps.corrwith(crispr_neurobl_int*-1, method='spearman', axis=1)
            gene_ap, assay_ap = [], []
            for i, row in tot_pred_deps.iterrows():
                assay_ap.append(average_precision_score(y_true=crispr_neurobl_bin.loc[i].values,
                                                        y_score=row.values))
            for col in tot_pred_deps.columns:
                gene_ap.append(average_precision_score(y_true=crispr_neurobl_bin[col].values,
                                                        y_score=tot_pred_deps[col].values))
            
            assay_ap_total.append(assay_ap)
            gene_ap_total.append(gene_ap)


        if args.log:
            if args.val_ratio != 0.0:
                run.log({'epoch': epoch, 'train loss': total_train_loss/len(train_loader),
                        'val loss': val_loss, 'val auc': auc_val, 'val ap': ap_val,
                        'assay_ap': np.mean(assay_ap), 'gene_ap': np.mean(gene_ap),
                        'assay_corr_sp': assay_corr.mean()})
            else:
                run.log({'epoch': epoch, 'train loss': total_train_loss/len(train_loader),
                        'assay_ap': np.mean(assay_ap), 'gene_ap': np.mean(gene_ap),
                        'assay_corr_sp': assay_corr.mean()})

    # Save model
    path = BASE_PATH + f'Model/{args.exp_name}-{args.cell_feat}-{args.seed}-{final_epoch}.pt'
    # torch.save(best_ap_model, path)
    torch.save(best_loss_model, path)

    if args.test_ratio != 0.0:
        test_data.to(device)
        
        # Model Load
        DLP_model.load_state_dict(torch.load(path))
        out = DLP_model(test_data)

        pred = torch.sigmoid(out).detach().cpu()
        ground_truth = test_data.edge_label.detach().cpu()

        index = test_data.edge_label_index.detach().cpu()

        test_gene_ap, test_assay_ap = [], []

        for cell in set(index[1]):
            assay_msk = index[1] == cell    
            assay_msk.cpu() 
            test_assay_ap.append(average_precision_score(y_true=ground_truth[assay_msk],
                                                        y_score=pred[assay_msk]))

        # Filter out the list that only contain TN   
        for gene in set(index[0]):
            gene_msk = index[0] == gene   
            gene_msk.cpu()  
            if ground_truth[gene_msk].sum() + pred[gene_msk].sum() > 0.5:      
                test_gene_ap.append(average_precision_score(y_true=ground_truth[gene_msk],
                                                            y_score=pred[gene_msk]))   

        ap_test = average_precision_score(ground_truth, pred)

        run.log({"test AP": ap_test, "test gene AP": np.mean(test_gene_ap), "test assay AP": np.mean(test_assay_ap),})

    # Calculate top prediction and send to Kaat
    cl_probs = torch.zeros((2, len(cls_int)*heterodata_obj['gene'].num_nodes), dtype=torch.long)

    for i, cl in enumerate(cls_int):
        # cl = 20
        x_ = torch.stack((heterodata_obj['gene'].node_id,
                        torch.tensor([cl]*heterodata_obj['gene'].num_nodes)), dim=0)
        cl_probs[:, i*heterodata_obj['gene'].num_nodes:(i+1)*heterodata_obj['gene'].num_nodes] = x_

    full_pred_data_all = homodata_obj.clone()
    full_pred_data_all.edge_label_index = cl_probs

    full_pred_data.cell_feat = cell_feat
    full_pred_data.gene_feat = gene_feat

    full_pred_data_all.to('cpu')
    DLP_model.to('cpu')
    DLP_model.eval()
    with torch.no_grad():
        out_full_all, emb = DLP_model(data=full_pred_data_all, return_embeddings = True)
        preds_full_all = torch.sigmoid(out_full_all).cpu().numpy()

    embs = {}

    embs['gene'] = tensor_zeros = torch.zeros((int(node_type_counts[0]), emb_dim))
    embs['cell'] = tensor_zeros = torch.zeros((int(node_type_counts[1]), emb_dim))

    for idx, feat in enumerate(emb):
        if idx > (gene_num-1):
            embs['cell'][idx-gene_num-1] = feat
        else:
            embs['gene'][idx] = feat
    
    # gene_embeddings = hetGNNmodel.nt1_emb.weight.cpu().detach().numpy()
    gene_embs_df = pd.DataFrame(data=embs['gene'].cpu().detach().numpy(), index=genes)

    # cell_embeddings = hetGNNmodel.nt2_emb.weight.cpu().detach().numpy()
    cell_embs_df = pd.DataFrame(data=embs['cell'].cpu().detach().numpy(), index=cells)
    # cell_embs_df_copy = pd.DataFrame(data=cell_embeddings, index=cells)
    
    gene_embs_df.to_csv(BASE_PATH+f"results/"\
                        f"{args.cancer_type.replace(' ', '_')}_{args.ppi}{args.remove_rpl}_{args.useSTD}{args.remove_commonE}_crispr{str(args.crp_pos).replace('.','_')}_DLP_gene_embs.csv")
    cell_embs_df.to_csv(BASE_PATH+f"results/"\
                        f"{args.cancer_type.replace(' ', '_')}_{args.ppi}{args.remove_rpl}_{args.useSTD}{args.remove_commonE}_crispr{str(args.crp_pos).replace('.','_')}_DLP_cell_embs.csv")

    if args.drugs:
        drug_embs_df = pd.DataFrame(data=embs['drug'].cpu().detach().numpy(), index=drugs)
        drug_embs_df.to_csv(BASE_PATH+f"results/"\
                            f"{args.cancer_type.replace(' ', '_')}_{args.ppi}{args.remove_rpl}_{args.useSTD}{args.remove_commonE}_crispr{str(args.crp_pos).replace('.','_')}_HetGNN_drug_embs_{args.gene_feat}_{args.cell_feat}_{args.layer_name}.csv")
    
    if args.save_full_pred:
        tot_pred_deps = construct_complete_predMatrix(total_predictions=preds_full_all,
                                                    edge_index=cl_probs, index=cls_int.numpy(),
                                                    columns=heterodata_obj['gene'].node_id.numpy())

        tot_pred_deps.to_csv(BASE_PATH+f"results/"\
                            f"{args.cancer_type.replace(' ', '_')}_{args.ppi}{args.remove_rpl}_{args.useSTD}{args.remove_commonE}_crispr{str(args.crp_pos).replace('.','_')}_DLPGNN.csv")
        


        

