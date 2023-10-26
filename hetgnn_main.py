from gat_dependency.utils import read_h5py
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch.functional as F
import torch
import pickle 
from gat_dependency.GAT_model import HeteroData_GNNmodel
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import os
import argparse
import wandb
from datetime import datetime
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

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

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='args')

    # general args
    parser.add_argument('--save_path', type=str, default='/project_ghent/GAT_pred/', help='path to save results')
    parser.add_argument('--log', type=int, default=0, help='Whether to log to wandb')
    parser.add_argument('--wandb_user', type=str, default='one-team', help='wandb username')

    # model and train args
    parser.add_argument('--cancer_type', type=str, default="Neuroblastoma", help='Cancer type to train for')
    parser.add_argument('--drugs', type=int, default="1", help='Use the intergrated graph with drugs and targets')
    parser.add_argument('--ppi', type=str, default="Reactome", help='Which ppi to use as scaffold')
    parser.add_argument('-crp_pos', type=float, default=-1.5, help='crispr threshold for positives')
    parser.add_argument('--epochs', type=int, default=3, help='num epochs')
    parser.add_argument('--npr', type=float, default=3.0, help='Negatiev sampling ratio')
    parser.add_argument('--emb_dim', type=str, default="512", help='Embedding dimension')
    parser.add_argument('--train_neg_sampling', type=int, default=1, help='If 1(true) negatives will be sampled BEFORE training')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='percentage training data')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='percentage test data')
    parser.add_argument('--disjoint_train_ratio', type=float, default=0.0, help='percentage disjoint train data')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--hidden_features', type=str, default='-1,256,128', help='How many hidden features for each GNN layer')
    parser.add_argument('--patience', type=int, default=5, help='patience before breaking out of loop')
    parser.add_argument('--layer_name', type=str, default='GATv2', help='Which gnn layer to use in the model')
    parser.add_argument('--gcn_model', type=str, default='gat', help='which GNN model to use')
    parser.add_argument('--lp_model', type=str, default='simple', help='which LP model to use')
    parser.add_argument('--remove_rpl', type=int, default=1, help='removing RPL genes')
    parser.add_argument('--remove_commonE', type=int, default=0, help='removing common essentials')
    parser.add_argument('--useSTD', type=int, default=1, help='removing common essentials')
    parser.add_argument('--save_full_pred', type=int, default=1, help='If you want to save the full (all genes in scaffold) perdiction df')
    parser.add_argument('--plot_cell_embeddings', type=int, default=0, help='If you want to plot the cell embeddings colored by subtype')
    parser.add_argument('--heads', type=str, default='1,1', help='Number of multiheads to use per GATlayer, must be same length as hidden features')

    args = parser.parse_args()
    now = datetime.now()
    args.remove_rpl = "_noRPL" if args.remove_rpl else ""        
    args.remove_commonE = "commonE" if args.remove_commonE else ""
    args.useSTD = "STD" if args.useSTD else "NOSTD"
    args.drugs = "_drugtarget" if args.drugs else ""

    experiment_name = now.strftime("%m-%d-%Y;%H:%M:%S")

    if args.log:
        run = wandb.init(project="hetgnn", entity=args.wandb_user, config=args, name=experiment_name, group=f"{args.cancer_type}_Drug") 

    BASE_PATH = "/home/bioit/pstrybol/GNN_sensprediction_Marija/"
    # BASE_PATH = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])

    gpu_available = torch.cuda.is_available()
    print(f"GPU Available: {gpu_available}")
    if gpu_available:
        device = 'cuda'
    else:
        device = 'cpu'

    # Read in Heterogeneous graph and ori screening data
    heterodata_obj = torch.load(BASE_PATH+f"multigraphs/"\
                                f"heteroData_gene_cell_{args.cancer_type.replace(' ', '_')}_{args.ppi}"\
                                    f"_crispr{str(args.crp_pos).replace('.','_')}{args.drugs}_cgp_cnv.pt")
    
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

# with open(BASE_PATH+f"multigraphs/{args.cancer_type}_{args.ppi}{args.remove_rpl}_{args.useSTD}{args.remove_commonE}_crispr{str(args.crispr_threshold_pos).replace('.','_')}.pickle", 'rb') as handle:
#     mg_obj = pickle.load(handle)

    # Defining the model
    hidden_features = [int(i) for i in args.hidden_features.split(',')]
    if hidden_features[0] == -1:
        if args.drugs:
            hidden_features[0] = (-1, -1, -1)
        else:
            hidden_features[0] = (-1, -1)


    heads = [int(i) for i in args.heads.split(',')]
    if args.drugs:
        node_types = ['gene', 'cell', 'drug']
        features_dim = {'gene': heterodata_obj['gene'].x.shape[1],
                        'cell': heterodata_obj['cell'].x.shape[1],
                        'drug': heterodata_obj['drug'].x.shape[1]}
    else:
        node_types = ['gene', 'cell']
        features_dim = {'gene': heterodata_obj['gene'].x.shape[1],
                        'cell': heterodata_obj['cell'].x.shape[1]}

    emb_dim = args.emb_dim.split(',')
    if len(emb_dim) > 1:
        emb_dim = {nt: ed for nt, ed in zip(node_types, emb_dim)}
    else:
        emb_dim = int(args.emb_dim)
    
    hetGNNmodel = HeteroData_GNNmodel(heterodata=heterodata_obj,
                                  node_types=node_types,
                                  embedding_dim=emb_dim,
                                  gcn_model=args.gcn_model,
                                  features=hidden_features,
                                  layer_name=args.layer_name,
                                  heads=heads,
                                  dropout=args.dropout,
                                  act_fn=torch.nn.ReLU(),
                                  lp_model=args.lp_model,
                                  add_self_loops=False, 
                                  features_dim=features_dim,
                                  return_attention_weights=False)

    hetGNNmodel.to(device)
    print(hetGNNmodel)
    # Define the full probability matrix for validation
    cls_int = heterodata_obj['cell'].node_id
    cl_probs = torch.zeros((2, len(cls_int)*len(dep_genes)), dtype=torch.long)

    for i, cl in enumerate(cls_int):
        # cl = 20
        x_ = torch.stack((torch.tensor(dep_genes),
                        torch.tensor([cl]*len(dep_genes))), dim=0)
                        
        cl_probs[:, i*len(dep_genes):(i+1)*len(dep_genes)] = x_
    full_pred_data = heterodata_obj.clone()
    full_pred_data['gene', 'dependency_of', 'cell'].edge_label_index = cl_probs

    # Define training parameters
    optimizer = torch.optim.Adam(hetGNNmodel.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    patience = args.patience
    best_loss = np.inf
    epoch_since_best = 0
    n_epochs = args.epochs

    # Split graph in train/validation
    transform_traintest = T.RandomLinkSplit(num_val=args.val_ratio,
                                            num_test=args.test_ratio,
                                            disjoint_train_ratio=args.disjoint_train_ratio,
                                            neg_sampling_ratio=args.npr,
                                            add_negative_train_samples=args.train_neg_sampling,
                                            edge_types=('gene', 'dependency_of', 'cell'),
                                            rev_edge_types=('cell', 'rev_dependency_of', 'gene'),
                                            is_undirected=True)

    train_data, val_data, test_data = transform_traintest(heterodata_obj)

    # Define the loaders
    if args.train_neg_sampling:
        train_loader = LinkNeighborLoader(data=train_data,
                                        num_neighbors={et: [-1]*2 for et in heterodata_obj.edge_types}, 
                                        edge_label_index=(("gene", "dependency_of", "cell"),
                                                            train_data["gene", "dependency_of", "cell"].edge_label_index),
                                            edge_label=train_data["gene", "dependency_of", "cell"].edge_label,
                                            batch_size=args.batch_size, # how many pos per batch -> actual batch_size is (npr+1)*batch_size
                                            directed=True, # undirected het graphs not yet supported -> that is why the reverse type is added
                                            shuffle=True,
                                            num_workers=10)
    else:
        train_loader = LinkNeighborLoader(data=train_data,
                                        num_neighbors={et: [-1]*2 for et in heterodata_obj.edge_types}, 
                                        neg_sampling_ratio=args.npr,
                                        edge_label_index=(("gene", "dependency_of", "cell"),
                                                            train_data["gene", "dependency_of", "cell"].edge_label_index),
                                            edge_label=train_data["gene", "dependency_of", "cell"].edge_label,
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
        hetGNNmodel.train()
        for sampled_data in train_loader:
            # sampled_data = next(iter(train_loader))
            optimizer.zero_grad()
            sampled_data.to(device)
            if args.gcn_model == 'gat':
                out = hetGNNmodel(sampled_data, edge_type_label="gene,dependency_of,cell")
            else:
                out = hetGNNmodel(sampled_data, edge_type_label="gene,dependency_of,cell")
            ground_truth = sampled_data["gene", "dependency_of", "cell"].edge_label
            loss = loss_fn(out, ground_truth)
            total_train_loss += loss
            loss.backward()
            optimizer.step()

        ap_val, auc_val = 0, 0
        hetGNNmodel.eval()
        with torch.no_grad():
            if args.val_ratio != 0.0:
                val_data.to(device)
                if args.gcn_model == 'gat':
                    out, attw = hetGNNmodel(val_data, edge_type_label="gene,dependency_of,cell")
                else:
                    out = hetGNNmodel(val_data, edge_type_label="gene,dependency_of,cell")
                pred = torch.sigmoid(out)
                ground_truth = val_data["gene", "dependency_of", "cell"].edge_label
                val_loss = loss_fn(out, ground_truth)

                auc_val = roc_auc_score(ground_truth.cpu(), pred.cpu())
                ap_val = average_precision_score(ground_truth.cpu(), pred.cpu())

            full_pred_data.to(device)
            total_preds = hetGNNmodel(data=full_pred_data, edge_type_label="gene,dependency_of,cell")
            tot_pred_deps = construct_complete_predMatrix(total_predictions=total_preds.cpu().numpy(),
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

        if args.val_ratio != 0.0:
            if val_loss < best_loss:
                best_loss = val_loss
                epoch_since_best = 0
            else:
                epoch_since_best += 1
            
            if epoch_since_best == patience:
                print(f"Breaking out at epoch {epoch}")
                break

    if args.test_ratio != 0.0:
        test_data.to(device)
        
        out = hetGNNmodel(test_data, edge_type_label="gene,dependency_of,cell")
        pred = torch.sigmoid(out)
        ground_truth = test_data["gene", "dependency_of", "cell"].edge_label

        ap_test = average_precision_score(ground_truth.detach().cpu(), pred.detach().cpu())

        run.log({"test AP": ap_test})

    # Calculate top prediction and send to Kaat
    cl_probs = torch.zeros((2, len(cls_int)*heterodata_obj['gene'].num_nodes), dtype=torch.long)

    for i, cl in enumerate(cls_int):
        # cl = 20
        x_ = torch.stack((heterodata_obj['gene'].node_id,
                        torch.tensor([cl]*heterodata_obj['gene'].num_nodes)), dim=0)
        cl_probs[:, i*heterodata_obj['gene'].num_nodes:(i+1)*heterodata_obj['gene'].num_nodes] = x_

    full_pred_data_all = heterodata_obj.clone()
    full_pred_data_all['gene', 'dependency_of', 'cell'].edge_label_index = cl_probs

    # full_pred_data_all_loader = LinkNeighborLoader(data=full_pred_data_all,
    #                                                num_neighbors={et: [-1]*2 for et in heterodata_obj.edge_types}, 
    #                                                neg_sampling_ratio=args.npr,
    #                                                edge_label_index=(("gene", "dependency_of", "cell"),
    #                                                                  full_pred_data_all["gene", "dependency_of", "cell"].edge_label_index),
    #                                                 batch_size=args.batch_size, # how many pos per batch -> actual batch_size is (npr+1)*batch_size
    #                                                 directed=True, # undirected het graphs not yet supported -> that is why the reverse type is added
    #                                                 shuffle=True,
    #                                                 num_workers=10)
    
    full_pred_data_all.to('cpu')
    hetGNNmodel.to('cpu')
    hetGNNmodel.eval()
    with torch.no_grad():
        out_full_all, embs = hetGNNmodel(data=full_pred_data_all, edge_type_label="gene,dependency_of,cell", return_embeddings=True)
        preds_full_all = torch.sigmoid(out_full_all).cpu().numpy()
    
    # gene_embeddings = hetGNNmodel.nt1_emb.weight.cpu().detach().numpy()
    gene_embs_df = pd.DataFrame(data=embs['gene'].cpu().detach().numpy(), index=genes)

    # cell_embeddings = hetGNNmodel.nt2_emb.weight.cpu().detach().numpy()
    cell_embs_df = pd.DataFrame(data=embs['cell'].cpu().detach().numpy(), index=cells)
    # cell_embs_df_copy = pd.DataFrame(data=cell_embeddings, index=cells)
    
    gene_embs_df.to_csv(BASE_PATH+f"results/hetgnn/"\
                        f"{args.cancer_type.replace(' ', '_')}_{args.ppi}{args.remove_rpl}_{args.useSTD}{args.remove_commonE}_crispr{str(args.crp_pos).replace('.','_')}_HetGNN_gene_embs{args.drugs}_cgp_cnv.csv")
    cell_embs_df.to_csv(BASE_PATH+f"results/hetgnn/"\
                        f"{args.cancer_type.replace(' ', '_')}_{args.ppi}{args.remove_rpl}_{args.useSTD}{args.remove_commonE}_crispr{str(args.crp_pos).replace('.','_')}_HetGNN_cell_embs{args.drugs}_cgp_cnv.csv")

    if args.drugs:
        drug_embs_df = pd.DataFrame(data=embs['drug'].cpu().detach().numpy(), index=drugs)
        drug_embs_df.to_csv(BASE_PATH+f"results/hetgnn/"\
                            f"{args.cancer_type.replace(' ', '_')}_{args.ppi}{args.remove_rpl}_{args.useSTD}{args.remove_commonE}_crispr{str(args.crp_pos).replace('.','_')}_HetGNN_drug_embs_cgp_cnv.csv")
    
    if args.save_full_pred:
        tot_pred_deps = construct_complete_predMatrix(total_predictions=preds_full_all,
                                                    edge_index=cl_probs, index=cls_int.numpy(),
                                                    columns=heterodata_obj['gene'].node_id.numpy())

        tot_pred_deps.to_csv(BASE_PATH+f"results/hetgnn/"\
                            f"{args.cancer_type.replace(' ', '_')}_{args.ppi}{args.remove_rpl}_{args.useSTD}{args.remove_commonE}_crispr{str(args.crp_pos).replace('.','_')}_HetGNN{args.drugs}_cgp_cnv.csv")
        

    if args.plot_cell_embeddings:
        # Plot embeddings
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, verbose=1)
        tsne_results = tsne.fit_transform(cell_embs_df.values)
        tsne_results = pd.DataFrame(data=tsne_results, index=cell_embs_df.index, columns=["dim1", "dim2"])    
        ccles = pd.read_csv(BASE_PATH+"data/Model.csv", header=0, index_col=0)
        if '_' in args.cancer_type:
            tsne_results['cancer_type'] = ccles.loc[cells, 'OncotreePrimaryDisease']
        else:
            tsne_results['cancer_type'] = ccles.loc[cells, 'OncotreeSubtype']
        
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.scatterplot(
            x="dim1", y="dim2",
            hue="cancer_type",
            palette=sns.color_palette("colorblind", tsne_results['cancer_type'].unique().shape[0]),
            data=tsne_results,
            legend="full",
            alpha=0.8,
            ax=ax
        )
        plt.title(f"{args.cancer_type} clustering after {epoch} epochs")
        # plt.show()
        # plt.savefig(BASE_PATH+f"Figures/hetGNN/{epoch}_GAT_loss_auc")
        # Log the plot
        wandb.log({"plot TSNE embeddings": wandb.Image(fig)})
        plt.close()

        

