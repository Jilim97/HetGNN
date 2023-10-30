import h5py
import random
import torch
import json
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import pdb
import random
import torch.nn as nn
import torch_geometric.nn as geom_nn
from collections import OrderedDict
from torch_geometric.loader.dataloader import DataLoader
from scipy.stats import hypergeom, spearmanr, pearsonr
import gseapy as gp
from statsmodels.stats.multitest import multipletests
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_mutual_info_score
from itertools import combinations

def train_loop(dataloader, model, loss_fn, optimizer, name_layer, device, return_embs=False,
               verbose=True, return_alphas=False):
    num_batches = len(dataloader)
    aps, aucs = [], []
    epoch_loss = 0
    alphas_temp = {}
    single_sample = False if isinstance(dataloader, DataLoader) else True
    embeddings = {}

    if return_alphas:
        alphas = {}
        def getAlphas(name):
            def hook(model, input, output):
                alphas_temp[name] = output[1]
            return hook

        for i, name in enumerate(model.gnn_layers._names):
            if 'gat' in name:
                model.gnn_layers[i].register_forward_hook(getAlphas(name))

    if single_sample:
        dataloader = dataloader.to(device)
        embs, out = model(dataloader)
        loss = loss_fn(out, dataloader.y)
        epoch_loss += loss.item()

        # pdb.set_trace()
        if device == 'cuda':
            out_np = torch.sigmoid(out).cpu().detach().numpy()
        else:
            out_np = torch.sigmoid(out).detach().numpy()
        

        aps.append(average_precision_score(y_true=dataloader.y.cpu(), y_score=out_np))
        aucs.append(roc_auc_score(y_true=dataloader.y.cpu(), y_score=out_np))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        for batch, graph in enumerate(dataloader):
            graph = graph.to(device)
            embs, out = model(graph)
            loss = loss_fn(out, graph.y)
            epoch_loss += loss.item()
            # pdb.set_trace()
            if device == 'cuda':
                out_np = torch.sigmoid(out).cpu().detach().numpy()
            else:
                out_np = torch.sigmoid(out).detach().numpy()
            aps.append(average_precision_score(y_true=graph.y.cpu(), y_score=out_np))
            aucs.append(roc_auc_score(y_true=graph.y.cpu(), y_score=out_np))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if return_alphas:
                for i in range(graph.ptr.shape[0] - 1):
                    alphas[graph.name[i]] = OrderedDict([(k, v[1][graph.ptr[i]:graph.ptr[i + 1]])
                                                         for k, v in alphas_temp.items()])

            if return_embs:
                for i in range(graph.ptr.shape[0] - 1):
                    embeddings[graph.name[i]] = embs[graph.ptr[i]:graph.ptr[i + 1], :]

        # if batch % 10 == 0:
        #     loss, current = loss.item(), (batch+1) * len(graph)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    if verbose:
        if single_sample:
            print(f"TRAINING | Epoch loss: {epoch_loss:.3f} "
                  f"Avg AP: {np.mean(aps) * 100:.3f}% - Avg AUC: {np.mean(aucs) * 100:.3f}%")
        else:
            print(f"TRAINING | Epoch loss: {epoch_loss/num_batches:.3f} "
                  f"Avg AP: {np.mean(aps)*100:.3f}% - Avg AUC: {np.mean(aucs)*100:.3f}%")

    if return_alphas and return_embs:
        return epoch_loss/num_batches, np.mean(aps), np.mean(aucs), alphas, embeddings
    else:
        return epoch_loss/num_batches, np.mean(aps), np.mean(aucs)


def train_loop_baseline(sample, y, model, loss_fn, optimizer, device,
                        return_embs=False, extra_features=None,
                        verbose=True, calculate_performance=True,
                        exclude_genes=None, exclude_loss=False,
                        exp_loss=False, gamma=2, edge_embed=False):

    sample = sample.to(device)
    if extra_features is not None:
        if return_embs:
            out, embs = model(sample, extra_features, edge_embed)
        else:
            out = model(sample, extra_features, edge_embed)
    else:
        if return_embs:
            out, embs = model(sample, extra_features, edge_embed)
        else:
            out = model(sample, extra_features, edge_embed)

    if exclude_loss:
        ix = ~np.in1d(np.arange(out.shape[0]), np.array(exclude_genes))
        loss = loss_fn(out[ix], y[ix])
    else:
        loss = loss_fn(out, y)

    if exp_loss:
        loss = torch.exp(loss)
        # loss = loss**(gamma+2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if calculate_performance:
        # pdb.set_trace()
        if device == 'cuda':
            out_np = torch.sigmoid(out).cpu().detach().numpy()
        else:
            out_np = torch.sigmoid(out).detach().numpy()

        if exclude_genes is not None:
            ix = ~np.in1d(np.arange(out_np.shape[0]), np.array(exclude_genes))
            aps = average_precision_score(y_true=y.cpu()[ix], y_score=out_np[ix])
            aucs = roc_auc_score(y_true=y.cpu()[ix], y_score=out_np[ix])
        else:
            aps = average_precision_score(y_true=y.cpu(), y_score=out_np)
            aucs = roc_auc_score(y_true=y.cpu(), y_score=out_np)

        if verbose:
            print(f"TRAINING | Epoch loss: {loss.item():.3f} "
                    f"Avg AP: {np.mean(aps) * 100:.3f}% - Avg AUC: {np.mean(aucs) * 100:.3f}%")

        if return_embs:
            return loss.item(), aps, aucs, embs, out_np
        else:
            return loss.item(), aps, aucs, out_np
    else:
        return loss.item(), out.detach()



def test_loop(dataloader, model, loss_fn, device,
              name_layer, test=False, verbose=True):
    num_batches = len(dataloader)
    test_loss = 0
    aps, aucs = [], []

    with torch.no_grad():
        for batch, graph in enumerate(dataloader):
            graph = graph.to(device)
            if name_layer == 'GAT':
                # out, (_, _) = model(graph)
                embs, out = model(graph)
            else:
                out = model(graph)

            test_loss += loss_fn(out, graph.y).item()

            pred = torch.sigmoid(out)
            if device == 'cuda':
                pred = pred.cpu().detach().numpy()
            else:
                pred = pred.detach().numpy()

            aps.append(average_precision_score(y_true=graph.y.cpu(), y_score=pred))
            aucs.append(roc_auc_score(y_true=graph.y.cpu(), y_score=pred))
    if verbose:
        if test:
            print(f"TEST | Avg AP: {np.mean(aps) * 100:.3f}% - Avg AUC: {np.mean(aucs) * 100:.3f}%")
        else:
            print(f"VALIDATION | Epoch loss: {test_loss / num_batches:.3f} "
                  f"Avg AP: {np.mean(aps) * 100:.3f}% - Avg AUC: {np.mean(aucs) * 100:.3f}%")
    return test_loss/num_batches, np.mean(aps), np.mean(aucs)



def test_loop_baseline(sample, y, model, loss_fn, device,
                       test=False, verbose=True, 
                       extra_features=None, calculate_performance=True, 
                       crispr_effect=None, focus2int=None,
                       cl=None, return_embs=False, exclude_genes=None,
                       exclude_loss=False,
                       exp_loss=False, gamma=2, edge_embed=False):
    
    sample = sample.to(device)
        
    if extra_features is not None:
        if return_embs:
            out, embs = model(sample, extra_features, edge_embed)
        else:
            out = model(sample, extra_features, edge_embed)
    else:
        if return_embs:
            out, embs = model(sample, extra_features, edge_embed)
        else:
            out = model(sample, extra_features, edge_embed)

    # y = y[sample]
    if exclude_loss:
        ix = ~np.in1d(np.arange(out.shape[0]), np.array(exclude_genes))
        loss = loss_fn(out[ix], y[ix])
    else:
        loss = loss_fn(out, y)
    
    if exp_loss:
        loss = torch.exp(loss)
        # loss = loss**(gamma+2)

    if calculate_performance:
        pred = torch.sigmoid(out)
        if device == 'cuda':
            pred = pred.cpu().detach().numpy()
        else:
            pred = pred.detach().numpy()

        if exclude_genes is not None:
            ix = ~np.in1d(np.arange(pred.shape[0]), np.array(exclude_genes))
            ap = average_precision_score(y_true=y.cpu()[ix], y_score=pred[ix])
            auc = roc_auc_score(y_true=y.cpu()[ix], y_score=pred[ix])
        else:
            ap = average_precision_score(y_true=y, y_score=pred)
            auc = roc_auc_score(y_true=y, y_score=pred)

    if calculate_performance:
        if verbose:
            if test:
                print(f"TEST | Avg AP: {ap * 100:.3f}% - Avg AUC: {auc * 100:.3f}%")
            else:
                print(f"VALIDATION | Epoch loss: {loss:.3f} "
                    f"Avg AP: {ap * 100:.3f}% - Avg AUC: {auc * 100:.3f}%")
        return loss, ap, auc, torch.sigmoid(out)
    else:
        return loss, out.detach()



def make_paramfile(train_fraction, val_fraction, in_features, hidden_features, layer_name,
                   heads, epochs, learning_rate, loss_fn, optimizer, filename):
    param_dict = {
        'train_fraction': train_fraction,
        'val_fraction': val_fraction,
        'in_features': in_features,
        'hidden_features': hidden_features,
        'layer_name': layer_name,
        'heads': heads,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'loss_fn': loss_fn,
        'optimizer': optimizer
    }
    with open(filename, "w") as outfile:
        json.dump(param_dict, outfile)


def gat_layer(in_features, hidden_features, act_fn, dropout, ix, layer_name, layer, heads=None, add_self_loops=True):
    if layer_name == 'GAT':
        # return OrderedDict([
        #     (f'gat_{ix}', (layer(in_channels=in_features,
        #                          out_channels=hidden_features,
        #                          heads=heads,
        #                          add_self_loops=add_self_loops), 'x, edge_index, return_attention_weights=return_attention_weights -> x, alpha')),
        #     (f'act_fn_{ix}', (act_fn, 'x -> x')),
        #     (f'dropout_{ix}', (nn.Dropout(p=dropout), 'x -> x'))])
        return OrderedDict([
            (f'gat_{ix}', (layer(in_channels=in_features,
                                 out_channels=hidden_features,
                                 heads=heads,
                                 add_self_loops=add_self_loops), 'x, edge_index -> x')),
            (f'act_fn_{ix}', (act_fn, 'x -> x')),
            (f'dropout_{ix}', (nn.Dropout(p=dropout), 'x -> x'))])
    else:
        return OrderedDict([
            (f'gat_{ix}', (layer(in_channels=in_features,
                                        out_channels=hidden_features), 'x, edge_index -> x')),
            (f'act_fn_{ix}', (act_fn, 'x -> x')),
            (f'dropout_{ix}', (nn.Dropout(p=dropout), 'x -> x'))])
        # return [(layer(in_channels=in_features, out_channels=hidden_features), 'x, edge_index -> x'),
                # act_fn, nn.Dropout(p=dropout)]


def dense_layer(in_features, out_features, act_fn, dropout, ix):
    return OrderedDict([
        (f'dense_{ix}', nn.Linear(in_features=in_features, out_features=out_features)),
        (f'act_fn_{ix}', act_fn),
        (f'drop_{ix}', nn.Dropout(p=dropout))
    ])


def model_baseline_train_validation(input, model, loss_fn, optimizer, device, verbose, return_embs,
                                    epochs, log, validation, early_stop, patience, run, train_cls,
                                    train_xy, val_cls=None, val_xy=None, save_model_name=None, 
                                    crispr_effect=None, focus_int2gene=None, include_extra_features=True,
                                    exclude_genes=None, exclude_loss=False, equal_training=False, npr=5,
                                    exclude_negatives=None):
    total_train_loss, total_val_loss = [], []
    total_train_auc, total_val_auc = [], []
    total_train_ap, total_val_ap = [], []
    total_train_spr, total_val_spr = [], []
    prob_vectors = {}
    patience_ = 0
    if exclude_loss:
        print("Excluding genes that are always 0/1 from loss calculation")
    for ep in range(epochs):
        # ep = 0
        print(f"Epoch {ep+1}\n ---------------------------------------------")
        train_loss, val_loss = [], []
        train_auc, val_auc = [], []
        train_ap, val_ap = [], []
        train_spr, val_spr = [], []
        model.train()
        random.shuffle(train_cls)
        for name in train_cls:
            x, y = train_xy[name][0], train_xy[name][1]

            if equal_training:
                input_pos_ = torch.where(y == 1)[0]
                if exclude_negatives is None:
                    negs = list(set(np.arange(y.shape[0]))-set(input_pos_.numpy()))
                    neg_prob = ((1/((crispr_effect[negs] > -1).sum().sort_values()+1))/(1/((crispr_effect[negs] > -1).sum().sort_values()+1)).sum()).values
                    input_neg_ = np.random.choice(negs, size=input_pos_.shape[0]*npr, replace=False, p=neg_prob)
                else:
                    negs = list(set(np.arange(y.shape[0]))-set(input_pos_.numpy())-exclude_negatives)
                    neg_prob = ((1/((crispr_effect[negs] > -1).sum().sort_values()+1))/(1/((crispr_effect[negs] > -1).sum().sort_values()+1)).sum()).values
                    input_neg_ = np.random.choice(a=negs, size=input_pos_.shape[0]*npr, replace=False, p=neg_prob)
                input = torch.hstack((input_pos_, torch.tensor(input_neg_)))
            y = y[input]

            # Train
            if include_extra_features:
                loss, ap, auc, probs = train_loop_baseline(sample=input, y=y, model=model,
                                                            loss_fn=loss_fn, optimizer=optimizer,
                                                            device=device, verbose=verbose,
                                                            return_embs=return_embs, extra_features=x,
                                                            exclude_genes=exclude_genes, exclude_loss=exclude_loss)
            else:
                loss, ap, auc, probs = train_loop_baseline(sample=input, y=y, model=model,
                                                            loss_fn=loss_fn, optimizer=optimizer,
                                                            device=device, verbose=verbose,
                                                            return_embs=return_embs, extra_features=None,
                                                            exclude_genes=exclude_genes, exclude_loss=exclude_loss)
            spr_st, spr_pv = spearmanr(probs.ravel(), crispr_effect.loc[name, input.numpy()].values*-1)
            
            train_spr.append(spr_st)
            train_loss.append(loss)
            train_ap.append(ap)
            train_auc.append(auc)
            
            prob_vectors[name] = probs.reshape(1, -1).squeeze()
        
        total_train_spr.append(np.mean(train_spr))
        total_train_loss.append(np.mean(train_loss))
        total_train_ap.append(np.mean(train_ap))
        total_train_auc.append(np.mean(train_auc))

        if log:
            run.log({'epoch': ep, 'train spr': total_train_spr[-1]})
            run.log({'epoch': ep, 'train loss': total_train_loss[-1]})
            run.log({'epoch': ep, 'train auc': total_train_auc[-1]})
            run.log({'epoch': ep, 'train ap': total_train_ap[-1]})

        # Validation
        if validation:
            random.shuffle(val_cls)
            model.eval()
            for name in val_cls:
                x, y = val_xy[name][0], val_xy[name][1]

                if equal_training:
                    input_pos_ = torch.where(y == 1)[0]
                    if exclude_negatives is None:
                        negs = list(set(np.arange(y.shape[0]))-set(input_pos_.numpy()))
                        neg_prob = ((1/((crispr_effect[negs] > -1).sum().sort_values()+1))/(1/((crispr_effect[negs] > -1).sum().sort_values()+1)).sum()).values
                        input_neg_ = np.random.choice(negs, size=input_pos_.shape[0]*npr, replace=False, p=neg_prob)
                    else:
                        negs = list(set(np.arange(y.shape[0]))-set(input_pos_.numpy())-exclude_negatives)
                        neg_prob = ((1/((crispr_effect[negs] > -1).sum().sort_values()+1))/(1/((crispr_effect[negs] > -1).sum().sort_values()+1)).sum()).values
                        input_neg_ = np.random.choice(a=negs, size=input_pos_.shape[0]*npr, replace=False, p=neg_prob)
                    input = torch.hstack((input_pos_, torch.tensor(input_neg_)))
                y = y[input]
                    
                if include_extra_features:
                    val_loss_, val_ap_, val_auc_, val_probs_ = test_loop_baseline(sample=input, y=y, model=model,
                                                                        loss_fn=loss_fn, device=device,
                                                                        test=False, verbose=verbose,
                                                                        extra_features=x, crispr_effect=crispr_effect,
                                                                        focus2int=None,
                                                                        cl=name, exclude_genes=exclude_genes,
                                                                        exclude_loss=exclude_loss)
                else:
                    val_loss_, val_ap_, val_auc_, val_probs_ = test_loop_baseline(sample=input, y=y, model=model,
                                                                        loss_fn=loss_fn, device=device,
                                                                        test=False, verbose=verbose,
                                                                        extra_features=None, crispr_effect=crispr_effect,
                                                                        focus2int=None,
                                                                        cl=name, exclude_genes=exclude_genes,
                                                                        exclude_loss=exclude_loss)
                spr_st, spr_pv = spearmanr(val_probs_.ravel(), crispr_effect.loc[name, input.numpy()].values*-1)
                            
                val_spr.append(spr_st)
                val_loss.append(val_loss_)
                val_ap.append(val_ap_)
                val_auc.append(val_auc_)

            total_val_spr.append(np.mean(val_spr))
            total_val_loss.append(np.mean(val_loss))
            total_val_ap.append(np.mean(val_ap))
            total_val_auc.append(np.mean(val_auc))


            if log:
                run.log({'epoch': ep, 'val spr': total_val_spr[-1]})
                run.log({'epoch': ep, 'val loss': total_val_loss[-1]})
                run.log({'epoch': ep, 'val auc': total_val_auc[-1]})
                run.log({'epoch': ep, 'val ap': total_val_ap[-1]})
        
        # if validation:
            print(f"Train Loss - {np.mean(total_train_loss[-1]):.3f} "
                f"Validation Loss {np.mean(total_val_loss[-1]):.3f} "
                f"Train SPR {np.mean(total_train_spr[-1]):.3f} "
                f"Train AUC {np.mean(total_train_auc[-1]):.3f} "
                f"Validation AUC {np.mean(total_val_auc[-1]):.3f} "
                f"Train AP {np.mean(total_train_ap[-1]):.3f} "
                f"Validation AP {np.mean(total_val_ap[-1]):.3f} "
                f"Validation SPR {np.mean(total_val_spr[-1]):.3f}")
            
            if save_model_name is not None:
                if total_val_loss[ep] < np.min(total_val_loss) or ep+1 == epochs:
                    print("saving model")
                    torch.save(model.state_dict(), save_model_name)

            if early_stop:
                if total_val_loss[ep] > total_val_loss[ep-1]:
                    patience_ += 1
                    print(patience_)
                else:
                    patience_ = 0
                if patience_ == patience:
                    if save_model_name is not None:
                        print("saving model")
                        torch.save(model.state_dict(), save_model_name)
                    break

        else:
            print(f"Train Loss - {np.mean(total_train_loss[-1]):.3f} "
                f"Train AUC {np.mean(total_train_auc[-1]):.3f} "
                f"Train AP {np.mean(total_train_ap[-1]):.3f}")
    
    return total_train_loss, total_val_loss, total_train_auc, total_val_auc, total_train_ap, total_val_ap, prob_vectors


def model_baseline_train_validation_regression(input, model, loss_fn, optimizer, device, verbose, return_embs,
                                               epochs, log, validation, early_stop, patience, run, train_cls,
                                               train_xy, val_cls=None, val_xy=None, save_model_name=None, 
                                               crispr_effect=None, focus2int=None, include_extra_features=True,
                                               exclude_genes=None, exclude_loss=False, npr=5, exp_loss=False):
    total_train_loss, total_val_loss = [], []
    total_train_spr, total_val_spr = [], []
    total_train_pr, total_val_pr = [], []
    prob_vectors = {}
    prob_vectors_val = {}
    patience_ = 0
    for ep in range(epochs):
        # ep = 0
        print(f"Epoch {ep+1}\n ---------------------------------------------")
        train_loss, val_loss = [], []
        train_spr, val_spr = [], []
        train_pr, val_pr = [], []
        model.train()
        random.shuffle(train_cls)
        for name in train_cls:
            x, y = train_xy[name][0], train_xy[name][1]
            y = y[input] # Always make sure the labels equal the input

            # Train
            if include_extra_features:
                loss, probs = train_loop_baseline(sample=input, y=y, model=model,
                                                  loss_fn=loss_fn, optimizer=optimizer,
                                                  device=device, verbose=verbose,
                                                  return_embs=return_embs, extra_features=x,
                                                  exclude_genes=exclude_genes, exclude_loss=exclude_loss,
                                                  calculate_performance=False, exp_loss=exp_loss)
            else:
                loss, probs = train_loop_baseline(sample=input, y=y, model=model,
                                                  loss_fn=loss_fn, optimizer=optimizer,
                                                  device=device, verbose=verbose,
                                                  return_embs=return_embs, extra_features=None,
                                                  exclude_genes=exclude_genes, exclude_loss=exclude_loss,
                                                  calculate_performance=False, exp_loss=exp_loss)
            train_loss.append(loss)
            if exclude_genes is not None:
                ix = ~np.in1d(np.arange(probs.shape[0]), np.array(exclude_genes))
                spr = spearmanr(y[ix].ravel(), probs[ix].detach().numpy().ravel())
                pr = pearsonr(y[ix].ravel(), probs[ix].detach().numpy().ravel())
            else:
                spr = spearmanr(y.ravel(), probs.detach().numpy().ravel())
                pr = pearsonr(y.ravel(), probs.detach().numpy().ravel())

            train_spr.append(spr)
            train_pr.append(pr)
            prob_vectors[name] = probs.reshape(1, -1).squeeze()
        
        total_train_loss.append(np.mean(train_loss))
        total_train_spr.append(np.mean(train_spr))
        total_train_pr.append(np.mean(train_pr))

        if log:
            run.log({'epoch': ep, 'train loss': total_train_loss[-1]})
            run.log({'epoch': ep, 'train spr': total_train_spr[-1]})
            run.log({'epoch': ep, 'train pr': total_train_pr[-1]})

        # Validation
        if validation:
            random.shuffle(val_cls)
            model.eval()
            for name in val_cls:
                x, y = val_xy[name][0], val_xy[name][1]
                y = y[input] # Always make sure the labels equal the input

                    
                if include_extra_features:
                    val_loss_, val_probs_ = test_loop_baseline(sample=input, y=y, model=model,
                                                               loss_fn=loss_fn, device=device,
                                                               test=False, verbose=verbose,
                                                               extra_features=x, crispr_effect=crispr_effect,
                                                               focus2int=focus2int,
                                                               cl=name, exclude_genes=exclude_genes,
                                                               exclude_loss=exclude_loss,
                                                               calculate_performance=False, exp_loss=exp_loss)
                else:
                    val_loss_, val_probs_ = test_loop_baseline(sample=input, y=y, model=model,
                                                               loss_fn=loss_fn, device=device,
                                                               test=False, verbose=verbose,
                                                               extra_features=None, crispr_effect=crispr_effect,
                                                               focus2int=focus2int,
                                                               cl=name, exclude_genes=exclude_genes,
                                                               exclude_loss=exclude_loss,
                                                               calculate_performance=False, exp_loss=exp_loss)
        
                val_loss.append(val_loss_)
                if exclude_genes is not None:
                    ix = ~np.in1d(np.arange(val_probs_.shape[0]), np.array(exclude_genes))
                    spr = spearmanr(y[ix].ravel(), val_probs_[ix].detach().numpy().ravel())
                    pr = pearsonr(y[ix].ravel(), val_probs_[ix].detach().numpy().ravel())
                else:
                    spr = spearmanr(y.ravel(), val_probs_.detach().numpy().ravel())
                    pr = pearsonr(y.ravel(), val_probs_.detach().numpy().ravel())
                val_spr.append(spr)
                val_pr.append(pr)
                prob_vectors_val[name] = val_probs_.reshape(1, -1).squeeze()


            total_val_loss.append(np.mean(val_loss))
            total_val_spr.append(np.mean(val_spr))
            total_val_pr.append(np.mean(val_pr))


            if log:
                run.log({'epoch': ep, 'val loss': total_val_loss[-1]})
                run.log({'epoch': ep, 'val spr': total_val_spr[-1]})
                run.log({'epoch': ep, 'val pr': total_val_pr[-1]})
        
        # if validation:
            print(f"Train Loss - {np.mean(total_train_loss[-1]):.3f} "
                f"Validation Loss {np.mean(total_val_loss[-1]):.3f} "
                f"Train SPR {np.mean(total_train_spr[-1]):.3f} "
                f"Train PR {np.mean(total_train_pr[-1]):.3f} "
                f"Validation SPR {np.mean(total_val_spr[-1]):.3f} "
                f"Validation PR {np.mean(total_val_pr[-1]):.3f} ")
            
            if save_model_name is not None:
                if total_val_loss[ep] < np.min(total_val_loss) or ep+1 == epochs:
                    print("saving model")
                    torch.save(model.state_dict(), save_model_name)

            if early_stop:
                if total_val_loss[ep] > total_val_loss[ep-1]:
                    patience_ += 1
                    print(patience_)
                else:
                    patience_ = 0
                if patience_ == patience:
                    if save_model_name is not None:
                        print("saving model")
                        torch.save(model.state_dict(), save_model_name)
                    break

        else:
            print(f"Train Loss - {np.mean(total_train_loss[-1]):.3f} "
                  f"Train AUC {np.mean(total_train_spr[-1]):.3f} ")
    
    return total_train_loss, total_val_loss, total_train_spr, total_val_spr, prob_vectors, prob_vectors_val



def calculate_gsea(pathway_d, gene_list, M=None, pval=0.05, gene_to_include=None, prerank=False, outdir=None,
                   processes=12, max_size=500, min_size=15):
    """
    :param pathway_d: dict of pathways to calculate enrichment for
    :param gene_list: ranked df if prerank=True, else normal list of genes
    :param M:
    :param pval:
    :param gene_to_include:
    :param prerank:
    :param outdir:
    :return:
    """
    if prerank:
        res = gp.prerank(rnk=gene_list, gene_sets=pathway_d, outdir=outdir, processes=processes,
                         min_size=min_size, max_size=max_size).results
        gsea = {}
        if gene_to_include is not None:
            gsea[gene_to_include] = []
            for path, d_ in res.items():
                if (d_["es"] > 0) & (d_['fdr'] < 0.05) & (gene_to_include in d_["ledge_genes"]):
                    gsea[gene_to_include].append(' '.join(path.split('_')[1:]).lower().capitalize())
            return res, gsea
        else:
            return res

    else:

        pval_l = []
        path_list = []
        for pathway, members in pathway_d.items():
            hypergeom_reactome = hypergeom(M=M, n=len(members), N=len(gene_list))
            overlap = set(members) & gene_list
            pval_l.append(hypergeom_reactome.sf(len(overlap) - 1))
            path_list.append(pathway)
        reject, qvals, _, _ = multipletests(pval_l, pval, method='fdr_bh')
        significant_pathways = dict(zip(np.array(path_list)[reject], qvals[reject]))
        df = pd.Series(significant_pathways,
                         index=significant_pathways.keys()).sort_values().to_frame(name='FDR q-value')

        if gene_to_include is not None:
            gsea = []
            for path in df.index:
                if gene_to_include in pathway_d[path]:
                    gsea.append(' '.join(path.split('_')[1:]).lower().capitalize())
            return df, gsea
        else:
            return df
        

def read_gmt_file(fp, nw_obj):
    genes_per_DB = {}
    if isinstance(nw_obj, list):
        focus_genes = set(nw_obj)
    else:
        focus_genes = set(nw_obj.node_names)
    with open(fp) as f:
        lines = f.readlines()
        for line in lines:
            temp = line.strip('\n').split('\t')
            genes_per_DB[temp[0]] = set(gene for gene in temp[2:]) & focus_genes
    return genes_per_DB


def calculate_traintestval_thresholds(dep_dict, train_test_ratio, train_validation_ratio):
    out_d = {}

    if train_test_ratio is not None:
        for cl, genes in dep_dict.items():
            ttr = int(len(genes) * train_test_ratio)
            if train_validation_ratio is not None:
                tvr = int(ttr * train_validation_ratio)
            else:
                tvr = None
            out_d[cl] = tuple((ttr, tvr))
    else:
        return None

    return out_d


def generate_traintest_dependencies(dependency_data, threshold_neg, threshold_pos, npr, gene2int,
                                    train_test_ratio=0.8, train_validaiton_ratio=None, exclude_negs=None):
    """
    Returns a number of negative and positive interactions based on dependency threshold and a predefined negative:positive ratio

    :param dependency_data: dataframe with cols=genes, index=cell line
    :param pos_dict: dictionary containing positives per cell line
    :param threshold: dependency log2 threshold
    :param npr: negative to positive ratio for dependencies
    :param gene2int: dictionary that maps each node to its index
    :return: array of negative interactions
    """
    min_no_deps = 3
    pos = {}
    negs = {}
    intermediate = {}
    for cl in dependency_data.index:
        nans = set(dependency_data.loc[cl][dependency_data.loc[cl].isna()].index)
        tmp_pos = dependency_data.loc[cl][dependency_data.loc[cl] < threshold_pos].index.tolist()
        if tmp_pos and len(tmp_pos) > min_no_deps-1:
            pos[cl] = tmp_pos
        else:
            # print(
            #     f"For cell line {cl} {len(tmp_pos)} postives are found at threshold {threshold_pos}, increasing threshold by 0.5")
            # thresh_pos_new = threshold_pos
            # while len(tmp_pos) < min_no_deps:
            #     thresh_pos_new += + 0.5
            #     tmp_pos = dependency_data.loc[cl][dependency_data.loc[cl] < thresh_pos_new].index.tolist()
            # print(f"For cell line {cl}, {len(tmp_pos)} positives were found at threshold {thresh_pos_new}")
            # pos[cl] = tmp_pos
            print(f"For cell line {cl} {len(tmp_pos)} postives are found at threshold {threshold_pos}")
            continue
        N_negs = int(npr * len(pos[cl]))
        all_negs = dependency_data.loc[cl][dependency_data.loc[cl] > threshold_neg].index.tolist()
        if len(all_negs) < N_negs:
            print(f"Too few negatives available, taking the maximum possible: {len(all_negs)}")
            negs[cl] = list(set(all_negs) - nans)
        else:
            negs[cl] = list(set(random.sample(all_negs, N_negs)) - nans)

        tmp_interm = dependency_data.loc[cl][
            (dependency_data.loc[cl] < threshold_neg) & (dependency_data.loc[cl] > threshold_pos)].index.tolist()
        intermediate[cl] = list(set(tmp_interm) - nans)
        # pdb.set_trace()

    if exclude_negs:
        negs = {k: list(set(random.sample(v, len(v)))-exclude_negs) for k, v in negs.items()}
    pos = {k:random.sample(v, len(v)) for k, v in pos.items()}
    intermediate = {k: random.sample(v, len(v)) for k, v in intermediate.items()}


    neg_thresh = calculate_traintestval_thresholds(negs, train_test_ratio, train_validaiton_ratio) # cl = tuple((ttr, tvr))
    pos_thresh = calculate_traintestval_thresholds(pos, train_test_ratio, train_validaiton_ratio)
    interm_thresh = calculate_traintestval_thresholds(intermediate, train_test_ratio, train_validaiton_ratio)

    if train_test_ratio is not None:
        negs_arr_train = np.array(
            [[gene2int[celline], gene2int[gene]] for celline, genes in negs.items() for gene in
             genes[:neg_thresh[celline][0]]])
        pos_arr_train = np.array(
            [[gene2int[celline], gene2int[gene]] for celline, genes in pos.items() for gene in
                                  genes[:pos_thresh[celline][0]]])
        intermediate_arr_train = np.array(
            [[gene2int[celline], gene2int[gene]] for celline, genes in intermediate.items() for gene in
             genes[:interm_thresh[celline][0]]])

        negs_arr_test = np.array([[gene2int[celline], gene2int[gene]] for celline, genes in negs.items() for gene in
                                 genes[neg_thresh[celline][0]:]])
        pos_arr_test = np.array([[gene2int[celline], gene2int[gene]] for celline, genes in pos.items() for gene in
                                genes[pos_thresh[celline][0]:]])
        intermediate_arr_test = np.array(
            [[gene2int[celline], gene2int[gene]] for celline, genes in intermediate.items() for gene in
             genes[interm_thresh[celline][0]:]])

        if train_validaiton_ratio is not None:
            negs_arr_train = np.array(
                [[gene2int[celline], gene2int[gene]] for celline, genes in negs.items() for gene in
                 genes[:neg_thresh[celline][1]]])
            pos_arr_train = np.array(
                [[gene2int[celline], gene2int[gene]] for celline, genes in pos.items() for gene in
                 genes[:pos_thresh[celline][1]]])
            intermediate_arr_train = np.array(
                [[gene2int[celline], gene2int[gene]] for celline, genes in intermediate.items() for gene in
                 genes[:interm_thresh[celline][1]]])

            negs_arr_val = np.array(
                [[gene2int[celline], gene2int[gene]] for celline, genes in negs.items() for gene in
                 genes[neg_thresh[celline][1]:neg_thresh[celline][0]]])
            pos_arr_val = np.array(
                [[gene2int[celline], gene2int[gene]] for celline, genes in pos.items() for gene in
                 genes[pos_thresh[celline][1]:pos_thresh[celline][0]]])
            intermediate_arr_val = np.array(
                [[gene2int[celline], gene2int[gene]] for celline, genes in intermediate.items() for gene in
                 genes[interm_thresh[celline][1]:interm_thresh[celline][0]]])

            return negs, negs_arr_train, negs_arr_val, negs_arr_test, pos, pos_arr_train, pos_arr_val, pos_arr_test,\
                       intermediate, intermediate_arr_train, intermediate_arr_val, intermediate_arr_test

        return negs, negs_arr_train, negs_arr_test, pos, pos_arr_train, pos_arr_test, \
                   intermediate, intermediate_arr_train, intermediate_arr_test

    else:
        negs_arr = np.array(
            [[gene2int[celline], gene2int[gene]] for celline, genes in negs.items() for gene in genes])
        pos_arr = np.array(
            [[gene2int[celline], gene2int[gene]] for celline, genes in pos.items() for gene in genes])
        intermediate_arr = np.array(
            [[gene2int[celline], gene2int[gene]] for celline, genes in intermediate.items() for gene in genes])

        return negs, negs_arr, pos, pos_arr, intermediate, intermediate_arr


def construct_combined_traintest(pos_arr_train, negs_arr_train, X_train_, Y_train_,
                                 pos_arr_val=None, negs_arr_val=None, X_val_=None, Y_val_=None,
                                 pos_arr_test=None, negs_arr_test=None, X_test_=None, Y_test_=None):
    """
    Construct combined PPI & dependencies train/test test
    :param pos_arr_train: Positive TRAIN dependencies as returned by function generate_traintest_dependencies
    :param pos_arr_val: Positive TEST dependencies as returned by function generate_traintest_dependencies
    :param negs_arr_train: Negative TRAIN dependencies as returned by function generate_traintest_dependencies
    :param negs_arr_val: Negative TEST dependencies as returned by function generate_traintest_dependencies
    :param X_train_: train PPI interactions
    :param Y_train_: train PPI labels
    :param X_test_: test PPI interactions
    :param Y_test_: test PPI labels
    :return: Final X_train, y_train, X_test, y_test
    """
    assert len(
        set(map(tuple, pos_arr_train)) & set(map(tuple, negs_arr_train))) == 0, "Error overlapping neg pos TRAIN interactions"
    if pos_arr_val is not None and negs_arr_val is not None:
        assert len(
            set(map(tuple, pos_arr_val)) & set(map(tuple, negs_arr_val))) == 0, "Error overlapping neg pos VAL interactions"
    if pos_arr_test is not None and negs_arr_test is not None:
        assert len(
            set(map(tuple, pos_arr_test)) & set(map(tuple, negs_arr_test))) == 0, "Error overlapping neg pos test interactions"

    train_dependencies = np.vstack((pos_arr_train, negs_arr_train))
    train_labels = np.hstack(
        (np.ones(pos_arr_train.shape[0]), np.zeros(negs_arr_train.shape[0])))
    assert train_dependencies.shape[0] == train_labels.shape[0], 'ERROR'

    if pos_arr_val is not None and negs_arr_val is not None and pos_arr_val.shape[0] != 0:
        val_dependencies = np.vstack((pos_arr_val, negs_arr_val))
        val_labels = np.hstack(
            (np.ones(pos_arr_val.shape[0]), np.zeros(negs_arr_val.shape[0])))
        assert val_dependencies.shape[0] == val_labels.shape[0], 'ERROR'
    else:
        val_dependencies = None
        val_labels = None

    if pos_arr_test is not None and negs_arr_test is not None and pos_arr_test.shape[0] != 0:
        test_dependencies = np.vstack((pos_arr_test, negs_arr_test))
        test_labels = np.hstack(
            (np.ones(pos_arr_test.shape[0]), np.zeros(negs_arr_test.shape[0])))
        assert test_dependencies.shape[0] == test_labels.shape[0], 'ERROR'
    else:
        test_dependencies = None
        test_labels = None

    X_train = np.vstack((X_train_, train_dependencies))
    if X_test_ is not None and X_test_.shape[0] != 0:
        X_test = np.vstack((X_test_, test_dependencies))
    else:
        X_test = None
    if X_val_ is not None and X_val_.shape[0] != 0:
        X_val = np.vstack((X_val_, val_dependencies))
    else:
        X_val = None

    y_train = np.hstack((Y_train_, train_labels))
    if Y_test_ is not None and Y_test_.shape[0] != 0:
        y_test = np.hstack((Y_test_, test_labels))
    else:
        y_test = None
    if Y_val_ is not None and Y_val_.shape[0] != 0:
        y_val = np.hstack((Y_val_, val_labels))
    else:
        y_val = None

    if X_test_ is not None:
        if X_val_ is not None:
            return X_train, X_val, X_test, y_train, y_val, y_test
        return X_train, X_test, y_train, y_test
    else:
        return X_train, y_train
    

def read_h5py(fp, dtype=int):
    hf = h5py.File(fp, 'r')
    x = np.array(hf.get(fp.split('/')[-1][:-5]), dtype=dtype)
    hf.close()
    return x


def write_h5py(fp, data):
    hf = h5py.File(fp, 'w')
    hf.create_dataset(fp.split('/')[-1][:-5], data=data)
    hf.close()
    return None


def get_sensitive_drugs_and_targets(screen_df, screen_info_df, cell_line, threshold=0.3):
    """
    returns the sensitive drugs with corresponding targets for a certain cell line based on a provided threshold. 
    Note that sensitive drugs for which the target is unknown are not returned
    param:
    screen_df - drug screening dataframe with viability measures
    screen_info_df - metadata over screened drugs
    cell_line - cell line of interest
    threshold - viability threshold at which a cell line is considered sensitive if below this threhsold
    """
    if threshold is None:
        thresholded_ss = screen_df
    else:
        thresholded_ss = screen_df.applymap(lambda x: int(x < threshold))
    cl = thresholded_ss.loc[cell_line]
    sens_cl = cl[cl == 1].index
    sens_cl_info = screen_info_df.loc[sens_cl]
    sens_cl_info = sens_cl_info[~sens_cl_info['target'].isna()]
    tmp = sens_cl_info.groupby('name').groups
    return {k: set(sens_cl_info.loc[v, 'target'].values) for k, v in tmp.items()}


def moa_analysis(cell_gene_mat: pd.DataFrame, query_gene: str, ppi_mat: pd.DataFrame = None) -> pd.Series:
    """AI is creating summary for moa_analysis
    Performs a Mode of Action analysis of a gene of interest using the dependency and PPI score.
    
    Args:
        cell_gene_mat (pd.DataFrame): [description]
        query_gene (str): [description]
        ppi_mat (pd.DataFrame, optional): [description]. Defaults to None.

    Returns:
        pd.Series: [description]
    """
    query2cell_vec = cell_gene_mat[query_gene].values
    scores = np.matmul(query2cell_vec, cell_gene_mat.values)

    if ppi_mat is None:
        return pd.Series(scores, index=cell_gene_mat.columns).sort_values(ascending=False)

    else:

        cell_scores = pd.Series(scores, index=cell_gene_mat.columns).sort_values(ascending=False)
        total_score = ppi_mat[query_gene] + cell_scores.loc[ppi_mat.index]/query2cell_vec.shape[0]

        return total_score.sort_values(ascending=False)
    


def gene_assay_performance(genes: list, ori_df: pd.DataFrame, prob_df: pd.DataFrame,
                           performance_metric: sklearn.metrics) -> dict:
    """AI is creating summary for gene_assay_performance

    Args:
        genes (list): [description]
        ori_df (pd.DataFrame): needs to be binary
        prob_df (pd.DataFrame): [description]
        performance_metric (sklearn.metrics): [description]

    Returns:
        dict: [description]
    """
    gene_ap, assay_ap = [], []
    for i, row in prob_df.loc[:, genes].iterrows():
        assay_ap.append(performance_metric(y_true=ori_df.loc[i, genes].values,
                                                y_score=row.values))
    for col in genes:
        gene_ap.append(performance_metric(y_true=ori_df[col].values,
                                                y_score=prob_df[col].values))
    return {"gene": gene_ap, "assay": assay_ap}


def plot_TSNE(df1, title1, hue, df2=None, title2=None, save_fp=None, annotate=False):   

    if df2 is not None: 
        _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
        sns.scatterplot(data=df1, x='dim1', y='dim2', hue=hue, ax=ax1)
        ax1.set_title(title1)
        sns.scatterplot(data=df2, x='dim1', y='dim2', hue=hue, ax=ax2)
        ax2.set_title(title2)
    else:
        _, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        sns.scatterplot(data=df1, x='dim1', y='dim2', hue=hue, ax=ax1)
        ax1.set_title(title1)

    if save_fp is not None:
        plt.savefig(save_fp)
        plt.close()


def calculate_AMI(cls_interest, topK1, topK2, score_df, score_df_ori,
                  num_clusters, true_labels, ascending, exclude=None, exclude_ori=None):
    gene_overlap_df = pd.DataFrame(np.eye(len(cls_interest)), columns=cls_interest, index=cls_interest)
    gene_overlap_df_ori = gene_overlap_df.copy(deep=True)
    exclude = set() if exclude is None else exclude
    exclude_ori = set() if exclude_ori is None else exclude_ori
    for a, b in combinations(cls_interest, 2):
        numerator = set(score_df.loc[a].sort_values(ascending=ascending)[topK1:topK2].index) &\
                                    set(score_df.loc[b].sort_values(ascending=ascending)[topK1:topK2].index) - exclude
        denominator = set(score_df.loc[a].sort_values(ascending=ascending)[topK1:topK2].index) |\
                                    set(score_df.loc[b].sort_values(ascending=ascending)[topK1:topK2].index) - exclude
        gene_overlap_df.loc[a, b] = len(numerator) / len(denominator)

        numerator = set(score_df_ori.loc[a].sort_values(ascending=ascending)[topK1:topK2].index) &\
                                    set(score_df_ori.loc[b].sort_values(ascending=ascending)[topK1:topK2].index) - exclude_ori
        denominator = set(score_df_ori.loc[a].sort_values(ascending=ascending)[topK1:topK2].index) |\
                                    set(score_df_ori.loc[b].sort_values(ascending=ascending)[topK1:topK2].index) - exclude_ori
    
        gene_overlap_df_ori.loc[a, b] = len(numerator) / len(denominator)
    
    gene_overlap_distance = 1 - gene_overlap_df
    gene_overlap_distance_ori = 1 - gene_overlap_df_ori

    distance_matrix = linkage(gene_overlap_distance.values[np.triu_indices(gene_overlap_df.shape[0], k=1)],
                              method = 'ward')
    distance_matrix_ori = linkage(gene_overlap_distance_ori.values[np.triu_indices(gene_overlap_df_ori.shape[0], k=1)],
                              method = 'ward')

    cluster_labels = fcluster(distance_matrix, num_clusters, criterion='maxclust')
    cluster_labels_ori = fcluster(distance_matrix_ori, num_clusters, criterion='maxclust')

    ami_pred = adjusted_mutual_info_score(labels_true=true_labels, labels_pred=cluster_labels)
    ami_ori = adjusted_mutual_info_score(labels_true=true_labels, labels_pred=cluster_labels_ori)
    return ami_pred, ami_ori