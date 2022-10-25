#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import os
import pickle
import pickle as pkl
import torch
import json
from argparse import Namespace
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import math
from utils import utils
import time
import sys

from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.preprocessing import OneHotEncoder as one_enc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, pairwise_distances, classification_report
from sklearn.decomposition import PCA

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from dgl import DGLGraph
from dgl.data.utils import load_graphs
from dgl.data.utils import save_graphs
import networkx as nx

import matplotlib
import matplotlib.pyplot as plt

#from args import get_args
from collections import OrderedDict, defaultdict
from json import dumps
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
#from torch.utils.tensorboard import SummaryWriter

gcn_msg = fn.copy_src(src='h', out='m')
gcn_mul = fn.u_mul_e('h', 'a', 'm')
gcn_reduce = fn.sum(msg='m', out='h')


def T_scaling(logits, temperature):
    return torch.div(logits, temperature)

def calibrate(model, graph, features, labels, nid, loss_fn):
    model.eval()
    temperature = nn.Parameter(torch.ones(1))#.cuda())

    # Removing strong_wolfe line search results in jump after 50 epochs
    optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=10000, 
        line_search_fn='strong_wolfe')

    losses = []
    temps = []
    with torch.no_grad():
        logits = model(graph, features)
        if logits.shape[-1] == 1:
            logits = logits.view(-1)
        logits = logits[nid]
        labels = labels[nid]
        loss = loss_fn(logits, labels)

    def _eval():
        loss = loss_fn(T_scaling(logits, temperature), labels)
        loss.backward()
        temps.append(temperature.item())
        losses.append(loss)
        return loss

    optimizer.step(_eval)

    return temperature


def evaluate(model, graph, features, labels, nid, loss_fn, best_thresh=0.5,
            save_file=None, thresh_search=False, temperature=None, print_report=False):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[nid]
        if temperature is not None:
            logits = T_scaling(logits, temperature)

        if logits.shape[-1] == 1:
            logits = logits.view(-1)  # (batch_size,)

        if isinstance(loss_fn, nn.BCEWithLogitsLoss):
            logits = logits.view(-1)  # (batch_size,)
            probs = torch.sigmoid(logits).cpu().numpy()  # (batch_size, )
            preds = (probs >= best_thresh).astype(int)  # (batch_size, )
        else:
            # (batch_size, num_classes)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1).reshape(-1)  # (batch_size,)
            probs = np.max(probs, axis=1).reshape(-1)
        eval_results = {'message':'No groundtruth available'}
        if labels is not None:
            labels = labels[nid] 
            print('PREDICTED:\t\tZeros:\t', len([p for p in preds if p==0]), 'Ones:\t', len([p for p in preds if p==1]))
            print('GROUNDTRUTH:\t\tZeros:\t', len([p for p in labels if p==0]), 'Ones:\t', len([p for p in labels if p==1]))
            eval_results = utils.eval_dict(y=labels.data.cpu().numpy(), 
                                        y_pred=preds, 
                                        y_prob=probs, 
                                        average='macro',
                                        thresh_search=thresh_search,
                                        best_thresh=best_thresh)
            try:
                loss = loss_fn(logits, labels)   
                eval_results["loss"] = loss.item()
            except:
                print('cannot compute loss')
            if print_report:
                print(classification_report(labels.data.cpu().numpy(), preds))
    if save_file is not None:
        with open(save_file, "wb") as pf:
            pickle.dump({"probs": probs, "labels": labels, "preds": preds}, pf)
    
    return eval_results, probs, preds, labels



from dgl.nn import GraphConv, TAGConv, SAGEConv, GATConv

class GCN(nn.Module):
    def __init__(self, n_in_feats, n_h_feats, n_classes, dropout, upsample = False, bn_flag=False, 
                 gnn_type = 'GraphConv', no_conv_layers = 1, **kwargs):
        super(GCN, self).__init__()
        self.n_in_feats = n_in_feats
        self.n_h_feats = n_h_feats
        self.bn_flag = bn_flag
        self.n_classes = n_classes
        self.no_conv_layers = no_conv_layers
        self.upsample = upsample
        self.gnn_type = gnn_type

        if self.upsample:
            self.linear_upsample = nn.Linear(n_in_feats, kwargs["n_upsample_feats"])
            self.n_in_feats = kwargs["n_upsample_feats"]
        self.conv_layers = nn.ModuleList()
        if gnn_type == 'GraphConv':
            self.conv_layers.append(GraphConv(self.n_in_feats, self.n_h_feats))
            for i in range(1, self.no_conv_layers):
                self.conv_layers.append(GraphConv(self.n_h_feats, self.n_h_feats))
            
        elif gnn_type == 'SAGEConv':
            self.conv_layers.append(SAGEConv(self.n_in_feats, self.n_h_feats,  aggregator_type=kwargs["aggregator_type"]))
            for i in range(1, self.no_conv_layers):
                self.conv_layers.append(SAGEConv(self.n_h_feats, self.n_h_feats,  aggregator_type=kwargs["aggregator_type"]))

        elif gnn_type == 'GATConv':
            self.conv_layers.append(GATConv(self.n_in_feats, self.n_h_feats,  num_heads=kwargs["num_heads"]))
            for i in range(1, self.no_conv_layers):
                self.conv_layers.append(GATConv(self.n_h_feats, self.n_h_feats,  num_heads=kwargs["num_heads"]))
            
        self.bn1 = nn.BatchNorm1d(self.n_h_feats)
        
       
        self.linear = nn.Linear(self.n_h_feats,self.n_classes)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, g, in_feat):
        if self.upsample:
            in_feat = F.relu(self.linear_upsample(in_feat))
        if self.gnn_type == 'SAGEConv':
            h = self.conv_layers[0](g, in_feat, edge_weight=g.edata['weights'])
        else:
            h = self.conv_layers[0](g, in_feat)
        if self.gnn_type == 'GATConv':
            h = h.mean(1)
        if self.bn_flag == True:
            h = self.bn1(h)
        h = F.relu(self.dropout(h))
        
        for i in range(1, self.no_conv_layers):
            if self.gnn_type == 'SAGEConv':
                h = self.conv_layers[i](g, h,  edge_weight=g.edata['weights'])
            else:
                h = self.conv_layers[i](g, h)
            if self.gnn_type == 'GATConv':
                h = h.mean(1)
            if self.bn_flag == True:
                h = self.bn1(h)
            h = F.relu(self.dropout(h))
        h = self.linear(h)
        return h


def main(graph_path, do_train=False, best_path=None, no_conv_layers=2, gnn_type='SAGEConv'):

    cuda = False#torch.cuda.is_available()
    rand_seed = 0
    pos_weight = 1
    sample_weight = 1
    eval_every = 1
    metric_name = "auroc"
    maximize_metric=True
    lr = 1e-3
    l2_wd = 0

    n_classes = 1
    upsample = False
    n_upsample_feats = 2048
    pca_flag=False
    pca_n_components = 64
    n_h_feats = 128
    dropout = 0.25
    bn_flag = False
    num_heads = 1 #useful with GTA type GNN
    aggregator_type = 'mean' 

    ## useful only if do_train=True
    num_epochs = 100
    loss_func = "binary_cross_entropy"
    undersample=True
    calibrate=True
    
    base_path = ''
    save_dir = base_path+'results/class'
    load_model_path = best_path
    patience = 10
    thresh_search=True

    ## device
    device = "cpu"
    print('Device:\t', device)


    ## set random seed
    utils.seed_torch(seed=rand_seed)
    ## get save directories
    save_dir = utils.get_save_dir(
        save_dir, training=True if do_train else False
    )

    logger = utils.get_logger(save_dir, "train")
    logger.propagate = False


    ##load graph
    g = load_graphs(graph_path)[0]
    g = g[0]
    print(g)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    print('graph loaded')

    ##egde weights
    g.edata['weights'] = torch.tensor(g.edata['weights'].numpy().astype('float32'))

    #masks 
    mask_test = g.ndata['mask_test'].squeeze()
    test_nid = torch.nonzero(mask_test).squeeze()
    try:
        mask_train = g.ndata['mask_train'].squeeze()
        train_nid = torch.nonzero(mask_train).squeeze()
    except:
        print('no train mask provided') 
    try:
        mask_val = g.ndata['mask_val'].squeeze()
        val_nid = torch.nonzero(mask_val).squeeze()
    except:
        print('no train mask provided')   

    try:
        labels = list(g.ndata['labels'].numpy().squeeze())
        labels = torch.from_numpy(np.array(labels).reshape(-1,1))
        labels = labels.squeeze()
        labels = labels.double()
    except:
        labels=None
        print('no labels provided')

    ##features
    features = g.ndata['features']
    print('features created')
    sys.stdout.flush()

    logger.info("""----Data statistics------'
    #Edges %d
    #Average degree %d
    """ %
        (n_edges,
        g.in_degrees().float().mean().item()
        ))

    logger.info("Graph name:\t {}".format(graph_path))
    print('logger created')


    ##create model
    model = GCN(n_in_feats = features.shape[1], n_h_feats = n_h_feats, n_classes=n_classes,upsample = upsample, n_upsample_feats = n_upsample_feats, dropout=dropout,no_conv_layers = no_conv_layers, bn_flag=bn_flag, gnn_type=gnn_type, num_heads = num_heads, aggregator_type = aggregator_type)
    print('model created')
    ## count params
    params = utils.count_parameters(model)
    print("Trainable parameters: {}".format(params))

    if do_train: #if training is to be done, graph needs to have labels, and train and validation samples
        ## define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)#, weight_decay=l2_wd)
        ## loss func
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight])).to(device)
        ## define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)#, weight_decay=l2_wd)
        ## scheduler
        print("Using cosine annealing scheduler...")
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)


    ## checkpoint saver
    saver = utils.CheckpointSaver(
        save_dir=save_dir,
        metric_name=metric_name,
        maximize_metric=maximize_metric,
        log=logger,
    )

    print('init done')
    logger.info(model)


    if do_train:
        # if undersampling majority class during training
        if undersample:
            num_one = int((labels[train_nid]==1).sum())
            num_zero = int((labels[train_nid]==0).sum())
            zero_weight = 1-(num_zero/(num_zero+num_one))
            one_weight = 1-(num_one/(num_zero+num_one))
            weights = torch.Tensor([zero_weight if labels[tid]==0 else one_weight for tid in train_nid])
            num_samples = num_one+num_zero
            idx_one = torch.Tensor([i for i in train_nid if labels[i]==1])
            idx_zero = torch.Tensor([i for i in train_nid if labels[i]==0])
            print("pos weight: {}, sample_weight: {}".format(pos_weight, sample_weight))
        ## Train
        print("Training...")
        model.train()
        epoch = 0
        prev_val_loss = 1e10
        patience_count = 0
        early_stop = False
        train_loss = []
        val_loss = []
        while (epoch != num_epochs) and (not early_stop):
     
            print("Starting epoch {}...".format(epoch))

            if undersample:
                train_idxs = []
                num_one = int((labels[train_nid]==1).sum())
                num_zero = int((labels[train_nid]==0).sum())
                
                print("Before sampling: zero sample: {}, one samples: {}...".format(num_zero, num_one))
                
                idx_ii = np.array(list(WeightedRandomSampler(weights, num_samples, replacement=True)))
                
                
                np.random.shuffle(idx_ii)
                train_idxs = train_nid[torch.LongTensor(idx_ii)]
                num_one = int((labels[train_idxs]==1).sum())
                num_zero = int((labels[train_idxs]==0).sum())
                
                print("After sampling: zero sample: {}, one samples: {}...".format(num_zero, num_one))
            else:
                train_idxs = train_nid       
            
            # forward
            logits = model(g, features)
            if logits.shape[-1] == 1:
                logits = logits.view(-1)
            loss = loss_fn(logits[train_idxs], labels[train_idxs])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            train_loss.append(loss.item())
            print("train/loss {} {}".format(loss.item(), epoch))
            sys.stdout.flush()

            # evaluate on val set
            if epoch % eval_every == 0:
                print("Evaluating at epoch {}...".format(epoch))
                eval_results, _, _, _ = evaluate(model=model, 
                                        graph=g, 
                                        features=features, 
                                        labels=labels, 
                                        nid=val_nid, 
                                        loss_fn=loss_fn)
                model.train()
                saver.save(
                        epoch, model, optimizer, eval_results[metric_name]
                    )
                # accumulate patience for early stopping
                if eval_results["loss"] < prev_val_loss:
                    patience_count = 0
                else:
                    patience_count += 1
                prev_val_loss = eval_results["loss"]

                # Early stop
                if patience_count == patience:
                    early_stop = True

                # Log to console
                results_str = ", ".join(
                    "{}: {:.4f}".format(k, v) for k, v in eval_results.items()
                )
                print("VAL - {}".format(results_str))

            val_loss.append(prev_val_loss)
            epoch += 1
            # step lr scheduler
            scheduler.step()

        print("Training DONE.")
    
            
        matplotlib.use('Agg')
        plt.figure(figsize=(10,8))
        plt.title(loss_func)
        plt.plot(train_loss, label='train')
        plt.plot(val_loss, label='val')
        plt.legend(fontsize=30)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "graph.png"))
        print('Plotting done')

    
        best_path = os.path.join(save_dir, "best.pth.tar")
        model = utils.load_model_checkpoint(best_path, model)

    else: #do_train=False

        model = utils.load_model_checkpoint(load_model_path, model)


        # eval 
        test_results, probs, preds, gt = evaluate(model=model, 
                            graph=g, 
                            features=features,  
                            labels=labels, 
                            nid=nid, 
                            loss_fn=loss_fn,
                            save_file=os.path.join(save_dir, "test_predictions.pkl"),
                            best_thresh=0.5,
                            temperature=temperature, print_report=True)
        test_results_str = ", ".join(
            "{}: {:.4f}".format(k, v) for k, v in test_results.items()
        )
        logger.info("TEST - {}".format(test_results_str))


