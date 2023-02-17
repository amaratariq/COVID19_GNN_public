#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import time
import numpy as np
import sys
import pickle as pkl
import os

from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.preprocessing import OneHotEncoder as one_enc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data.utils import load_graphs
from dgl.data.utils import save_graphs
import networkx as nx

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

gcn_msg = fn.copy_src(src='h', out='m')
gcn_mul = fn.u_mul_e('h', 'a', 'm')
gcn_reduce = fn.sum(msg='m', out='h')

 


def graph_creation(base_file, enc_demo, sim_threshold = 1, graph_header = 'graphs/', label = 'discharged_in_3days', edge_feats='cpt', node_feats='xray', **kwargs):
    
    """
    base_file: /path/to/base cohort file with one xray per row with image features, procedures, conditions, and demographics

    """
    graph_name = graph_header+'graph_threshold_'+str(sim_threshold)
   
    
    dfkey = pd.read_csv(base_file)
    dfkey = dfkey.drop_duplicates(subset=['person_id', 'xray_datetime'], keep='first')
    print('patient data read:\t', len(dfkey), len(dfkey.person_id.unique()))
    sys.stdout.flush()

    dfkey['xrayDate'] = pd.to_datetime(dfkey['xray_datetime'], errors='coerce').dt.date



    demo = ['PATIENT_AGE_BINNED',  'PATIENT_RACE_NAME', 'PATIENT_ETHNICITY_NAME', 'PATIENT_GENDER_NAME']

    cpt = np.array(['Hematology and Coagulation Procedures', 'Organ or Disease Oriented Panels', 'Drug Assay Procedures', 'Chemistry Procedures',
            'Diagnostic Radiology (Diagnostic Imaging) Procedures', 'Microbiology Procedures', 'Cardiovascular Procedures',
            'Surgical Procedures on the Cardiovascular System', 'Pulmonary Procedures', 'Immunology Procedures',
            'Non-Face-to-Face Evaluation and Management Services', 'Transfusion Medicine Procedures',
            'Hydration, Therapeutic, Prophylactic, Diagnostic Injections and Infusions, and Chemotherapy and Other Highly Complex Drug or Highly Complex Biologic Agent Administration',
            'Non-Invasive Vascular Diagnostic Studies', 'Therapeutic Drug Assays', 'Surgical Procedures on the Respiratory System', 'Other Pathology and Laboratory Procedures',
            'Physical Medicine and Rehabilitation Evaluations', 'Diagnostic Ultrasound Procedures', 'Dialysis Services and Procedures',
            'Special Otorhinolaryngologic Services and Procedures'])

    

    icd = np.array(['K55-K64', 'I30-I52', 'E70-E88', 'A00-A09', 'D70-D77', 'D60-D64',
        'T36-T50', 'K00-K14', 'B95-B97', 'A30-A49', 'T80-T88', 'G40-G47',
        'G89-G99', 'J40-J47', 'I95-I99', 'J09-J18', 'N17-N19', 'N40-N53',
        'C43-C44', 'C81-C96', 'N30-N39', 'N10-N16', 'I10-I16', 'I20-I25',
        'L80-L99', 'D65-D69', 'J96-J99', 'I80-I89', 'M05-M14', 'A90-A99',
        'I26-I28', 'M20-M25', 'J90-J94', 'K20-K31', 'E50-E64', 'M60-M63',
        'F40-F48', 'F30-F39', 'N80-N98', 'E40-E46', 'L20-L30', 'I60-I69',
        'J30-J39', 'E65-E68', 'B35-B49', 'G35-G37', 'K50-K52', 'E08-E13',
        'J95-J95', 'J80-J84', 'J00-J06', 'K90-K95', 'D37-D48', 'K80-K87',
        'I05-I09', 'F01-F09', 'I70-I79', 'E00-E07', 'K70-K77', 'D80-D89',
        'H49-H52', 'B25-B34'])
    
    xray = np.array(['img_'+str(i) for i in range(1024)])

    #fill any missing features with default values
    for c in cpt:
        if c not in dfkey.columns:
            dfkey[c] = 0
    for c in icd:
        if c not in dfkey.columns:
            dfkey[c] = 0
    dfkey[cpt] = dfkey[cpt].fillna(0)
    dfkey[icd] = dfkey[icd].fillna(0)
    
    dfkey = dfkey.dropna(subset=xray, how='all') #drop is xray features are not available

    x_xray_train = dfkey.loc[dfkey.split=='train'][xray].to_numpy().astype('float32')
    x_xray_test = dfkey.loc[dfkey.split=='test'][xray].to_numpy().astype('float32')
    x_xray_val = dfkey.loc[dfkey.split=='val'][xray].to_numpy().astype('float32')

    x_cpt_train = dfkey.loc[dfkey.split=='train'][cpt].to_numpy().astype('float32')
    x_cpt_test = dfkey.loc[dfkey.split=='test'][cpt].to_numpy().astype('float32')
    x_cpt_val = dfkey.loc[dfkey.split=='val'][cpt].to_numpy().astype('float32')

    x_icd_train = dfkey.loc[dfkey.split=='train'][icd].to_numpy().astype('float32')
    x_icd_test = dfkey.loc[dfkey.split=='test'][icd].to_numpy().astype('float32')
    x_icd_val = dfkey.loc[dfkey.split=='val'][icd].to_numpy().astype('float32')

    x_demo_train = enc_demo.transform(dfkey.loc[dfkey.split=='train'][demo].to_numpy()).astype('float32').todense()
    x_demo_test = enc_demo.transform(dfkey.loc[dfkey.split=='test'][demo].to_numpy()).astype('float32').todense()
    x_demo_val = enc_demo.transform(dfkey.loc[dfkey.split=='val'][demo].to_numpy()).astype('float32').todense()
    
    mask_train = [True for i in range(len(dfkey.loc[dfkey.split=='train']))] + [False for i in range(len(dfkey.loc[dfkey.split=='val']))] + [False for i in range(len(dfkey.loc[dfkey.split=='test']))]
    mask_val = [False for i in range(len(dfkey.loc[dfkey.split=='train']))] + [True for i in range(len(dfkey.loc[dfkey.split=='val']))] + [False for i in range(len(dfkey.loc[dfkey.split=='test']))]
    mask_test = [False for i in range(len(dfkey.loc[dfkey.split=='train']))] + [False for i in range(len(dfkey.loc[dfkey.split=='val']))] + [True for i in range(len(dfkey.loc[dfkey.split=='test']))]

    labels = list(dfkey.loc[dfkey.split=='train'][label].value.squeeze()) + list(dfkey.loc[dfkey.split=='val'][label].value.squeeze()) + list(dfkey.loc[dfkey.split=='test'][label].value.squeeze()) 
    
    def features_set(n):
        if n=='demo':
            return x_demo_train, x_demo_val, x_demo_test
        elif n=='cpt':
            return x_cpt_train, x_cpt_val, x_cpt_test
        elif n=='icd':
            return x_icd_train, x_icd_val, x_icd_test
        elif n=='xray':
            return x_xray_train, x_xray_val, x_xray_test
        else:
            print(n, 'Not Implemented')

    node_features = [n.strip() for n in node_feats.split(',')]
    edge_features = [n.strip() for n in edge_feats.split(',')]
    
    n = node_features[0]
    mat_nodes = np.concatenate((features_set(n)[0], features_set(n)[1], features_set(n)[2]), axis = 0)
    if len(node_features)>1:
        for n in node_features[1:]:
            mat_nodes = np.concatenate(mat_nodes, np.concatenate((features_set(n)[0], features_set(n)[1], features_set(n)[2]), axis = 0), axis=1)
        
    n = edge_features[0]
    mat_edges = np.concatenate((features_set(n)[0], features_set(n)[1], features_set(n)[2]), axis = 0)
    if len(edge_features)>1:
        for n in edge_features[1:]:
            mat_edges = np.concatenate(mat_edges, np.concatenate((features_set(n)[0], features_set(n)[1], features_set(n)[2]), axis = 0), axis=1)
            
            
    print('Characteristics matrix formed', mat_nodes.shape, mat_edges.shape)
    sys.stdout.flush()
    
    mat_edges[mat_edges>0] = 1
    
    mat_nodes = mat_nodes.astype('float32')
    mat_edges = mat_edges.astype('float32')
    
    sim_mat = cos_sim(mat_edges)
                
    sim_mat = np.round(sim_mat, decimals=2)
    sim_mat = sim_mat.astype('float16')
    print('Similariy matrix formed')
    sys.stdout.flush()

    graph_name = graph_name+'_'+node_feats+'_'+edge_feats

    ##graph formation
    sim_mat[sim_mat<sim_threshold] = 0
    G = nx.from_numpy_matrix(sim_mat)
    print('graph created')
    sys.stdout.flush()
    sim_arr = sim_mat.flatten()
    edge_weights = sim_arr[sim_arr!=0]
    print(edge_weights.shape)
    print('weights computed')
    sys.stdout.flush()
    g = DGLGraph(G)
    print('DGL graph created')
    sys.stdout.flush()
    g.edata['weights'] = torch.from_numpy(edge_weights.reshape(-1,1))
    g.ndata['features'] = torch.from_numpy(mat_nodes)
    g.ndata['mask_train'] = torch.from_numpy(np.array(mask_train).reshape(-1,1))
    g.ndata['mask_test'] = torch.from_numpy(np.array(mask_test).reshape(-1,1))
    g.ndata['mask_val'] = torch.from_numpy(np.array(mask_val).reshape(-1,1))
    g.ndata['labels'] = torch.from_numpy(np.array(labels).reshape(-1,1))

    print('graph weights attached')
    sys.stdout.flush()
    
    
    save_graphs(graph_name+'.gml', g)
    print(g)
    print('graph saved as\t', graph_name)
    sys.stdout.flush()
    



        
#################################################################################################################################################


