from typing import Optional
from source import graph_formation
import pickle as pkl
import os
import argparse


def build_graph():
   
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--sim_threshold', type=float, required=True, help='threshold to decide edge between two nodes')
    parser.add_argument('--edge_feats', type=str, required=True, help='comma separated identifiers from (demo, cpt, icd, lab, vit, days) for edge formation')
    parser.add_argument('--label', type=str, required=True, help='discharge_in_3days/admitted_to_ICU_in_3days/expired_in_3days')

    
    args = parser.parse_args()
    
    print(args)
    
    graph_header = 'graphs/'
    
    
    base_file = 'data/processed/cohort_file_w_ehr_xray.csv' # assumes preprocessign has been done to create this file
    if os.path.exists(base_file)==False:
        print('run preprocessing code to generate base file with EHR data')
    else:
        dct = pkl.load(open('code/utils/encoders.pkl', 'rb'))
        enc_demo = dct['enc_demo']
        print('building graph ...')
        graph_formation.graph_creation(base_file, enc_demo, sim_threshold=args.sim_threshold, graph_header = graph_header, label = args.label, edge_feats=args.edge_feats)
        
if __name__ == "__main__":
    build_graph()   