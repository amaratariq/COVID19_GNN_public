from xmlrpc.client import boolean
from source import GNN_weighted
import pickle as pkl
import os
import argparse



def apply_GNN():
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', type=str, required=True, help='set to False for testing ')
    parser.add_argument('--graph_name', type=str, required=True, help='graph_threshold_[threhsold value]_[label]_xray_[edge features list].gml')
    parser.add_argument('--target', type=str, required=True, help='discharged_in_3_days/admitted_to_ICU_in_3days/expired_in_3days')
    parser.add_argument('--model_path', type=str, required=False, default=None, help='path/to/pretrained/model')
    
    args = parser.parse_args()

    do_train = args.do_train == 'true'
    if do_train==False and args.model_path==None:
        print('either train a new model or provide path to pretrianed model')
    else:
        graph_path = 'graphs/'+args.graph_name
        if os.path.exists(graph_path)==False:
            print('run build_graph.py for graph formation')
        else:
            GNN_weighted.main(do_train=args.do_train, graph_path = graph_path, target=args.target, model_path = args.model_path)
        

if __name__ == "__main__":
    apply_GNN()   