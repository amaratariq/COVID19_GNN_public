from utils import demo_featurization
from utils import procedures_featurization
from utils import conditions_featurization
from utils import xray_featurization


from source import merge_processed_data

import argparse
import pandas as pd



def preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort_file', type=str, required=True, help='name of csv file containig all chest X-rays of COVID_19 positive hospitalized patients, with patient ID and hospital admission and discharge timestamps ')
    parser.add_argument('--demo_file', type=str, required=True, help='name of csv file containig all demogrpahics')
    parser.add_argument('--cpt_file', type=str, required=True, help='name of csv file containig all procedure')
    parser.add_argument('--icd_file', type=str, required=True, help='name of csv file containig all conditions')
    
    
    args = parser.parse_args()

    header_data_raw_ehr = 'data/raw/ehr/'
    header_data_raw_xray = 'data/raw/xray/'
    header_data_proc = 'data/'

    
    print('processing demographics')
    demo_featurization.demo_feature_vector_per_instance(header_data_proc+args.cohort_file, header_data_raw_ehr+args.demo_file, header_data_proc+'demo_file.csv')
    print('processing procedures')
    procedures_featurization.procedures_feature_vector_per_instance(header_data_proc+args.cohort_file, header_data_raw_ehr+args.cpt_file, header_data_proc+'cpt_file.csv')
    print('processing conditions')
    conditions_featurization.conditions_feature_vector_per_instance(header_data_proc+args.cohort_file, header_data_raw_ehr+args.icd_file, header_data_proc+'icd_file.csv')
    print('processing image')
    xray_featurization.xray_feature_vector_per_instance(header_data_proc+args.cohort_file, header_data_proc+'xray_file.csv')

    merge_processed_data.merge_data(header_data_proc+args.cohort_file, 
                                    header_data_proc+'demo_file.csv',
                                    header_data_proc+'cpt_file.csv',
                                    header_data_proc+'icd_file.csv',
                                    header_data_proc+'xray_file.csv',
                                    header_data_proc+'cohort_file_w_ehr_xray.csv')

if __name__ == "__main__":
    preprocess()   