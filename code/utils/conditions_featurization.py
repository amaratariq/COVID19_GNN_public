#!/usr/bin/env python3

import pandas as pd
import numpy as np
import datetime
import os
import sys
import shutil
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
icd = pd.read_csv('code/utils/ICD10_Groups.csv') #ICD10 hierarchy

def find_group(code):
    global icd
    group = ''
    letter = code[0]
    number = code[1:].split('.')[0]
    if number.isnumeric():
        number = (float(number))
        icd_sel = icd.loc[icd.SUBGROUP.str.startswith(letter)].copy()
        icd_sel = icd_sel.loc[(icd_sel.START_IDX.str.isnumeric()) & (icd_sel.END_IDX.str.isnumeric())].copy()
        icd_sel = icd_sel.loc[ (icd_sel.START_IDX.astype(float)<=number) & (icd_sel.END_IDX.astype(float)>=number)].copy()
        if len(icd_sel)>0:
            group = icd_sel.at[icd_sel.index[0], 'SUBGROUP']
        else:
            group = 'UNKNOWN'
    else:
        icd_sel = icd.loc[icd.SUBGROUP.str.startswith(letter)].copy()
        icd_sel = icd_sel.loc[(icd_sel.START_IDX.str.isnumeric()==False) & (icd_sel.END_IDX.str.isnumeric()==False)].copy()
        numheader = number[:-1]
        icd_sel = icd_sel.loc[(icd_sel.START_IDX.str.startswith(numheader)) & (icd_sel.END_IDX.str.startswith(numheader))].copy()
        if len(icd_sel)>0:
            group = icd_sel.at[icd_sel.index[0], 'SUBGROUP']
        else:
            group = 'UNKNOWN'
    return group
    
def conditions_feature_vector_per_insatnce(base_file, icd_file, out_file):
    """
    base_file: /path/to/file with one hospitalization per row
    icd_file: /path/to/icd codes file with one icd code per row
    out_file: /path/to/output file with one hsopitalization per row along with all ICD10 subgroups
    """
    ## use this as base file
    df = pd.read_csv(base_file)
    
    df_icd = pd.read_csv(icd_file, error_bad_lines=False)  ## all icd recorded /one icd code per row
    print('Length of diagnoses file:\t', len(df_icd), 'No of unique patients in diagnoses file:\t', len(df_icd.PATIENT_DK.unique()))
    sys.stdout.flush()

    

    df['admitDate'] = pd.to_datetime(df['ADMIT_DTM'], errors='coerce').dt.date
    df['xrayDate'] = pd.to_datetime(df['RADIOLOGY_DTM'], errors='coerce').dt.date

    

    df_icd = df_icd.loc[df_icd.PATIENT_DK.isin(df.PATIENT_DK.unique())]
    print('AFTER FILTERING WITH BASE FILE \n Length of diagnoses file:\t', len(df_icd), 'No of unique patients in diagnoses file:\t', len(df_icd.PATIENT_DK.unique()))
    df_icd['diagnosisDate'] = pd.to_datetime(df_icd['DIAGNOSIS_DTM'], errors = 'coerce').dt.date
    sys.stdout.flush()


        
    idx = 0
    df_icd_all = df_icd.copy()
    df_icd = df_icd_all.loc[df_icd_all.PATIENT_DK.isin(df.iloc[idx:min(idx+500,len(df))]['PATIENT_DK'])]
    print('Processing rows {} to {} of the base file'.format(idx, min(idx+500,len(df))))
    
    for c in icd.SUBGROUP.unique():
        df[c] = None
    for idx in range(0, len(df)):#i,j in df.iterrows():
        i = df.index[idx]
        pid = df.at[i, 'PATIENT_DK']
        admit_dt = df.at[i, 'admitDate']   
        xray_dt = df.at[i, 'xrayDate']
        st = admit_dt   
        ed = xray_dt
        temp = df_icd.loc[(df_icd.PATIENT_DK==pid) & (df_icd.diagnosisDate>=st)
                         & (df_icd.diagnosisDate<=ed)]

        if len(temp)>0:

            temp['SUBGROUPS'] = temp.DIAGNOSIS_CODE.apply(find_group)           
            temp2 = temp.drop_duplicates(subset=['DIAGNOSIS_CODE'])
            temp2['SUBGROUP'] = temp2.DIAGNOSIS_CODE.apply(find_group) 
            for s in temp2.SUBGROUP.unique():
                df.at[idx, s] = len(temp.loc[temp.DIAGNOSIS_CODE.isin(temp2.loc[temp2.SUBGROUP==s]['DIAGNOSIS_CODE'].unique())])

        if i%500==0:
            print('saving and updating')
            df.to_csv(out_file)
            df_icd = df_icd_all.loc[df_icd_all.PATIENT_DK.isin(df.iloc[idx:min(idx+500,len(df))]['PATIENT_DK'])]
            print('Processing rows {} to {} of the base file'.format(idx, min(idx+500,len(df))))
        sys.stdout.flush()

    df.to_csv(out_file)