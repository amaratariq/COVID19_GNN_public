#!/usr/bin/env python3

import pandas as pd
import numpy as np
import datetime
import os
import sys
import shutil
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date


dfcpt_groups = pd.read_csv("code/utils/CPT_group_structure.csv") #cpt code structure
print('No of CPT subgroups:\t', len(dfcpt_groups))
def to_cpt_group(x):
    out=None
    if type(x)==str and x.isnumeric():
        x = int(x)
        temp = dfcpt_groups.loc[(dfcpt_groups['Low']<=x) & (dfcpt_groups['High']>=x) & (dfcpt_groups['Modifier'].isna())]
        if len(temp)>0:
            out = temp.at[temp.index[0], 'Subgroup']
    elif type(x) == str and x[:-1].isnumeric():
        m = x[-1]
        x = int(x[:-1])
        temp = dfcpt_groups.loc[(dfcpt_groups['Low']<=x) & (dfcpt_groups['High']>=x) & (dfcpt_groups['Modifier']==m)]
        if len(temp)>0:
            out = temp.at[temp.index[0], 'Subgroup']
    return out
sys.stdout.flush()

def procedures_feature_vector_per_instance(base_file, cpt_file, out_file):
    """
    base_file: /path/to/file with one hospitalization per row
    cpt_file: /path/to/cpt codes file with one cpt code per row
    out_file: /path/to/output file with one hsopitalization per row along with all CPT subgroups
    """
    ## use this as base file
    df = pd.read_csv(base_file)

    df_cpt = pd.read_csv(cpt_file, error_bad_lines=False)  ## all cpt recorded /one cpt code per row
    print('Length of procedures file:\t', len(df_cpt), 'No of unique patients in procedure file:\t', len(df_cpt.PATIENT_DK.unique()))
    sys.stdout.flush()

    df['admitDate'] = pd.to_datetime(df['ADMIT_DTM'], errors='coerce').dt.date
    df['xrayDate'] = pd.to_datetime(df['RADIOLOGY_DTM'], errors='coerce').dt.date
    
    
    df_cpt = df_cpt.loc[df_cpt.PATIENT_DK.isin(df.PATIENT_DK.unique())]
    print('AFTER FILTERING WITH BASE FILE \n Length of procedures file:\t', len(df_cpt), 'No of unique patients in procedure file:\t', len(df_cpt.PATIENT_DK.unique()))
    sys.stdout.flush()
    
    df_cpt['procedureDate'] = pd.to_datetime(df_cpt['PROCEDURE_DTM'], errors = 'coerce').dt.date

    idx = 0
    df_cpt_all = df_cpt.copy()
    df_cpt = df_cpt_all.loc[df_cpt_all.PATIENT_DK.isin(df.iloc[idx:min(idx+1000,len(df))]['PATIENT_DK'])]
    print('Processing rows {} to {} of the base file'.format(idx, min(idx+1000,len(df))))
    
    for c in dfcpt_groups.Subgroup.unique():
        df[c] = None

    for idx in range(0, len(df)):
        i = df.index[idx]
        pid = df.at[i, 'PATIENT_DK']
        admit_dt = df.at[i, 'admitDate']   

        xray_dt = df.at[i, 'xrayDate']
        ed = xray_dt
        st = admit_dt     

        temp = df_cpt.loc[(df_cpt.PATIENT_DK==pid) & (df_cpt.procedureDate>=st) & (df_cpt.procedureDate<=ed)]
        temp2 = temp.drop_duplicates(subset=['PROCEDURE_CODE'])
        if len(temp)>0:
            temp2['SUBGROUP'] = temp2.PROCEDURE_CODE.apply(to_cpt_group) 
            for s in temp2.SUBGROUP.unique():
                df.at[idx, s] = len(temp.loc[temp.PROCEDURE_CODE.isin(temp2.loc[temp2.SUBGROUP==s]['PROCEDURE_CODE'].unique())])
    
        if i%1000==0:
            print('saving and updating')
            df.to_csv(out_file)
            df_cpt = df_cpt_all.loc[df_cpt_all.PATIENT_DK.isin(df.iloc[idx:min(idx+1000,len(df))]['PATIENT_DK'])]
            print('Processing rows {} to {} of the base file'.format(idx, min(idx+1000,len(df))))
        sys.stdout.flush()

    df.to_csv(out_file)
    