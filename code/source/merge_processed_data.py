import pandas as pd
import numpy as np
import datetime
import os
import sys


def merge_data(base_file, demo_file, cpt_file, icd_file, xray_file, out_file):#


    df = pd.read_csv(base_file)
    df_demo = pd.read_csv(demo_file)
    df_cpt = pd.read_csv(cpt_file)
    df_icd = pd.read_csv(icd_file)
    df_xray = pd.read_csv(xray_file)


    df['xrayDate'] = pd.to_datetime(df['RADIOLOGY_DTM'], errors='coerce').dt.date
    df_demo['xrayDate'] = pd.to_datetime(df_demo['RADIOLOGY_DTM'], errors='coerce').dt.date
    df_cpt['xrayDate'] = pd.to_datetime(df_cpt['RADIOLOGY_DTM'], errors='coerce').dt.date
    df_icd['xrayDate'] = pd.to_datetime(df_icd['RADIOLOGY_DTM'], errors='coerce').dt.date
    df_xray['xrayDate'] = pd.to_datetime(df_xray['RADIOLOGY_DTM'], errors='coerce').dt.date
   

    cols = set(df_demo.columns) - set(df.columns)
    cols = ['PATIENT_DK', 'xrayDate']+list(cols)
    df = df.merge(df_demo[cols], on=['PATIENT_DK', 'xrayDate'], how='left')
    df = df.loc[:,~df.columns.duplicated()].copy()

    cols = set(df_cpt.columns) - set(df.columns)
    cols = ['PATIENT_DK', 'xrayDate']+list(cols)
    df = df.merge(df_cpt[cols], on=['PATIENT_DK', 'xrayDate'], how='left')
    df = df.loc[:,~df.columns.duplicated()].copy()

    cols = set(df_icd.columns) - set(df.columns)
    cols = ['PATIENT_DK', 'xrayDate']+list(cols)
    df = df.merge(df_icd[cols], on=['PATIENT_DK', 'xrayDate'], how='left')
    df = df.loc[:,~df.columns.duplicated()].copy()

    cols = set(df_xray.columns) - set(df.columns)
    cols = ['PATIENT_DK', 'xrayDate']+list(cols)
    df = df.merge(df_xray[cols], on=['PATIENT_DK', 'xrayDate'], how='left')
    df = df.loc[:,~df.columns.duplicated()].copy()

    df.to_csv(out_file)

