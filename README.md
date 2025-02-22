# COVID19_GNN
 GNN model for COVID19 Disease Trajectory Prediction


![](covid19_trajectory.jpg)
* Branched predictive framework for clinical event prediction for COVID-19 patients

## requirements

Place cohort file in data/raw
*   cohort_file -- all xrays with corresponding patient and hospitalization details, split (train/test/val) indicator and label (discharge from hsopital/ICU admission/mortality in 3 days of xray date)
Place data files in data/raw/ehr
*   demo_file -- demographics.csv
*   cpt_file   -- procedures.csv
*   icd_file -- conditions.csv
Place xrays as png files with RADIOLOGY_ID from cohort file as file name in data/raw/xray/


## preprocess
```
python3 code/preprocess.py --cohort_file file_name --demo_file file_name --cpt_file file_name --icd_file file_name
```
## build test graph from scratch

```
python3 code/build_graph.py --sim_threshold 0.9 --label discharge_in_3days --edge_feats cpt
```
Use appropriate edge feature indicator (cpt/icd/demo)
Use appropriate label indicator (discharge_in_3days/admitted_to_ICU_in_3days/expired_in_3days)

## apply GNN
```
python3 code/apply_GNN.py --do_train true --graph_name graph_threshold_0.9_discharged_in_3days_xray_cpt.gml
```
Use appropriate label indicator (discharge_in_3days/admitted_to_ICU_in_3days/expired_in_3days)


## Data
The model is designed for a cohort with following characteristics

- patients with positive RT-PCR test for COVID-19
- patients admitted to the hospital
- patients undergoing regular chest x-ray examination (within 3 day intervals) 

Labels must be decided on per-X-ray vased by counting number of days between an X-ray and discharge from hospital/admission into ICU/mortality.

code/source/graph_formation.csv lists ICD and CPT subgroups found to be frequenct in our dataset. 

The cohort file expects one chest X-ray per row. Expected columns for data files are provided in sample csv files.

The model predicts a label for each chest X-ray. Essentially, predictions are made for a patient whenever they undergo a chest X-ray exam, thus building a predicitve trajectory.
