# COVID19_GNN
 GNN model for COVID19 Disease Trajectory Prediction

## requirements

Place cohort file in data/raw
*   cohort_file -- all xrays with corresponding patient and hospitalization details, split (train/test/val) indicator and label (discharge from hsopital/ICU admission/mortality in 3 days of xray date)
Place data files in data/raw/ehr
*   demo_file -- person file
*   cpt_file   -- procedure_occurrence
*   icd_file -- condition_occurrence
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
