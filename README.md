[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10150392.svg)](https://doi.org/10.5281/zenodo.10150392)
# MLPrior
MLPrior is our proposed test prioritization approach specifically for classical machine learning (ML) classifiers.

## Main Requirements
    python 3.7
    Scikit-learn 1.0
    XGBoost 1.5.0

##  Repository catalogue
    data: dataset for evaluating MLPrior.
    impact_mlprior: parameter analysis for MLPrior.
    models: classical ML models for evaluating MLPrior.
    shell: all scripts used to run MLPrior in the experiments.
    ----------------------
    get_fairness_data.py: script for getting fairness data.
    get_mixture_data.py: script for getting mixed noisy data.
    get_rank_idx.py: script for test input ranking.
    get_target_model.py: script for getting evaluated ML models.
    mlprior_decision_tree.py: script of MLPrior for Decision Tree model.
    mlprior_knn.py: script of MLPrior for KNN model.
    mlprior_lr.py: script of MLPrior for LR model.
    mlprior_nb.py: script of MLPrior for NB model.
    mlprior_xgb.py: script of MLPrior for XGBoost model.
    utils.py: tool script.
    ----------------------
    sh mkdirFile.sh: result' directory preparation.
    sh get_ML_models.sh: get evaluated ML models.
    run_MLPrior.sh: run MLPrior to get all results.

## How to run MLPrior
### Step1: 'result' directory preparation:  
```sh mkdirFile.sh```

### Step2: Get evaluated ML models.
```sh get_ML_models.sh```

### Step3: Run MLPrior
```sh run_MLPrior.sh```  
All results will be saved in the 'result' directory.  

## Reference
If our project is helpful to you, please consider citing our paper.
```
@article{mlprior,
title={Test input prioritization for Machine Learning Classifiers},
author={Dang, Xueqi and Li, Yinghua and Papadakis, Mike and Klein, Jacques and Bissyand{\'e}, Tegawend{\'e} F and Le Traon, Yves},
journal={IEEE Transactions on Software Engineering},
year={2024},
doi={10.1109/TSE.2024.3350019},
publisher={IEEE}
}
```