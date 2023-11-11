# MLPrior
MLPrior is our proposed test prioritization approach specifically for classical machine learning (ML) classifiers.

## Main Requirements
    python 3.7
    Scikit-learn 1.0
    XGBoost 1.5.0

##  Repository catalogue
    data: dataset.
    impact_mlprior: parameter analysis.
    models: the evaluated ML models.
    shell: scripts to obtain the experimental results of the paper.
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
    run_MLPrior.sh: run MLPrior to get all results.

## How to run MLPrior
### Step1: Result directory preparation:  
```sh mkdirFile.sh```

### Step2: Run MLPrior
All results will be saved in the 'result' directory.
```
sh run_MLPrior.sh
```
