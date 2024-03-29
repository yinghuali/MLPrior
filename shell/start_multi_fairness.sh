#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 10G


python mlprior_lr.py --path_data 'data/age_exchange_diabetes.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/diabetes_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'
python mlprior_lr.py --path_data 'data/gender_exchange_diabetes.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/diabetes_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'

python mlprior_decision_tree.py --path_data 'data/age_exchange_diabetes.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/diabetes_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'
python mlprior_decision_tree.py --path_data 'data/gender_exchange_diabetes.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/diabetes_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'

python mlprior_knn.py --path_data 'data/age_exchange_diabetes.csv' --model_name 'knn' --n_mutants 5  --mutation_level 20 --path_target_model 'models/target_models/diabetes_knn.model' --mutation_cols_level 5 --n_mutants_data 5 --label_name 'Diabetes'
python mlprior_knn.py --path_data 'data/gender_exchange_diabetes.csv' --model_name 'knn' --n_mutants 5  --mutation_level 20 --path_target_model 'models/target_models/diabetes_knn.model' --mutation_cols_level 5 --n_mutants_data 5 --label_name 'Diabetes'


python mlprior_nb.py --path_data 'data/age_exchange_diabetes.csv' --model_name 'nb' --n_mutants 100 --path_target_model 'models/target_models/diabetes_nb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'
python mlprior_nb.py --path_data 'data/gender_exchange_diabetes.csv' --model_name 'nb' --n_mutants 100 --path_target_model 'models/target_models/diabetes_nb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'

python mlprior_xgb.py --path_data 'data/age_exchange_diabetes.csv' --model_name 'xgb' --n_mutants 100 --path_target_model 'models/target_models/diabetes_xgb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'
python mlprior_xgb.py --path_data 'data/gender_exchange_diabetes.csv' --model_name 'xgb' --n_mutants 100 --path_target_model 'models/target_models/diabetes_xgb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'

