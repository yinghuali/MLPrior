#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 10G

python ablation_decision_tree.py --path_data '../data/adult.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model '../models/target_models/adult_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python ablation_decision_tree.py --path_data '../data/bank.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model '../models/target_models/bank_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python ablation_decision_tree.py --path_data '../data/stroke.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model '../models/target_models/stroke_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'

python ablation_knn.py --path_data '../data/adult.csv' --model_name 'knn' --n_mutants 5  --mutation_level 20 --path_target_model '../models/target_models/adult_knn.model' --mutation_cols_level 5 --n_mutants_data 5 --label_name 'income'
python ablation_knn.py --path_data '../data/bank.csv' --model_name 'knn' --n_mutants 5  --mutation_level 20 --path_target_model '../models/target_models/bank_knn.model' --mutation_cols_level 5 --n_mutants_data 5 --label_name 'y'
python ablation_knn.py --path_data '../data/stroke.csv' --model_name 'knn' --n_mutants 5  --mutation_level 20 --path_target_model '../models/target_models/stroke_knn.model' --mutation_cols_level 5 --n_mutants_data 5 --label_name 'stroke'

python ablation_lr.py --path_data '../data/adult.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model '../models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python ablation_lr.py --path_data '../data/bank.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model '../models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python ablation_lr.py --path_data '../data/stroke.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model '../models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'

python ablation_nb.py --path_data '../data/adult.csv' --model_name 'nb' --n_mutants 100 --path_target_model '../models/target_models/adult_nb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python ablation_nb.py --path_data '../data/bank.csv' --model_name 'nb' --n_mutants 100 --path_target_model '../models/target_models/bank_nb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python ablation_nb.py --path_data '../data/stroke.csv' --model_name 'nb' --n_mutants 100 --path_target_model '../models/target_models/stroke_nb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'

python ablation_xgb.py --path_data '../data/adult.csv' --model_name 'xgb' --n_mutants 100 --path_target_model '../models/target_models/adult_xgb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python ablation_xgb.py --path_data '../data/bank.csv' --model_name 'xgb' --n_mutants 100 --path_target_model '../models/target_models/bank_xgb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python ablation_xgb.py --path_data '../data/stroke.csv' --model_name 'xgb' --n_mutants 100 --path_target_model '../models/target_models/stroke_xgb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'

