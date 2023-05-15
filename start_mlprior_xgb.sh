#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 10G

python mlprior_xgb.py --path_data 'data/adult.csv' --model_name 'xgb' --n_mutants 100 --path_target_model 'models/target_models/adult_xgb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_xgb.py --path_data 'data/bank.csv' --model_name 'xgb' --n_mutants 100 --path_target_model 'models/target_models/bank_xgb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_xgb.py --path_data 'data/stroke.csv' --model_name 'xgb' --n_mutants 100 --path_target_model 'models/target_models/stroke_xgb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
