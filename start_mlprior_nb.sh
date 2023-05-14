#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 10G

python mlprior_nb.py --path_data 'data/adult.csv' --model_name 'nb' --n_mutants 100 --path_target_model 'models/target_models/adult_nb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_nb.py --path_data 'data/heart.csv' --model_name 'nb' --n_mutants 100 --path_target_model 'models/target_models/heart_nb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_nb.py --path_data 'data/bank.csv' --model_name 'nb' --n_mutants 100 --path_target_model 'models/target_models/bank_nb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_nb.py --path_data 'data/stroke.csv' --model_name 'nb' --n_mutants 100 --path_target_model 'models/target_models/stroke_nb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_nb.py --path_data 'data/fairness_age.csv' --model_name 'nb' --n_mutants 100 --path_target_model 'models/target_models/bank_nb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_nb.py --path_data 'data/fairness_gender.csv' --model_name 'nb' --n_mutants 100 --path_target_model 'models/target_models/adult_nb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_nb.py --path_data 'data/fairness_race.csv' --model_name 'nb' --n_mutants 100 --path_target_model 'models/target_models/adult_nb.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'

