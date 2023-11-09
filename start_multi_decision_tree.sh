#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 10G


python mlprior_decision_tree.py --path_data 'data/noise/diabetes_mixture_noise_0.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/diabetes_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'
python mlprior_decision_tree.py --path_data 'data/noise/diabetes_mixture_noise_1.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/diabetes_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'
python mlprior_decision_tree.py --path_data 'data/noise/diabetes_mixture_noise_2.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/diabetes_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'
python mlprior_decision_tree.py --path_data 'data/noise/diabetes_mixture_noise_3.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/diabetes_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'
python mlprior_decision_tree.py --path_data 'data/noise/diabetes_mixture_noise_4.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/diabetes_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'
python mlprior_decision_tree.py --path_data 'data/noise/diabetes_mixture_noise_5.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/diabetes_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'
python mlprior_decision_tree.py --path_data 'data/noise/diabetes_mixture_noise_6.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/diabetes_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'
python mlprior_decision_tree.py --path_data 'data/noise/diabetes_mixture_noise_7.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/diabetes_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'
python mlprior_decision_tree.py --path_data 'data/noise/diabetes_mixture_noise_8.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/diabetes_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'
python mlprior_decision_tree.py --path_data 'data/noise/diabetes_mixture_noise_9.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/diabetes_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'Diabetes'


python mlprior_decision_tree.py --path_data 'data/noise/heartbeat_mixture_noise_0.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/heartbeat_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_decision_tree.py --path_data 'data/noise/heartbeat_mixture_noise_1.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/heartbeat_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_decision_tree.py --path_data 'data/noise/heartbeat_mixture_noise_2.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/heartbeat_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_decision_tree.py --path_data 'data/noise/heartbeat_mixture_noise_3.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/heartbeat_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_decision_tree.py --path_data 'data/noise/heartbeat_mixture_noise_4.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/heartbeat_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_decision_tree.py --path_data 'data/noise/heartbeat_mixture_noise_5.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/heartbeat_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_decision_tree.py --path_data 'data/noise/heartbeat_mixture_noise_6.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/heartbeat_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_decision_tree.py --path_data 'data/noise/heartbeat_mixture_noise_7.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/heartbeat_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_decision_tree.py --path_data 'data/noise/heartbeat_mixture_noise_8.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/heartbeat_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_decision_tree.py --path_data 'data/noise/heartbeat_mixture_noise_9.csv' --model_name 'dtree' --n_mutants 100  --ratio_mutation_node 0.1 --path_target_model 'models/target_models/heartbeat_dtree.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'

