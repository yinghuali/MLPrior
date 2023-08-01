#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 10G

python mlprior_lr.py --path_data 'data/adult.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/bank.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/stroke.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'


python mlprior_lr.py --path_data 'data/noise/adult_mixture_noise_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_mixture_noise_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_mixture_noise_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_mixture_noise_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_mixture_noise_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_mixture_noise_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_mixture_noise_6.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_mixture_noise_7.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_mixture_noise_8.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_mixture_noise_9.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'

python mlprior_lr.py --path_data 'data/noise/bank_mixture_noise_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_mixture_noise_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_mixture_noise_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_mixture_noise_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_mixture_noise_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_mixture_noise_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_mixture_noise_6.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_mixture_noise_7.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_mixture_noise_8.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_mixture_noise_9.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'

python mlprior_lr.py --path_data 'data/noise/stroke_mixture_noise_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_mixture_noise_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_mixture_noise_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_mixture_noise_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_mixture_noise_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_mixture_noise_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_mixture_noise_6.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_mixture_noise_7.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_mixture_noise_8.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_mixture_noise_9.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'

