#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p bigmem
#SBATCH --mem 300G

python mlprior_lr.py --path_data 'data/noise/adult_1_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_1_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_1_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_1_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_1_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_1_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'

python mlprior_lr.py --path_data 'data/noise/adult_2_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_2_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_2_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_2_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_2_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_2_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'

python mlprior_lr.py --path_data 'data/noise/adult_3_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_3_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_3_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_3_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_3_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_3_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'

python mlprior_lr.py --path_data 'data/noise/adult_4_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_4_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_4_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_4_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_4_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
python mlprior_lr.py --path_data 'data/noise/adult_4_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 10 --path_target_model 'models/target_models/adult_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'

python mlprior_lr.py --path_data 'data/noise/bank_1_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_1_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_1_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_1_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_1_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_1_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'

python mlprior_lr.py --path_data 'data/noise/bank_2_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_2_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_2_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_2_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_2_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_2_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'

python mlprior_lr.py --path_data 'data/noise/bank_3_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_3_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_3_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_3_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_3_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_3_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'

python mlprior_lr.py --path_data 'data/noise/bank_4_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_4_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_4_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_4_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_4_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'
python mlprior_lr.py --path_data 'data/noise/bank_4_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/bank_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'y'

python mlprior_lr.py --path_data 'data/noise/stroke_1_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_1_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_1_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_1_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_1_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_1_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'

python mlprior_lr.py --path_data 'data/noise/stroke_2_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_2_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_2_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_2_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_2_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_2_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'

python mlprior_lr.py --path_data 'data/noise/stroke_3_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_3_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_3_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_3_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_3_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_3_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'

python mlprior_lr.py --path_data 'data/noise/stroke_4_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_4_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_4_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_4_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_4_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'
python mlprior_lr.py --path_data 'data/noise/stroke_4_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/stroke_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'stroke'

python mlprior_lr.py --path_data 'data/noise/heart_1_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_1_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_1_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_1_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_1_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_1_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'

python mlprior_lr.py --path_data 'data/noise/heart_2_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_2_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_2_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_2_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_2_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_2_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'

python mlprior_lr.py --path_data 'data/noise/heart_3_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_3_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_3_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_3_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_3_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_3_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'

python mlprior_lr.py --path_data 'data/noise/heart_4_0.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_4_1.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_4_2.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_4_3.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_4_4.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_lr.py --path_data 'data/noise/heart_4_5.csv' --model_name 'lr' --n_mutants 20  --mutation_level 3 --path_target_model 'models/target_models/heart_lr.model' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'

