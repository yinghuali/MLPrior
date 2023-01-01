#python mlprior_tree.py --path_data 'data/adult.csv' --model_name 'lgb' --path_target_model 'models/target_models/adult_lgb.model' --path_mutation_models 'models/mutation_models/adult/lgb' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
#python mlprior_tree.py --path_data 'data/adult.csv' --model_name 'rf' --path_target_model 'models/target_models/adult_rf.model' --path_mutation_models 'models/mutation_models/adult/rf' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'
#python mlprior_tree.py --path_data 'data/adult.csv' --model_name 'xgboost' --path_target_model 'models/target_models/adult_xgboost.model' --path_mutation_models 'models/mutation_models/adult/xgboost' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'income'

python mlprior_tree.py --path_data 'data/heart.csv' --model_name 'lgb' --path_target_model 'models/target_models/heart_lgb.model' --path_mutation_models 'models/mutation_models/heart/lgb' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'
python mlprior_tree.py --path_data 'data/heart.csv' --model_name 'rf' --path_target_model 'models/target_models/heart_rf.model' --path_mutation_models 'models/mutation_models/heart/rf' --mutation_cols_level 5 --n_mutants_data 10 --label_name 'label'
python mlprior_tree.py --path_data 'data/heart.csv' --model_name 'xgboost' --path_target_model 'models/target_models/heart_xgboost.model' --path_mutation_models 'models/mutation_models/heart/xgboost' --mutation_cols_level 5 --n_mutants_data 20 --label_name 'label'








