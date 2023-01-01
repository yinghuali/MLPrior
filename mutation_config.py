
dic_mutation_heart_rf = {
                   'n_estimators': [10, 20, 30, 40],
                   'max_depth': [5, 10, 20],
                   }


dic_mutation_heart_xgboost = {
                   'n_estimators': [1, 2, 3, 5, 10],
                   'max_depth': [2, 3],
                   'eta': [0.1, 3],
                   }

dic_mutation_heart_lgb = {
                   'n_estimators': [1, 2, 3, 5, 10],
                   'max_depth': [2, 3],
                   'num_leaves': [10, 30, 50],
                   'min_data_in_leaf': [40],
                   }


dic_mutation_adult_rf = {
                   'n_estimators': [5, 10, 15, 20, 25, 30],
                   'max_depth': [2, 3],
                   'max_features': [1, 2, 3],
                   'min_samples_split': [2, 3],
                   'min_samples_leaf': [1, 2, 3]
                   }


dic_mutation_adult_xgboost = {
                   'n_estimators': [5, 10, 15, 20, 25, 30],
                   'max_depth': [2, 3],
                   'eta': [0.1, 3],
                   'gamma': [1, 10, 20],
                   'min_child_weight': [1, 10],
                   'refresh_leaf': [0, 1]
                   }

dic_mutation_adult_lgb = {
                   'n_estimators': [5, 10, 15, 20, 25, 30],
                   'max_depth': [2, 3],
                   'num_leaves': [10, 30, 50],
                   'min_data_in_leaf': [40],
                   'bagging_fraction': [0.1, 0.9],
                   'pos_bagging_fraction': [0.1, 0.9],
                   'lambda_l1': [1, 10]
                   }

