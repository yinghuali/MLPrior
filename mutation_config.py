
dic_mutation_adult_rf = {
                   'n_estimators': [1, 2, 3, 4, 5, 6, 7],
                   'max_depth': [2, 3],
                   'max_features': [1, 2, 3],
                   'min_samples_split': [2, 3],
                   'min_samples_leaf': [1, 2, 3]
                   }


dic_mutation_adult_xgboost = {
                   'n_estimators': [1, 2, 3, 4, 5],
                   'max_depth': [2, 3],
                   'eta': [0.1, 3],
                   'gamma': [1, 10, 20],
                   'min_child_weight': [1, 10],
                   'refresh_leaf': [0, 1]
                   }

dic_mutation_adult_lgb = {
                   'n_estimators': [10, 20, 30, 40],
                   'max_depth': [2, 3],
                   'num_leaves': [10, 30, 50],
                   'min_data_in_leaf': [40],
                   'bagging_fraction': [0.1, 0.9],
                   'pos_bagging_fraction': [0.1, 0.9],
                   'lambda_l1': [1, 10]
                   }

dic_mutation_adult_dt = {
                   'criterion': ['gini', 'entropy', 'log_loss'],
                   'splitter': ['best', 'random'],
                   'min_samples_leaf': [5, 6, 7],
                   'max_features': [1, 2]
                   }






# n_estimators = 100, *,
# criterion = "gini",
# max_depth = None,
# min_samples_split = 2,
# min_samples_leaf = 1,
# min_weight_fraction_leaf = 0.,
# max_features = "auto",
# max_leaf_nodes = None,
# min_impurity_decrease = 0.,
# min_impurity_split = None,
# bootstrap = True,
# oob_score = False,
# n_jobs = None,
# random_state = None,
# verbose = 0,
# warm_start = False,
# class_weight = None,
# ccp_alpha = 0.0,
# max_samples = None):

