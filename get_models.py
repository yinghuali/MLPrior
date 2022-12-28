import pandas as pd
import joblib
import pickle
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from mutation_config import *


path_data = 'data/adult.csv'
model_name = 'lr'

data_name = path_data.split('/')[-1].split('.')[0]


x_train, x_test, y_train, y_test, x_normalization_train, x_normalization_test, y_train_, y_test_ = get_adult_x_y(path_data)
# adult LR
# model = LogisticRegression(tol=1500)
# model.fit(x_normalization_train, y_train)
# joblib.dump(model, 'models/target_models/adult_lr.model')
#
# y_pre = model.predict(x_normalization_test)
# acc = accuracy_score(y_pre, y_test)
# print(acc)

# RF
model = RandomForestClassifier(n_estimators=3, max_features=2)
model.fit(x_normalization_train, y_train)
joblib.dump(model, 'models/target_models/adult_rf.model')
y_pre = model.predict(x_normalization_test)
acc = accuracy_score(y_pre, y_test)
print(acc)

# model = RandomForestClassifier(n_estimators=1, min_samples_split=6, max_features=1)
# model.fit(x_train, y_train)
# y_pre = model.predict(x_test)
# acc = accuracy_score(y_pre, y_test)
# print(acc)

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