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
# model = RandomForestClassifier(n_estimators=3, max_features=2)
# model.fit(x_normalization_train, y_train)
# joblib.dump(model, 'models/target_models/adult_rf.model')
# y_pre = model.predict(x_normalization_test)
# acc = accuracy_score(y_pre, y_test)
# print(acc)

# xgboost
# model = XGBClassifier(max_depth=2, n_estimators=2)
# model.fit(x_normalization_train, y_train)
# joblib.dump(model, 'models/target_models/adult_xgboost.model')
# y_pre = model.predict(x_normalization_test)
# acc = accuracy_score(y_pre, y_test)
# print(acc)

# lightgbm
# model = LGBMClassifier(max_depth=3, n_estimators=5)
# model.fit(x_normalization_train, y_train)
# joblib.dump(model, 'models/target_models/adult_lgb.model')
# y_pre = model.predict(x_normalization_test)
# acc = accuracy_score(y_pre, y_test)
# print(acc)


# model = joblib.load('models/target_models/adult_xgboost.model')
# y_pre = model.predict(x_normalization_test)
# acc = accuracy_score(y_pre, y_test)
# print(acc)


