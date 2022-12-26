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


path_data = 'data/adult.csv'
model_name = 'lr'

data_name = path_data.split('/')[-1].split('.')[0]


def get_adult_x_y():
    df = read_adult(path_data)
    df_normalization = get_df_normalization(df, ['income'])
    x = df.to_numpy()[:, 0:-1]
    x_normalization = df_normalization.to_numpy()[:, 0:-1]
    y = df.to_numpy()[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=17)
    x_normalization_train, x_normalization_test, y_train_, y_test_ = train_test_split(x_normalization, y, test_size=0.3, random_state=17)
    return x_train, x_test, y_train, y_test, x_normalization_train, x_normalization_test, y_train_, y_test_


# adult LR
# x_train, x_test, y_train, y_test, x_normalization_train, x_normalization_test, y_train_, y_test_ = get_adult_x_y()
# model = LogisticRegression(tol=1500)
# model.fit(x_normalization_train, y_train)
# joblib.dump(model, 'models/target_models/lr.model')
#
# y_pre = model.predict(x_normalization_test)
# acc = accuracy_score(y_pre, y_test)
# print(acc)

