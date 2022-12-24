import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


path_data = 'data/adult.csv'

data_name = path_data.split('/')[-1].split('.')[0]


def get_adult_x_y():
    df = read_adult(path_data)
    df_normalization = get_df_normalization(df, ['income'])
    x = df.to_numpy()[:, 0:-1]
    x_normalization = df_normalization.to_numpy()[:, 0:-1]
    y = df.to_numpy()[:, -1]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=17)
    train_x_normalization, test_x_normalization, train_y_, test_y_ = train_test_split(x_normalization, y, test_size=0.3, random_state=17)
    return train_x, test_x, train_y, test_y, train_x_normalization, test_x_normalization, train_y_, test_y_


train_x, test_x, train_y, test_y, train_x_normalization, test_x_normalization, train_y_, test_y_ = get_adult_x_y()

