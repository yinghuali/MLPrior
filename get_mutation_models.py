from mutation_config import *
import pandas as pd
import joblib
import pickle
from utils import get_adult_x_y
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
model_name = 'rf'
path_save_model = 'models/mutation_models/adult/'

data_name = path_data.split('/')[-1].split('.')[0]

if data_name=='adult':
    x_train, x_test, y_train, y_test, x_normalization_train, x_normalization_test, y_train_, y_test_ = get_adult_x_y(path_data)
    dic = dic_mutation_rf


def main():
    j = 0
    list_dic = list(ParameterGrid(dic))
    for i in range(len(list_dic)):
        tmp_dic = list_dic[i]
        save_model_name = path_save_model + model_name + '_' + str(i) + '.model'
        if model_name=='rf':
            model = RandomForestClassifier(**tmp_dic)
        model.fit(x_normalization_train, y_train)
        joblib.dump(model, save_model_name)
        pickle.dump(tmp_dic, open(path_save_model + model_name + '_' + str(i) + '_config.pkl', 'wb'))
        j += 1


if __name__ == '__main__':
    main()