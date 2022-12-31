from mutation_config import *
import pandas as pd
import argparse
import joblib
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from mutation_config import *
from utils import *

ap = argparse.ArgumentParser()
ap.add_argument("--path_data", type=str)
ap.add_argument("--label_name", type=str)
ap.add_argument("--path_save_model", type=str)
ap.add_argument("--model_name", type=str)
args = ap.parse_args()

# python get_mutation_models.py --path_data 'data/wine.csv' --label_name 'quality' --path_save_model 'models/mutation_models/wine/'
# python get_mutation_models.py --path_data 'data/adult.csv' --label_name 'income' --path_save_model 'models/mutation_models/adult/'


path_data = args.path_data
label_name = args.label_name
path_save_model = args.path_save_model
model_name = args.model_name

data_name = path_data.split('/')[-1].split('.')[0]

x, y = read_data(path_data, label_name)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


def get_mutation_config(data_name):
    if data_name=='wine':
        dic_rf = dic_mutation_wine_rf
        dic_xgboost = dic_mutation_wine_xgboost
        dic_lgb = dic_mutation_wine_lgb
    elif data_name=='adult':
        dic_rf = dic_mutation_adult_rf
        dic_xgboost = dic_mutation_adult_xgboost
        dic_lgb = dic_mutation_adult_lgb
    else:
        print('please input data')
        dic_rf, dic_xgboost, dic_lgb = None, None, None

    return dic_rf, dic_xgboost, dic_lgb


def main():
    dic_rf, dic_xgboost, dic_lgb = get_mutation_config(data_name)
    list_dic_rf = list(ParameterGrid(dic_rf))
    list_dic_xgboost = list(ParameterGrid(dic_xgboost))
    list_dic_lgb = list(ParameterGrid(dic_lgb))

    for i in range(len(list_dic_rf)):
        tmp_dic = list_dic_rf[i]
        save_model_name = path_save_model+'rf/' + str(i) + '.model'
        model = RandomForestClassifier(**tmp_dic)
        model.fit(x_train, y_train)
        joblib.dump(model, save_model_name)
        pickle.dump(tmp_dic, open(path_save_model+'rf/' + str(i) + '_config.pkl', 'wb'))

    for i in range(len(list_dic_xgboost)):
        tmp_dic = list_dic_xgboost[i]
        save_model_name = path_save_model+'xgboost/' + str(i) + '.model'
        model = XGBClassifier(**tmp_dic)
        model.fit(x_train, y_train)
        joblib.dump(model, save_model_name)
        pickle.dump(tmp_dic, open(path_save_model+'xgboost/' + str(i) + '_config.pkl', 'wb'))

    for i in range(len(list_dic_lgb)):
        tmp_dic = list_dic_lgb[i]
        save_model_name = path_save_model+'lgb/' + str(i) + '.model'
        model = LGBMClassifier(**tmp_dic)
        model.fit(x_train, y_train)
        joblib.dump(model, save_model_name)
        pickle.dump(tmp_dic, open(path_save_model+'lgb/' + str(i) + '_config.pkl', 'wb'))


if __name__ == '__main__':
    main()
