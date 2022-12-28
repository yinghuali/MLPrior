import pandas as pd
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from config import *


def apfd(error_idx_list, pri_idx_list):
    error_idx_list = list(error_idx_list)
    pri_idx_list = list(pri_idx_list)
    n = len(pri_idx_list)
    m = len(error_idx_list)
    TF_list = [pri_idx_list.index(i) for i in error_idx_list]
    apfd = 1 - sum(TF_list)*1.0 / (n*m) + 1 / (2*n)
    return apfd


def get_idx_miss_class(target_pre, test_y):
    idx_miss_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != test_y[i]:
            idx_miss_list.append(i)
    idx_miss_list.append(i)
    return idx_miss_list


def read_adult(path_adult):
    df = pd.read_csv(path_adult)
    df['workclass'] = [dic_adult_workclass[i] for i in df['workclass']]
    df['education'] = [dic_adult_education[i] for i in df['education']]
    df['marital-status'] = [dic_adult_marital_status[i] for i in df['marital-status']]
    df['occupation'] = [dic_adult_occupation[i] for i in df['occupation']]
    df['relationship'] = [dic_adult_relationship[i] for i in df['relationship']]
    df['race'] = [dic_adult_race[i] for i in df['race']]
    df['gender'] = [dic_adult_gender[i] for i in df['gender']]
    df['native-country'] = [dic_adult_native_country[i] for i in df['native-country']]
    df['income'] = [dic_adult_income[i] for i in df['income']]
    return df


def get_df_normalization(pdf, protect_cols_list):
    df = pdf.copy()
    cols = list(df.columns)
    select_cols = [i for i in cols if i not in protect_cols_list]
    for i in select_cols:
        df[i] = (df[i]-min(df[i])) / (max(df[i])-min(df[i]))
    return df


def get_adult_x_y(path_data):
    df = read_adult(path_data)
    df_normalization = get_df_normalization(df, ['income'])
    x = df.to_numpy()[:, 0:-1]
    x_normalization = df_normalization.to_numpy()[:, 0:-1]
    y = df.to_numpy()[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=17)
    x_normalization_train, x_normalization_test, y_train_, y_test_ = train_test_split(x_normalization, y, test_size=0.3, random_state=17)
    return x_train, x_test, y_train, y_test, x_normalization_train, x_normalization_test, y_train_, y_test_


def get_adult_x_y_all(path_data):
    df = read_adult(path_data)
    df_normalization = get_df_normalization(df, ['income'])
    x = df.to_numpy()[:, 0:-1]
    x_normalization = df_normalization.to_numpy()[:, 0:-1]
    y = df.to_numpy()[:, -1]
    return x, x_normalization, y


def get_mutation_feature(model_pre_np, target_pre):
    feature_list = []
    for i in range(len(target_pre)):
        tmp_list = []
        for j in range(len(model_pre_np)):
            if model_pre_np[j][i] != target_pre[i]:
                tmp_list.append(1)
            else:
                tmp_list.append(0)
        feature_list.append(tmp_list)
    feature_vec = np.array(feature_list)
    return feature_vec


def get_miss_lable(target_train_pre, target_test_pre, y_train, y_test):
    idx_miss_train_list = get_idx_miss_class(target_train_pre, y_train)
    idx_miss_test_list = get_idx_miss_class(target_test_pre, y_test)
    miss_train_label = [0]*len(y_train)
    for i in idx_miss_train_list:
        miss_train_label[i]=1
    miss_train_label = np.array(miss_train_label)

    miss_test_label = [0]*len(y_test)
    for i in idx_miss_test_list:
        miss_test_label[i]=1
    miss_test_label = np.array(miss_test_label)

    return miss_train_label, miss_test_label, idx_miss_test_list


def get_mutation_data(x, mutation_cols_level, n_mutants_data):
    res_list = []
    idx_waiting_list = list(range(len(x[0])))
    for _ in range(n_mutants_data):
        n_select_cols = random.sample(mutation_cols_level, 1)[0]
        res = np.copy(x)
        idx_list = random.sample(idx_waiting_list, n_select_cols)
        for idx in idx_list:
            res[:, idx] = 0
        res_list.append(res)
    res_np = np.array(res_list)
    return res_np


def get_model_path(path_dir_compile):
    model_path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.model'):
                    model_path_list.append(file_absolute_path)
    return model_path_list