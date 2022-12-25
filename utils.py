import pandas as pd
import os
from config import *


def get_model_path(path_dir_compile):
    model_path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.model'):
                    model_path_list.append(file_absolute_path)
    return model_path_list


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


# df = read_adult('data/adult.csv')
# df_normalization = get_df_normalization(df, ['income'])
