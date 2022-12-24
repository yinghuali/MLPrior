import pandas as pd
from config import *


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
