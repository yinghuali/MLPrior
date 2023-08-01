import pandas as pd
import random


def get_adult_gender_exchange(path_csv, col_name, path_save):
    df = pd.read_csv(path_csv)
    df_Male = df[df[col_name] == 'Male']
    df_Female = df[df[col_name] == 'Female']

    Male_index = list(df_Male.index)
    select_Male_index = random.sample(Male_index, int(len(Male_index)/2))

    Female_index = list(df_Female.index)
    select_Female_index = random.sample(Female_index, int(len(Female_index)/2))

    for i in select_Male_index:
        df.loc[i, col_name] = 'Female'

    for i in select_Female_index:
        df.loc[i, col_name] = 'Male'

    df.to_csv(path_save, index=False)


def get_stroke_sex_exchange(path_csv, col_name, path_save):
    df = pd.read_csv(path_csv)
    df_0 = df[df[col_name] == 0.0]
    df_1 = df[df[col_name] == 1.0]

    index_0 = list(df_0.index)
    select_0_index = random.sample(index_0, int(len(index_0) / 2))

    index_1 = list(df_1.index)
    select_1_index = random.sample(index_1, int(len(index_1) / 2))

    for i in select_0_index:
        df.loc[i, col_name] = 1.0

    for i in select_1_index:
        df.loc[i, col_name] = 0.0

    df.to_csv(path_save, index=False)


def get_age_exchange(path_csv, col_name, path_save):
    df = pd.read_csv(path_csv)
    for i in range(len(df)):
        if 18 <= df.loc[i, col_name] <= 29:
            df.loc[i, col_name] = random.randint(30, 59)
        elif 30 <= df.loc[i, col_name] <= 59:
            df.loc[i, col_name] = random.randint(18, 29)
        else:
            pass
    df.to_csv(path_save, index=False)


if __name__ == '__main__':
    get_adult_gender_exchange('./data/adult.csv', 'gender', './data/gender_exchange_adult.csv')
    get_stroke_sex_exchange('./data/stroke.csv', 'sex', './data/gender_exchange_stroke.csv')
    get_age_exchange('./data/adult.csv', 'age', './data/age_exchange_adult.csv')
    get_age_exchange('./data/bank.csv', 'age', './data/age_exchange_bank.csv')

