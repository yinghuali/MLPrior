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


if __name__ == '__main__':
    get_adult_gender_exchange( './data/adult.csv', 'gender', './data/gender_exchange_adult.csv')



