import pandas as pd
import random


def get_mixture_noise_data(path, n_data, ratio, label_col_name, save_path):
    df = pd.read_csv(path)
    cols_list = list(df.columns)
    cols_list.remove(label_col_name)
    select_cols_list = []
    for col in cols_list:
        if df[col].dtype == 'int64':
            select_cols_list.append(col)
    n = int(len(df) * ratio)
    for s in range(n_data):
        df_tmp = df.copy()
        for i in range(n):
            cols_name = random.sample(select_cols_list, 1)[0]
            idx = random.randint(0, len(df))
            try:
                df_tmp[cols_name][idx] = int(df_tmp[cols_name][idx]*random.uniform(0.5, 1)) + df_tmp[cols_name][idx]
            except:
                continue
        df_tmp.to_csv(save_path+'_'+str(s)+'.csv', index=False,sep=',')


def get_mixture_noise_data_dh(path, n_data, ratio, label_col_name, save_path):
    df = pd.read_csv(path)
    cols_list = list(df.columns)
    cols_list.remove(label_col_name)
    select_cols_list = []
    for col in cols_list:
        if df[col].dtype == 'int64' or df[col].dtype == 'float':
            select_cols_list.append(col)
    n = int(len(df) * ratio)
    for s in range(n_data):
        df_tmp = df.copy()
        for i in range(n):
            cols_name = random.sample(select_cols_list, 1)[0]
            idx = random.randint(0, len(df))
            try:
                df_tmp[cols_name][idx] = int(df_tmp[cols_name][idx]*random.uniform(0.5, 1)) + df_tmp[cols_name][idx]
            except:
                continue
        df_tmp.to_csv(save_path+'_'+str(s)+'.csv', index=False,sep=',')


if __name__ == '__main__':
    # get_mixture_noise_data('./data/adult.csv', 10, 0.3, 'income', './data/noise/adult_mixture_noise')
    # get_mixture_noise_data('./data/bank.csv', 10, 0.3, 'y', './data/noise/bank_mixture_noise')
    # get_mixture_noise_data('./data/stroke.csv', 10, 0.3, 'stroke', './data/noise/stroke_mixture_noise')

    get_mixture_noise_data_dh('./data/diabetes.csv', 10, 0.3, 'Diabetes', './data/noise/diabetes_mixture_noise')
    get_mixture_noise_data_dh('./data/heartbeat.csv', 10, 0.3, 'y', './data/noise/heartbeat_mixture_noise')




