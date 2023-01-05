import pandas as pd
import random


def get_missing_data(path, n_missing_col, label_col_name, save_path_dir, file_name):
    df = pd.read_csv(path)
    cols_list = list(df.columns)
    cols_list.remove(label_col_name)
    if n_missing_col == 1:
        for i in range(len(cols_list)):
            tmp_df = df.copy()
            tmp_df[cols_list[i]] = 0
            save_path = save_path_dir+file_name+'_'+str(n_missing_col)+'_'+str(i)+'.csv'
            tmp_df.to_csv(save_path, index=False, sep=',')
            #if i > 20:
            if i > 5:
                break
    elif n_missing_col > 1:
        #for i in range(20):
        for i in range(5):
            select_cols_list = random.sample(cols_list, n_missing_col)
            tmp_df = df.copy()
            for col in select_cols_list:
                tmp_df[col] = 0
            save_path = save_path_dir + file_name + '_' + str(n_missing_col) + '_' + str(i) + '.csv'
            tmp_df.to_csv(save_path, index=False, sep=',')
    else:
        print('Please input correct parameters')


if __name__ == '__main__':
    get_missing_data('./data/adult.csv', 1, 'income', './data/missing/', 'adult')
    get_missing_data('./data/adult.csv', 2, 'income', './data/missing/', 'adult')
    get_missing_data('./data/adult.csv', 3, 'income', './data/missing/', 'adult')
    get_missing_data('./data/adult.csv', 4, 'income', './data/missing/', 'adult')

    get_missing_data('./data/bank.csv', 1, 'y', './data/missing/', 'bank')
    get_missing_data('./data/bank.csv', 2, 'y', './data/missing/', 'bank')
    get_missing_data('./data/bank.csv', 3, 'y', './data/missing/', 'bank')
    get_missing_data('./data/bank.csv', 4, 'y', './data/missing/', 'bank')

    get_missing_data('./data/stroke.csv', 1, 'stroke', './data/missing/', 'stroke')
    get_missing_data('./data/stroke.csv', 2, 'stroke', './data/missing/', 'stroke')
    get_missing_data('./data/stroke.csv', 3, 'stroke', './data/missing/', 'stroke')
    get_missing_data('./data/stroke.csv', 4, 'stroke', './data/missing/', 'stroke')

    get_missing_data('./data/heart.csv', 1, 'label', './data/missing/', 'heart')
    get_missing_data('./data/heart.csv', 2, 'label', './data/missing/', 'heart')
    get_missing_data('./data/heart.csv', 3, 'label', './data/missing/', 'heart')
    get_missing_data('./data/heart.csv', 4, 'label', './data/missing/', 'heart')