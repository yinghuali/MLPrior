import pandas as pd
import random


def get_noise_data(path, label_col_name, save_path_dir, file_name, n_col):
    df = pd.read_csv(path)
    cols_list = list(df.columns)
    cols_list.remove(label_col_name)
    for i in range(len(cols_list)):
        if i > 5:
            break
        else:
            select_cols = random.sample(cols_list, n_col)
            pdf = df.copy()
            for col in select_cols:
                tmp_list = pdf[col]
                random.shuffle(tmp_list)
                pdf[col] = tmp_list

            save_path = save_path_dir + file_name + '_' + str(n_col) + '_' + str(i) + '.csv'
            pdf.to_csv(save_path, index=False, sep=',')


if __name__ == '__main__':
    get_noise_data('./data/adult.csv', 'income', './data/noise/', 'adult', 1)
    get_noise_data('./data/bank.csv', 'y', './data/noise/', 'bank', 1)
    get_noise_data('./data/stroke.csv', 'stroke', './data/noise/', 'stroke', 1)
    get_noise_data('./data/heart.csv', 'label', './data/noise/', 'heart', 1)

    get_noise_data('./data/adult.csv', 'income', './data/noise/', 'adult', 2)
    get_noise_data('./data/bank.csv', 'y', './data/noise/', 'bank', 2)
    get_noise_data('./data/stroke.csv', 'stroke', './data/noise/', 'stroke', 2)
    get_noise_data('./data/heart.csv', 'label', './data/noise/', 'heart', 2)

    get_noise_data('./data/adult.csv', 'income', './data/noise/', 'adult', 3)
    get_noise_data('./data/bank.csv', 'y', './data/noise/', 'bank', 3)
    get_noise_data('./data/stroke.csv', 'stroke', './data/noise/', 'stroke', 3)
    get_noise_data('./data/heart.csv', 'label', './data/noise/', 'heart', 3)

    get_noise_data('./data/adult.csv', 'income', './data/noise/', 'adult', 4)
    get_noise_data('./data/bank.csv', 'y', './data/noise/', 'bank', 4)
    get_noise_data('./data/stroke.csv', 'stroke', './data/noise/', 'stroke', 4)
    get_noise_data('./data/heart.csv', 'label', './data/noise/', 'heart', 4)
