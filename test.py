import pandas as pd
import os


def get_path(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.csv'):
                    path_list.append(file_absolute_path)
    return path_list


path_list = get_path('result/')
print(len(path_list))
path_list = [i for i in path_list if 'fairness' not in i and 'missing' not in i and 'original' not in i and 'table' not in i]
print(len(path_list))
path_list = [i for i in path_list if len(i.split('/'))==2]
print(len(path_list))
for i in path_list:
    cmd = 'rm ' + i
    os.system(cmd)





