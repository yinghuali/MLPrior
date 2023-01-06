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


path_list = get_path('/Users/yinghua.li/Desktop/nw/missing')
print(len(path_list))

cmd_list = ['mv '+i+' result/missing/' for i in path_list]
for i in cmd_list:
    os.system(i)




