# Datasets

1. Adult  48842 income
(https://archive.ics.uci.edu/ml/datasets/adult）
    'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
    'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-white'},
                                 {1.0: 'Male', 0.0: 'Female'}]

2. heart 10000 label 心跳信号分类预测
https://tianchi.aliyun.com/competition/entrance/531883/information/


3. Bank 49732 y
(https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets?select=train.csv)
国际年龄划分是指将人的年龄分成不同的阶段的一种方式。这种方式通常用于统计、教育、医疗等领域。

def udf_age(i):
    if 18<=i<=29:
        return 65
    if i >= 30:
        return 25

下面是一个常见的国际年龄划分方式:

儿童：0-11 岁
青少年：12-17 岁
青年：18-29 岁
中年：30-59 岁
老年：60 岁以上
注意：这只是一种常见的国际年龄划分方式

4. stroke.csv  40907  stroke
https://www.kaggle.com/datasets/prosperchuks/health-dataset?select=stroke_data.csv


wine 1143 quality
(https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)



creditcard.csv 284807 Class

churn.csv 10000 客户流失 churn
https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset

patient.csv 3000 SOURCE
https://www.kaggle.com/datasets/manishkc06/patient-treatment-classification?select=training_set.csv




LR, SVM, RF, and two other very popular ML algorithms, i.e., Naive Bayes (NB) and Decision


