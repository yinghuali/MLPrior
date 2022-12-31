# Datasets

1. Adult  48842 income
(https://archive.ics.uci.edu/ml/datasets/adult）
    'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
    'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-white'},
                                 {1.0: 'Male', 0.0: 'Female'}]

2. wine 1143 quality 效果差删掉，可能原因数据集太小
(https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

2. Bank 49732 y
(https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets?select=train.csv)


ing. diabetes_data.csv
https://www.kaggle.com/datasets/prosperchuks/health-dataset?select=diabetes_data.csv




Name      Protected attribute(s)        #Features               Favorable label                 Majority label          Size
Adult     Sex, Race                     14                      1 (income > 50K)                0 (75.2%)               45,222
Compas    Sex, Race                     10                      0 (no recidivism)               0 (54.5%)               6,167
German    Sex                           20                      1 (good credit)                 1 (70.0%)               1,000
Bank      Age                           20                      1 (subscriber)                  0 (87.3%)               30,488
Mep       Race                          41                      1 (utilizer)                    0 (82.8%)               15,830