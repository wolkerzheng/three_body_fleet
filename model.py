# -*- coding: utf-8 -*-
from sklearn.externals import joblib
from pandas import Series, DataFrame
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


data_type = {'jin': np.float64, 'wei': np.float64}
counter = 0
score = 0

# load test set
test_data = pd.read_csv('./test/test.csv', header=0, low_memory=False)
test_data.columns = ['rid', 'uid', 'mid', 'time', 'jing', 'wei', 'wifi', 's_wifi', 'c_wifi', 'is_conn', 'time_block']

test_data['y_label'] = 1
test_data = test_data.drop(['time'], axis=1)


for file_name in os.listdir('./feature'):
    current_mall_id = file_name.split(".")[0]  # m_xxxx
    file_path = './feature/' + file_name
    feature = pd.read_csv(file_path, header=0, dtype=data_type, low_memory=False)
    feature.columns = ['uid', 'sid', 'time', 'jing', 'wei', 'wifi', 'cid', 'mid', 's_wifi', 'c_wifi', 'is_conn',
                       'time_block']

    sid_label_encoder = LabelEncoder()
    feature['sid'] = sid_label_encoder.fit_transform(feature['sid'])

    y = feature.sid
    X = feature.drop(['uid', 'sid', 'time', 'wifi', 'mid', 'cid'], axis=1)
    # only keeps jing, wei, s_wifi, c_wifi, is_conn, time_block

    clf = RandomForestClassifier(n_estimators=9, max_depth=25, min_samples_split=3, random_state=0)
    clf.fit(X, y)

    current_test = test_data[test_data.mid == current_mall_id]
    x_test = current_test.drop(['rid', 'uid', 'mid', 'wifi', 'y_label'], axis=1)
    print(current_mall_id)
    print(x_test.size)
    if x_test.size > 0:
        y_pred = clf.predict(x_test)
        y_label = sid_label_encoder.inverse_transform(y_pred)

        test_data.loc[test_data.mid == current_mall_id, 'y_label'] = y_label

    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #
    # clf = RandomForestClassifier(n_estimators=9, max_depth=25, min_samples_split=3, random_state=0)
    # clf = clf.fit(x_train, y_train)
    #
    # y_pred = clf.predict(x_test)
    #
    # s = accuracy_score(y_test, y_pred)
    #
    # counter += y_test.size
    # score += y_test.size * s

    # save model
    # model_name = file_name.split(".")[0] + ".model"
    # model_path = "./model/" + model_name
    # joblib.dump(clf, model_path)

# print(score / counter)

pred = test_data[['rid', 'y_label']]
pred.to_csv("./test/raw_result.csv", index=None)




