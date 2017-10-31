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

for file_name in os.listdir('./feature'):
    file_path = './feature/' + file_name
    feature = pd.read_csv(file_path, header=0, dtype=data_type, low_memory=False)
    feature.columns = ['uid', 'sid', 'time', 'jin', 'wei', 'wifi', 'cid', 'mid', 's_wifi', 'c_wifi', 'is_conn', 'time_block']

    label_encoder = LabelEncoder()
    feature['sid'] = label_encoder.fit_transform(feature['sid'])
    feature['uid'] = label_encoder.fit_transform(feature['uid'])
    feature['cid'] = label_encoder.fit_transform(feature['cid'])

    y = feature.sid
    X = feature.drop(['sid', 'time', 'wifi', 'mid'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = RandomForestClassifier(n_estimators=9, max_depth=25, min_samples_split=3, random_state=0)
    clf = clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    s = accuracy_score(y_test, y_pred)

    counter += y_test.size
    score += y_test.size * s

    # save model
    model_name = file_name.split(".")[0] + ".model"
    model_path = "./model/" + model_name
    joblib.dump(clf, model_path)

print(score / counter)






