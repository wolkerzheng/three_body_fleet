# -*- coding: utf-8 -*-
from pandas import Series, DataFrame
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


data_type = {'jin': np.float64, 'wei': np.float64}

feature = pd.read_csv('./feature/feature1.csv', header=1, dtype=data_type, low_memory=False)
feature.columns = ['uid', 'sid', 'time', 'jin', 'wei', 'mid', 's_wifi', 'c_wifi', 'is_conn']

label_encoder = LabelEncoder()
feature['sid'] = label_encoder.fit_transform(feature['sid'])
feature['uid'] = label_encoder.fit_transform(feature['uid'])

y = feature.sid
X = feature.drop(['sid', 'time', 'mid'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = RandomForestClassifier(n_estimators=5, max_depth=10, min_samples_split=5, random_state=0)
clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

s = accuracy_score(y_test, y_pred)

print(s)





