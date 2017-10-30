# -*- coding: utf-8 -*-
from pandas import Series, DataFrame
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data_type = {'jin': np.float64, 'wei': np.float64}

feature = pd.read_csv('./feature/feature1.csv', header=1, dtype=data_type, low_memory=False)
feature.columns = ['uid', 'sid', 'time', 'jin', 'wei', 'mid', 's_wifi', 'c_wifi', 'is_conn']

label_encoder = LabelEncoder()
feature['sid'] = label_encoder.fit_transform(feature['sid'])
feature['uid'] = label_encoder.fit_transform(feature['uid'])

y = feature.sid
X = feature.drop(['sid', 'time', 'mid'], axis=1)

clf = RandomForestClassifier(n_estimators=5, max_depth=10, min_samples_split=5, random_state=0)
clf = clf.fit(X, y)

y_pred = clf.predict(X)

s = accuracy_score(y, y_pred)

print(s)





