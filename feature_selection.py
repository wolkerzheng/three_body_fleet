from pandas import Series, DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# test on a small set, where all shop belongs to one mail
# mail1 's id is m_1021 and it has 4924 records, 7 columns
data = pd.read_csv("./subset/mail1.csv", header=0, low_memory=False)

# [1.0] encode wifi_names
def get_all_wifi():
    _wifi = {"NaN": 0}
    i = 0
    for wifi_list in data.wifi:
        wifi_info_list = wifi_list.split(";")
        for wifi_info in wifi_info_list:
            wifi_name, _, _ = wifi_info.split("|")
            if wifi_name in _wifi:
                pass
            else:
                i += 1
                _wifi[wifi_name] = i
    return _wifi

wifi_dict = get_all_wifi()
wifi_names = list(wifi_dict.keys())
label_encoder = LabelEncoder()
label_encoder.fit(wifi_names)

# [2.0] two feature: strongest wifi and connected wifi

# handle wifi infos
wifi_infos = data.wifi


def strongest_wifi(x):
    wifi_list = x.split(";")
    max_intensity = -9999
    max_name = "FFF"
    for wifi in wifi_list:
        name, intensity, connection = wifi.split("|")
        intensity = int(intensity)
        if intensity > max_intensity:
            max_intensity = intensity
            max_name = name
    return max_name


def connected_wifi(x):
    _wifi_infos = x.split(";")
    max_intensity = -99999
    max_name = "FFF"
    for _wifi_info in _wifi_infos:
        name, intensity, connection = _wifi_info.split("|")
        if connection == "true":
            return name
    return "NaN"


def is_connnected(x):
    _wifi_infos = x.split(";")
    for _wifi_info in _wifi_infos:
        _, _, connection = _wifi_info.split("|")
        if connection == "true":
            return 1
    return 0


data['s_wifi'] = data.wifi.apply(strongest_wifi)
data['c_wifi'] = data.wifi.apply(connected_wifi)

# [3.0] use previously fitted label encoder to encode two newly extracted wifi feature
data.s_wifi = label_encoder.transform(data.s_wifi)
data.c_wifi = label_encoder.transform(data.c_wifi)

# [4.0] connected to wifi or not
data['is_conn'] = data.wifi.apply(is_connnected)

data = data.drop(['wifi'], axis=1)

data.to_csv("./feature/feature1.csv", index=None)
