# -*- coding: utf-8 -*-
from pandas import Series

fixed_wifi_dict = {}


def get_all_wifi(data):
    _wifi = {"NaN": 41}
    for wifi_list in data.wifi:
        wifi_info_list = wifi_list.split(";")
        for wifi_info in wifi_info_list:
            wifi_name, _, _ = wifi_info.split("|")
            if wifi_name in _wifi:
                _wifi[wifi_name] += 1
            else:
                _wifi[wifi_name] = 1
    return _wifi


def strongest_wifi(x):
    wifi_list = x.split(";")
    max_intensity = -9999
    max_name = None
    for wifi in wifi_list:
        name, intensity, connection = wifi.split("|")
        intensity = int(intensity)
        if intensity > max_intensity and name in fixed_wifi_dict:
            max_intensity = intensity
            max_name = name
    return max_name or "NaN"


def connected_wifi(x):
    _wifi_infos = x.split(";")
    max_intensity = -99999
    max_name = "FFF"
    for _wifi_info in _wifi_infos:
        name, intensity, connection = _wifi_info.split("|")
        if connection == "true" and name in fixed_wifi_dict:
            return name
    return "NaN"


def is_connnected(x):
    _wifi_infos = x.split(";")
    for _wifi_info in _wifi_infos:
        name, _, connection = _wifi_info.split("|")
        if connection == "true" and name in fixed_wifi_dict:
            return 1
    return 0

block_size = 60


def get_time_block(x):
    _, hour_mini = x.split(" ")
    hour, mini = hour_mini.split(":")
    return (int(hour) * 60 + int(mini)) // block_size


# just a naive implementation, todo
def wifi_filter(wifi_dict):
    obj = Series(wifi_dict)
    # threshold = 30
    obj = obj[obj > 30]
    global fixed_wifi_dict
    fixed_wifi_dict = obj.to_dict()
    return fixed_wifi_dict
