# -*- coding: utf-8 -*-
from pandas import Series
from sklearn.neighbors import BallTree, DistanceMetric
from math import radians, cos, sin, asin, sqrt
import numpy as np

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


#transform [latitude, longitude] to radians
def latlng2radians(X):
	radiansX = []
	for latlng in X:
		lat1, lng1 = map(radians, latlng)
		radiansX.append([lat1, lng1])
	return radiansX

#X is train_data, leafsize is the param of BallTree
def build_tree(radiansX, leafsize):
	#use radians build KNN tree
	tree = BallTree(radiansX, leaf_size = leafsize, metric = DistanceMetric.get_metric('haversine'))
	return tree

if  __name__ == "__main__":

	topK = 5
	
	np.random.seed(1)
	X = np.random.uniform(20, 70, (100,2)) + np.random.normal(0, 10, (100,2))
	testX = np.random.uniform(40,50, (20,2))
	radiansX = latlng2radians(X)
	
	#Note that the haversine distance metric requires data in the form of [latitude, longitude] and both inputs and outputs are in units of radians
	#tree = BallTree(radiansX, leaf_size = 5, metric = DistanceMetric.get_metric('haversine'))
	tree = build_tree(radiansX, 5)
	dist, index = tree.query(testX, k = topK, return_distance = True)
	print(index)
	print(dist)
