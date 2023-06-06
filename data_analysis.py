# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 18:16:38 2021

@author: Administrator
"""

import numpy as np
import pandas as pd
import copy
import sys

test_file = sys.argv[1]
result_file = sys.argv[2]

# 训练
data = pd.read_csv('C:/Users/Administrator/Desktop/大数据计算/任务5数据/hw1/train.csv')

# 提取数据
data = data.iloc[:, 3:]

# 清洗数据
data[data=='NR'] = 0

# 数据类型转换
for i in range(len(data)):
    for j in range(len(data.iloc[i])):
        data.iloc[i][j] = float(data.iloc[i][j])

data2 = data.to_numpy()

new_data = [[] for _ in range(18)]

# 转换数据的样式
for i in range(len(data2)):
    r = i % 18
    for j in range(len(data2[i])):
        new_data[r].append(data2[i][j])

new_data_2 = copy.deepcopy(new_data)

# 规范化数据
for r in new_data:
    max_value = max(r)
    min_value = min(r)
    value = max_value - min_value
    if value == 0:
        continue
    else:
        for i in range(len(r)):
            r[i] = r[i] / value
 
        
xs = np.empty([12 * 471, 18 *9+1], dtype=float)
ys = np.ones((12*471,))
# ys = np.empty([12*471, 1], dtype = float)

# 数据集制作
for i in range(len(new_data)):
    for j in range(12 * 471):
        if i == 9:
            if j > 8:
                ys[j-9] = new_data[i][j]
        xs[j][i*9+1:(i+1)*9+1] = new_data[i][j:j+9]

for x in xs:
    x[0] = 1
    

x_train = xs[:4521]
x_vali = xs[4521:]

y_train = ys[:4521]
y_vali = ys[4521:]

learn_rate = 1
w = np.ones((163,))
w /= 6000
loss_value = np.inf

min_w = None

for i in range(10):
    
    w_tmp = copy.deepcopy(w)
    for k in range(2000):
        pre_std = x_train.dot(w_tmp) - y_train
        # 求损失值
        loss_sum = 0 
        for j in range(len(pre_std)):
            loss_sum += pre_std[j]** 2    
        loss_value_tmp = loss_sum / 2 /4521
        
        if loss_value > loss_value_tmp:
            loss_value = loss_value_tmp
            min_w = copy.deepcopy(w_tmp)
        print(loss_value, loss_value_tmp, learn_rate, i, k) 
        # 梯度下降
        for t in range(len(w_tmp)):
            for tt in range(len(pre_std)):
                pre_std[tt] *= x_train[tt][t]
            w_tmp[t] -= learn_rate * np.sum(pre_std) / 4521
    learn_rate -= 0.1 * learn_rate
    
    
# 结果


data_2= pd.read_csv(test_file, header=None, names=range(-2,9))

# 提取数据
data_2 = data_2.iloc[:, 2:]

# 清洗数据
data_2[data_2=='NR'] = 0

# 数据类型转换
for i in range(len(data_2)):
    for j in range(len(data_2.iloc[i])):
        data_2.iloc[i][j] = float(data_2.iloc[i][j])

data_2 = data_2.to_numpy()

new_data_2 = [[] for _ in range(18)]

# 转换数据的样式
for i in range(len(data_2)):
    r = i % 18
    for j in range(len(data_2[i])):
        new_data_2[r].append(data_2[i][j])

new_data_3 = copy.deepcopy(new_data_2)
# 保存数据，后面存入文件回复数值
values = 0
# 规范化数据
for r in range(len(new_data_2)):
    max_value = max(new_data_2[r])
    min_value = min(new_data_2[r])
    value = max_value - min_value
    if r == 9:
        values = value
    if value == 0:
        continue
    else:
        for i in range(len(new_data_2[r])):
           new_data_2[r][i] = new_data_2[r][i] / value
 
        
xs_2 = np.empty([12 * 171, 18 *9+1], dtype=float)
ys_2 = np.ones((12*171,))
# ys = np.empty([12*171, 1], dtype = float)

# 数据集制作
for i in range(len(new_data_2)):
    for j in range(12 * 171):
        if i == 9:
            if j > 8:
                ys_2[j-9] = new_data_2[i][j]
        xs_2[j][i*9+1:(i+1)*9+1] = new_data_2[i][j:j+9]

for x in xs_2:
    x[0] = 1

    

# 存入数据

result = xs_2.dot(min_w)
import csv

with open(result_file, 'w+') as csv_file:
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(['id', '结果'])
    for i in range(len(result)):
        csv_writer.writerow([f'id_{i}', result[i] * values])