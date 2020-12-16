import pandas as pd
import numpy as np
import torch
import matplotlib.pylab as plt
#读取数据
train=pd.read_csv('./data/train_v9rqX0R.csv')
print(train.columns)
#检查每个变量中缺失值的比例
print('每个变量值缺失的比例：{}'.format(train.isnull().sum(axis=0)/len(train)*100))