import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import datetime
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# 正常显示中文和负号
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

starttime = datetime.datetime.now()
print('开始时间： ', starttime)
print('正在进行主成分分析...')

df_merge = pd.read_excel('数据汇总.xlsx')

X = df_merge.iloc[:, :16]
X = MinMaxScaler().fit_transform(X)

# print(X)
# y = df_merge['蠕变寿命lg(y)/h']
# xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=21)
# xtrain = xtrain.astype(np.float64)
# xtest = xtest.astype(np.float64)


estimator = PCA(n_components=9)
pca_train_x = estimator.fit_transform(X)
# print(pca_train_x)
df_out = pd.DataFrame(pca_train_x)
df_out.to_excel("9+"+"pca_x.xlsx", index=False)

print(1, estimator.components_, '\n')
print(2, "explained_variance_ratio_: ", estimator.explained_variance_ratio_, '\n')
print(3, "explained_variance_: ", estimator.explained_variance_, '\n')
# print(4, estimator.components_, '\n')
