import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import joblib

# 正常显示中文与负号
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

best_para = []

for i in [9]:
    # 读取数据，准备训练集、验证集、测试集
    df0 = pd.DataFrame(pd.read_excel(str(i)+"+pca_x.xlsx"))
    X = df0.iloc[:, :i]
    X = MinMaxScaler().fit_transform(X)
    df = pd.DataFrame(pd.read_excel("数据汇总.xlsx"))
    y = df["蠕变寿命lg(y)/h"]
    xtrian, xvalid, ytrain, yvalid = train_test_split(X, y, test_size=0.25) #, random_state=42
    xtrian = xtrian.astype(np.float64)
    xvalid = xvalid.astype(np.float64)

    # ！！！！ RF模型 ！！！！ #
    # ！！！！ RF模型 ！！！！ #
    print("正在训练：RF模型……"+str(i))
    rf_param = {"max_depth": list(np.arange(5, 51,3)),
                'n_estimators': list(np.arange(10, 300,5))}
    rf_model = RandomForestRegressor(random_state=42)
    grid_search_rf = GridSearchCV(rf_model, rf_param, n_jobs=-1, cv=10)
    grid_search_rf.fit(xtrian, ytrain)  # 训练，找到最优参数，同时用最优参数实例化一个回归器
    rf_ypre = grid_search_rf.predict(xvalid)
    # 保存模型
    # joblib.dump(grid_search_rf.best_estimator_, "RF_best_model.pkl")
    # 模型评估参数
    print("Valid set score: ", grid_search_rf.score(xvalid, yvalid))
    print("Valid set MAE: ", metrics.mean_absolute_error(yvalid, rf_ypre))
    print("Valid set RMSE: ", np.sqrt(metrics.mean_squared_error(yvalid, rf_ypre)))

    print("Test set score: {:.2f}".format(grid_search_rf.score(xvalid, yvalid)))
    print("Best parameters: {}".format(grid_search_rf.best_params_))
    print("Best score on train set: {:.2f}".format(grid_search_rf.best_score_))
    best_para.append(grid_search_rf.best_params_)

    print(str(i)+"RF模型训练完成！\n")

df_out = pd.DataFrame({"para":best_para})
df_out.to_excel("best_para.xlsx",index=False)
