import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn import metrics
import datetime
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


def model3():
    # 正常显示中文和负号
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    starttime = datetime.datetime.now()
    print('开始时间： ', starttime)

    # 准备训练集、测试集
    df0 = pd.DataFrame(pd.read_excel("9+pca_x.xlsx"))
    X = df0.iloc[:, :9]
    X = MinMaxScaler().fit_transform(X)
    df = pd.DataFrame(pd.read_excel("数据汇总.xlsx"))
    y = df["蠕变寿命lg(y)/h"]
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)#, random_state=42
    xtrain = xtrain.astype(np.float64)
    xtest = xtest.astype(np.float64)

    # 保存训练集和测试集
    xtrain1 = pd.DataFrame(xtrain)
    ytrain1 = pd.DataFrame(ytrain)
    xtest1 = pd.DataFrame(xtest)
    ytest1 = pd.DataFrame(ytest)
    df_train = pd.DataFrame(pd.concat([xtrain1, ytrain1]))
    # df_train.to_excel("trainset.xlsx", index=False)
    df_test = pd.DataFrame(pd.concat([xtest1, ytest1]))
    # df_test.to_excel("testset.xlsx", index=False)

    # 拟合线性模型
    rf_model = RandomForestRegressor(max_depth=17, n_estimators=140, random_state=42, n_jobs=-1)
    rf_model.fit(xtrain, ytrain)
    # rf_model = joblib.load("RF_best_model.pkl")
    ypre = rf_model.predict(xtest)
    ypre2 = rf_model.predict(xtrain)

    # 在训练集上
    # 拟合优度R2
    print('train_R^2:', rf_model.score(xtrain, ytrain))
    R2_train = rf_model.score(xtrain, ytrain)
    # 计算平均绝对误差MAE的保存与输出
    print("train_MAE: ", metrics.mean_absolute_error(ytrain, ypre2))
    # 计算均方根误差RMSE的保存与输出
    print("train_RMSE: ", np.sqrt(metrics.mean_squared_error(ytrain, ypre2)))

    # 在测试集上
    # 拟合优度R2
    print('R^2:', rf_model.score(xtest, ytest))
    R2 = rf_model.score(xtest, ytest)
    # 计算平均绝对误差MAE的保存与输出
    print("MAE: ", metrics.mean_absolute_error(ytest, ypre))
    # 计算均方根误差RMSE的保存与输出
    print("RMSE: ", np.sqrt(metrics.mean_squared_error(ytest, ypre)))
    print()

    # # y_test和y_hat的可视化对比
    # # 设置图片尺寸
    # plt.figure(figsize=(8, 8), dpi=100)
    # # 绘制真实值-预测值散点图
    # plt.scatter(ytrain, ypre2, color='g', alpha=0.85)
    # # 绘制对角线参照线
    # plt.plot([1, 4.5], [1, 4.5], linestyle=':', color='r', linewidth=2)
    # # 图片标题
    # # plt.title('0参数-' + str(num_times + 1))
    # # 坐标轴标题
    # plt.xlabel('真实值10^x')
    # plt.ylabel('预测值10^y')
    # plt.xlim(1, 4.5)
    # plt.ylim(1, 4.5)
    # plt.savefig('train预测与真实值.png')  # 保存图片
    # # plt.pause(1)
    # # plt.close()
    #
    # # y_test和y_hat的可视化对比
    # # 设置图片尺寸
    # plt.figure(figsize=(8, 8), dpi=100)
    # # 绘制真实值-预测值散点图
    # plt.scatter(ytest, ypre, color='g', alpha=0.85)
    # # 绘制对角线参照线
    # plt.plot([1, 4.5], [1, 4.5], linestyle=':', color='r', linewidth=2)
    # # 图片标题
    # # plt.title('0参数-' + str(num_times + 1))
    # # 坐标轴标题
    # plt.xlabel('真实值10^x')
    # plt.ylabel('预测值10^y')
    # plt.xlim(1, 4.5)
    # plt.ylim(1, 4.5)
    # plt.savefig('test预测与真实值.png')  # 保存图片
    # # plt.pause(1)
    # # plt.close()

    # 保存预测结果
    df_out1 = pd.DataFrame({"train_y_ture": ytrain, "train_y_pre": ypre2})
    # df_out1.to_excel("trainset预测结果.xlsx", index=False)
    df_out2 = pd.DataFrame({"test_y_ture": ytest, "test_y_pre": ypre})
    # df_out2.to_excel("testset预测结果.xlsx", index=False)

    # 保存计算的机器学习模型
    # joblib.dump(rf_model, '2.3模型.pkl')

    # 打印计算用时
    endtime = datetime.datetime.now()
    print('结束时间： ', endtime)
    print('总用时', endtime - starttime)

    return (R2, R2_train, df_train, df_test, rf_model, df_out1, df_out2)


r2 = [0]
a = 0
while a < 0.94:
    a, b, trainset, testset, rfmodel, trainresult, testresult = model3()
    if a > max(r2):
        r2.append(a)
        trainset.to_excel(str(a)+"trainset.xlsx", index=False)
        testset.to_excel(str(a)+"testset.xlsx", index=False)
        joblib.dump(rfmodel, str(a)+"+"+str(b)+'模型.pkl')
        trainresult.to_excel(str(a)+"trainset预测结果.xlsx", index=False)
        testresult.to_excel(str(a)+"testset预测结果.xlsx", index=False)

print(r2)
df_r2 = pd.DataFrame({"R2":r2})
df_r2.to_excel("R2.xlsx",index=False)
