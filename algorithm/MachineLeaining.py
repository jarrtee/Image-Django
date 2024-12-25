import os

import numpy as np
import pandas as pd
import pymssql
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import linear_model, datasets
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, roc_auc_score
from sklearn.datasets import make_classification
from sqlalchemy import create_engine


#线性回归
def linear_regression():
    # 示例数据集
    # X为特征，y为目标变量
    X = np.array([[1, 1], [1, 2], [2, 3], [2, 4], [3, 5], [4, 6]])
    y = np.array([1, 2, 3, 4, 5, 6])

    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建并训练线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 进行预测
    y_pred = model.predict(X_test)

    # 评估模型
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # 输出模型的参数
    print("Model intercept:", model.intercept_)
    print("Model coefficients:", model.coef_)


def Linear_Regression_training():
    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]  #np.newaxis增加一维 // 需二维数组

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()  #普通最小二乘线性回归

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)
    #self = fit(self,x,y,sample_weight = None) 拟合线性模型
    #x -> 训练数据
    #y -> 目标标签
    #sample_weight -> 每个样本的权重
    #self -> 返回估计器的实例

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)
    #self = predict(self,x) 使用线性模型进行预测
    #x -> 样本数据
    #self -> 预测值

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    #loss = sklearn.metrics.mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', squared=True) 均方误差回归损失
    #y_true -> 真实目标值
    #y_pred -> 预测目标值
    #sample_weight -> 样本权重
    #multioutput -> 定义多个输出值的汇总
    #squared -> 若为True,则返回MSE值,若为False,返回RMSE值
    #loss -> 非负浮点值(最佳值为0.0)或浮点值数组,每个目标对应一个浮点值

    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
    #z = sklearn.metrics.r2_score(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')#R^2（确定系数）回归得分函数/最佳可能得分为1.0,并且可能为负
    #y_true -> 真实目标值
    #y_pred -> 预测目标值
    #sample_weight -> 样本权重
    #multioutput -> 定义多个输出分数的汇总,默认值为'uniform_average'
    #- ‘raw_values’:
    #如果是多输出格式的输入，则返回完整的分数集
    #- ‘uniform_average’:
    #所有产出的分数均以统一权重平均
    #- ‘variance_weighted’:
    #将所有输出的分数平均, 并按每个单独输出的方差加
    #z -> 如果‘multioutput’为‘raw_values’，则为R^2分数或分数的ndarray

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()


def AUC_Learning():
    # 生成一个不平衡的武侠数据集
    # 假设特征表示武功修炼时间、战斗胜率等，标签表示是否为高手
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.9, 0.1], random_state=42)

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练一个逻辑回归模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 预测测试集
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # 计算 ROC 曲线和 AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)

    # 可视化结果
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("ROC 曲线")
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel("假阳性率")
    plt.ylabel("真阳性率")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.title("AUC 值示意")
    plt.fill_between(fpr, tpr, color='blue', alpha=0.3)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {auc:.2f}")
    plt.xlabel("假阳性率")
    plt.ylabel("真阳性率")
    plt.legend(loc="lower right")

    #plt.tight_layout()
    plt.show()

    print(f"AUC: {auc:.2f}")


def SQL_FILE():
    #sql文件夹路径
    sql_path = './SQL'+'\\'
    #sql文件名
    sql_file = 'sql1.sql'

    #读取sql内容
    sql = open(sql_path + sql_file, 'r',encoding='utf-8')
    sqltxt = sql.read()#sqltxt为list类型

    #关闭文件
    sql.close()
    # list 转 str
    sql = "".join(sqltxt)
    print(sql)

    con = pymssql.connect(host="10.100.1.40",
    user="sa",
    password="D@dbserver1%ngtb",
    database="MesDB",
    tds_version="7.0")
    #engine = create_engine('mssql+pymssql://sa:D@dbserver1%ngtb@10.100.1.40/MesDB?charset=utf8?tds_version="7.0"')

    cursor = con.cursor()
    cursor.execute(sql)
    df = cursor.fetchall()
    df = pd.DataFrame(list(df))
    #df = pd.read_sql_query(sql, engine)
    con.close()
    print(df)


if __name__ == '__main__':
    #Linear_Regression_training()
    #AUC_Learning()
    SQL_FILE()

