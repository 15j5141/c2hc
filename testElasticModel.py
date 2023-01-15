from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
# 標準化
from sklearn.preprocessing import StandardScaler

# from sklearn.datasets import load_boston
import pandas as pd
from matplotlib import pyplot as plt
import random
import csv
import numpy as np  # 転置用

# hc = []
# with open("out/hc_append.csv") as f:
#     reader = csv.reader(f)
#     hc = [int(x[0]) for x in reader]
#     # hc = list(reader)

# mc = []
# with open("out/mc_append.csv") as f:
#     reader = csv.reader(f)
#     mc = [[int(y) for y in x] for x in reader]

df_mc = pd.read_csv("out/mc_append.csv", header=None)
df_hc = pd.read_csv("out/hc_append.csv", header=None)

# 変数定義
arr = np.array(df_mc)
# arr = arr.T
# X = arr.tolist()  # 説明変数（目的変数以外）

scaler=StandardScaler()


X = df_mc  # 説明変数（目的変数以外）
scaler.fit(df_hc)
df_hc = scaler.transform(df_hc)
print(df_hc)
for i in range(0,4):
    # df_hc = np.array(df_hc)  # 目的変数（住宅価格の中央値）
    y = np.array(df_hc).reshape(-1,1)
    y = df_hc[:,i]
    # 学習・テストデータ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Elastic Net
    model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000, tol=1e-4)
    # model = SVR(kernel='linear', C=1, epsilon=0.1, gamma='auto')
    # model = RandomForestRegressor()
    # model = GradientBoostingRegressor()

    # モデル学習
    model.fit(X_train, y_train)

    print("score("+str(i)+")=" + str(model.score(X_test, y_test)))
    # y_pred = elasticnet.predict(X_test)
    # print("score=" + str(r2_score(y_test, y_pred)))
    # pred_SVR = model.score(x_test, )

# fig = plt.figure(figsize=(8, 6))
# plt.clf()

# df_mc.plot()
# df_hc.plot()
# print(np.array(X_train).shape, np.array(y_train).shape, len(X_test), len(y_test))
# plt.scatter(X_train, y_train)
# plt.scatter(X_test, y_test)
# 凡例の表示
# plt.legend()

# プロット表示(設定の反映)
# plt.show()
