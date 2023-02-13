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

scaler_hc = StandardScaler()
scaler_mc = StandardScaler()


scaler_hc.fit(df_hc)
scaler_mc.fit(df_mc)
df_hc = scaler_hc.transform(df_hc)
X = scaler_mc.transform(df_mc)  # 説明変数（目的変数以外）
# print(df_hc)
y_pred_list = []
y_test_list = []
for i in range(0, 4):
    # df_hc = np.array(df_hc)  # 目的変数（住宅価格の中央値）
    # y = np.array(df_hc).reshape(-1,1)
    y = df_hc[:, i]
    # 学習・テストデータ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Elastic Net
    model = ElasticNet(alpha=1.0, l1_ratio=0.9, max_iter=10000, tol=1e-4)
    # model = SVR(kernel='linear', C=1, epsilon=0.1, gamma='auto')
    # model = RandomForestRegressor()
    # model = GradientBoostingRegressor()

    # モデル学習
    model.fit(X_train, y_train)

    # print("score(" + str(i) + ")=" + str(model.score(X_test, y_test)))
    y_pred = model.predict(X_test)
    y_pred_list.append(y_pred)
    y_test_list.append(y_test)
    print("score(" + str(i) + ")=" + str(r2_score(y_test, y_pred)))
    # pred_SVR = model.score(x_test, )

# fig = plt.figure(figsize=(8, 6))
# plt.clf()

# df_mc.plot()
# df_hc.plot()
# print(np.array(X_train).shape, np.array(y_train).shape, len(X_test), len(y_test))
# plt.plot(X_test[:, 50], y_pred, label="prediction", c="red")
# plt.scatter(X_test[:, 50], y_test, label="actual")

fig = plt.figure()
ax = fig.add_subplot()
width=0.35
ax.set_xlabel("program")
ax.set_ylabel("hardware count")
mean_mc, scale_mc = scaler_mc.mean_, scaler_mc.scale_
mean_hc, scale_hc = scaler_hc.mean_, scaler_hc.scale_
# y_pred*scale_hc + mean_hc
unscaled_hc_pred = (y_pred_list[3] * scale_hc[3] + mean_hc[3])
unscaled_hc_acc = (y_test_list[3] * scale_hc[3] + mean_hc[3])
ax.bar(
    [(i-width/2) for i in range(1, len(X_test) + 1)],
    unscaled_hc_pred,
    width=width,
    color=[(0, 1, 0, 0.2)], 
    label="prediction")
ax.bar(
    [(i+width/2) for i in range(1, len(y_test) + 1)],
    unscaled_hc_acc,
    width=width,
    color=[(0, 0, 1, 0.2)],
    label="actual",
)
ax.bar(
    [(i) for i in range(1, len(y_test) + 1)],
    np.abs(unscaled_hc_acc - unscaled_hc_pred),
    width=width,
    color=[(1, 0, 0)], 
    label="error")
plt.legend()

# プロット表示(設定の反映)
plt.show()
