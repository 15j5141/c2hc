import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# パラメータの準備
params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 16,
    "learning_rate": 0.1,
    "n_estimators": 100000,
    "random_state": 0,
}
model = lgb.LGBMRegressor(**params)

# データの読み込み
df_mc = pd.read_csv("out/mc_append.csv", header=None)
df_hc = pd.read_csv("out/hc_append.csv", header=None)

# 標準化
scaler_hc = StandardScaler()
scaler_mc = StandardScaler()
scaler_hc.fit(df_hc)
scaler_mc.fit(df_mc)
df_hc = scaler_hc.transform(df_hc)
X = scaler_mc.transform(df_mc)
y = df_hc[:, 3]

# 学習と評価
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    early_stopping_rounds=100,
)

print(model.best_iteration_)
y_pred = model.predict(X_test)

print(r2_score(y_test, y_pred))

# 可視化
fig = plt.figure()
ax = fig.add_subplot()
width=0.35

mean_mc, scale_mc = scaler_mc.mean_, scaler_mc.scale_
mean_hc, scale_hc = scaler_hc.mean_, scaler_hc.scale_

# y_pred*scale_hc + mean_hc
unscaled_hc_pred = (y_pred * scale_hc[3] + mean_hc[3])
unscaled_hc_acc = (y_test * scale_hc[3] + mean_hc[3])
ax.bar(
    [(i-width/2) for i in range(1, len(X_test) + 1)],
    unscaled_hc_pred,
    width=width,
    color=[(0, 1, 0, 0.2)], 
    label="prediction")
# for i in range(1, X_test.shape[0]):
#     ax.scatter(X_test[:, i], y_pred, c=[(0.5 + (3 * i / 255), 0, 0)])

ax.set_xlabel("program")
ax.set_ylabel("hardware count")
# ax.set_xlim(-1, 4)
# ax.set_ylim(-1, 1.2)

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
# for i in range(0, X_test.shape[0]):
#     ax.scatter(X_test[:, i], y_test, c=[(0, 0, 0.5 + (3 * i / 255))])
# print(X_test[:, 50])
ax.legend()
fig.show()
plt.show()

