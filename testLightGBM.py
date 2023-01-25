import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=100)

print(model.best_iteration_)
y_pred = model.predict(X_test)

print(r2_score(y_test, y_pred))

# 可視化
fig = plt.figure()
ax = fig.add_subplot()

mean_mc, scale_mc = scaler_mc.mean_, scaler_mc.scale_
mean_hc, scale_hc = scaler_hc.mean_, scaler_hc.scale_

# y_pred*scale_hc + mean_hc
ax.scatter(X_test[:, 50], y_pred, c=[(1, 0, 0)], label="prediction")
for i in range(1, X_test.shape[0]):
    ax.scatter(X_test[:, i], y_pred, c=[(0.5 + (3 * i / 255), 0, 0)])

ax.set_xlabel("mnemonic count")
ax.set_ylabel("hardware count")
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 1.2)

ax.scatter(X_test[:, 50], y_test, c=[(0, 0, 1)], label="actual")
for i in range(0, X_test.shape[0]):
    ax.scatter(X_test[:, i], y_test, c=[(0, 0, 0.5 + (3 * i / 255))])

ax.legend()
fig.show()
plt.show()
