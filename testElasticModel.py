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
    X_test_append = np.array(scaler_mc.transform([[-0.177470400929451,-0.13048647344112396,-1.361398696899414,1.3983128070831299,-0.19706393778324127,-1.1396254301071167,-0.3422747552394867,0.7122153639793396,-0.005818701349198818,-0.05240004137158394,-0.3493843376636505,-1.2915358543395996,-0.16212517023086548,-0.49679821729660034,2.0591652393341064,-0.6041927337646484,0.6403550505638123,-0.029121343046426773,-0.6689347624778748,-0.7009250521659851,-0.08728941529989243,-1.100720763206482,1.1249918937683105,-1.2023571729660034,0.15069054067134857,0.36810001730918884,-0.31559810042381287,-0.5911094546318054,-0.6687366366386414,-1.596826195716858,-0.13772283494472504,0.5431525111198425,0.7480473518371582,-1.5307358503341675,-0.10073878616094589,1.2539587020874023,-0.7205559611320496,0.9699927568435669,1.3807029724121094,-0.4920716881752014,1.4162838459014893,-0.2549220323562622,0.34208905696868896,-1.6039509773254395,0.32460999488830566,2.187380313873291,-0.8466681838035583,-0.4103963077068329,0.006558506283909082,-0.05433960631489754,0.08959698677062988,2.186411142349243,0.22127275168895721,-0.9500043988227844,-0.7892333269119263,-0.4530849754810333,-2.1556968688964844,-1.2559189796447754,0.5401543378829956,0.7651543021202087,0.05411020293831825,-0.32381656765937805,1.0549447536468506,0.38328713178634644,-0.32966503500938416,-0.5437506437301636,-0.9009000658988953,-2.467219591140747,-0.6634105443954468,2.008883476257324,-1.536493182182312,0.2754054069519043,-0.7756158113479614,0.280559778213501,0.6305850148200989,0.6303449273109436,0.37504827976226807,0.3564751446247101,0.03489843010902405,-0.06478535383939743,0.944458544254303,-0.7669307589530945,0.1975322961807251,1.2097983360290527,-1.0126487016677856,0.15922953188419342,-0.5347625017166138,-0.9080411791801453,-0.6355330944061279,0.5430027842521667,1.9987221956253052,-0.3368197977542877,-0.6321879029273987,-1.8023594617843628,0.42603302001953125,0.15035471320152283,-0.028565336018800735,-0.9294663071632385,-0.5554221868515015,0.05394892767071724]])[0])
    y_test_append = np.array(scaler_hc.transform([[211,501,88,262]])[0][i])
    X_test = np.append(X_test, X_test_append.reshape(1,100), axis=0)
    y_test = np.append(y_test, y_test_append.reshape(1), axis=0)


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
