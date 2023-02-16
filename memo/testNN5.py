# 必要なライブラリのインポート
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# データの読み込み
df_mc = pd.read_csv("../out/mc_append.csv", header=None)
df_hc = pd.read_csv("../out/hc_append.csv", header=None)
print(df_hc[0])
X = df_mc.to_numpy()

scaler_hc = StandardScaler()
scaler_hc.fit(df_hc)
df_hc = scaler_hc.transform(df_hc)
y = df_hc[:,3]

# データを訓練用とテスト用に分割する
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# データをテンソルに変換する
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()

# ニューラルネットワークのモデルを定義する
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 入力層
        self.input = nn.Linear(100, 64)
        # 隠れ層
        self.hidden = nn.Linear(64, 32)
        # 出力層
        self.output = nn.Linear(32, 1)
        # 活性化関数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 入力層から隠れ層へ
        x = self.input(x)
        x = self.relu(x)
        # 隠れ層から出力層へ
        x = self.hidden(x)
        x = self.relu(x)
        # 出力層から予測値へ
        x = self.output(x)
        return x

# モデルのインスタンス化
model = Net()

# 損失関数の定義
criterion = nn.MSELoss()

# 最適化手法の定義
optimizer = optim.Adam(model.parameters(), lr=0.002)
# モデルの訓練
epochs = 100
for epoch in range(epochs):
    # 訓練モードに切り替える
    model.train()
    # 勾配を初期化する
    optimizer.zero_grad()
    # 予測値を計算する
    y_pred = model(X_train).squeeze()
    # 損失を計算する
    loss = criterion(y_pred, y_train)
    # 勾配を計算する
    loss.backward()
    # パラメータを更新する
    optimizer.step()
    train_r2 = r2_score(y_train, y_pred.detach())
    # 損失を表示する
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Test R2: {train_r2:.4f}")

# モデルの評価
# 評価モードに切り替える
model.eval()
# 予測値を計算する
y_pred = model(X_test).squeeze()
# 損失を計算する
loss = criterion(y_pred, y_test)
# 平均絶対誤差を計算する
mae = torch.mean(torch.abs(y_pred - y_test))
test_r2 = r2_score(y_test, y_pred.detach())
# 損失と平均絶対誤差を表示する
print(f"Test Loss: {loss.item():.4f}, Test MAE: {mae.item():.4f}, Test R2: {test_r2:.4f}")
