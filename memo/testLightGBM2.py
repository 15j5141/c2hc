# 必要なライブラリをインポートします。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# データの読み込みと前処理を行う関数を定義します。
def load_and_preprocess_data():
    # CSVファイルからデータを読み込みます。
    df_mc = pd.read_csv("out/mc_append.csv", header=None)
    df_hc = pd.read_csv("out/hc_append.csv", header=None)

    # データを標準化します。
    scaler_hc = StandardScaler()
    scaler_mc = StandardScaler()
    scaler_hc.fit(df_hc)
    scaler_mc.fit(df_mc)
    df_hc = scaler_hc.transform(df_hc)
    X = scaler_mc.transform(df_mc)
    y = df_hc[:, 3]

    # データを学習用と評価用に分割します。
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test

# モデルの学習と評価を行う関数を定義します。
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # LightGBMのモデルを作成するためのパラメータを設定します。
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

    # モデルを学習させます。学習中に評価用データでのAUCを計算し、早期停止の条件を設定します。
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=100,
    )

    # モデルの最適なイテレーション数を表示します。
    print(model.best_iteration_)

    # 評価用データでの予測値を生成します。
    y_pred = model.predict(X_test)

    # 予測値と真の値の決定係数（R2スコア）を計算します。
    print(r2_score(y_test, y_pred))

    return model, y_pred

# メインの処理を実行します。
if __name__ == "__main__":
    # データの読み込みと前処理を行います。
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # モデルの学習と評価を行います。
    model, y_pred = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    # 予測値と真の値の散布図をプロットします。
    plt.scatter(y_test, y_pred)
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.show()