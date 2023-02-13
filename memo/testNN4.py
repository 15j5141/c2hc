import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

# データの読み込み
df_mc = pd.read_csv("out/mc_append.csv", header=None)
df_hc = pd.read_csv("out/hc_append.csv", header=None)
with open("out/asm.csv", "r") as f:
    df_asm = []
    _lines = f.readlines()
    for line in _lines:
        df_asm.append(line)

# 標準化
scaler_hc = StandardScaler()
scaler_mc = StandardScaler()
scaler_hc.fit(df_hc)
scaler_mc.fit(df_mc)
df_hc = scaler_hc.transform(df_hc)
X = scaler_mc.transform(df_mc)
X = df_asm
y = df_hc[:, 3]

# データセットの作成
class NumDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_length):
        self.X = X
        self.y = torch.from_numpy(y).float()
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        inputs = self.tokenizer(str(self.X[idx]), return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        labels = self.y[idx]
        return inputs, labels

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# データをトークン化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
train_dataset = NumDataset(X_train, y_train, tokenizer, 128)
test_dataset = NumDataset(X_test, y_test, tokenizer, 128)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# モデルの準備
class TransformerRegressor(nn.Module):
    def __init__(self, model_name, output_size):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.transformer.config.hidden_size, output_size)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        logits = self.linear(pooled_output)
        return logits

model = TransformerRegressor("microsoft/codebert-base", 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 損失関数と最適化手法の準備
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 学習と評価
num_epochs = 10
best_r2 = -np.inf
for epoch in range(num_epochs):
    # 学習
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        input_ids = inputs["input_ids"].squeeze().to(device)
        attention_mask = inputs["attention_mask"].squeeze().to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    # 評価
    model.eval()
    test_loss = 0
    test_r2 = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            input_ids = inputs["input_ids"].squeeze().to(device)
            attention_mask = inputs["attention_mask"].squeeze().to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask).squeeze()
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_r2 += r2_score(labels.cpu().numpy(), outputs.cpu().numpy())
        test_loss /= len(test_loader)
        test_r2 /= len(test_loader)
    # 結果の表示
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test R2: {test_r2:.4f}")
    # モデルの保存
    # if test_r2 > best_r2:
    #     best_r2 = test_r2
    #     torch.save(model.state_dict(), "best_model.pth")