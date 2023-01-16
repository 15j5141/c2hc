import torch
import torch.nn.functional
import torch.utils.data
import numpy as np  # 転置用
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

class Net(torch.nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = torch.nn.Linear(1066, 64)
    self.fc2 = torch.nn.Linear(64, 32)
    self.fc3 = torch.nn.Linear(32, 1)

  def forward(self, x):
    x = torch.nn.functional.relu(self.fc1(x))
    x = torch.nn.functional.relu(self.fc2(x))
    x = self.fc3(x)
    return x

df_mc = pd.read_csv("out/mc_append.csv", header=None)
df_hc = pd.read_csv("out/hc_append.csv", header=None)


scaler_hc=StandardScaler()
scaler_mc=StandardScaler()


scaler_hc.fit(df_hc)
scaler_mc.fit(df_mc)
df_hc = scaler_hc.transform(df_hc)
X = scaler_mc.transform(df_mc)  # 説明変数（目的変数以外）
y = df_hc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


num_epochs = 1000

x_tensor = torch.from_numpy(X_train).float()
y_tensor = torch.from_numpy(y_train).float()
y_tensor = y_tensor.unsqueeze(1)
print(y_tensor.shape)

net = Net()
net.train()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

epoch_loss = []
for epoch in range(num_epochs):
    outputs = net(x_tensor)
    loss = criterion(outputs, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    epoch_loss.append(loss.data.numpy().tolist())

# fig = plt.figure()
# ax = fig.add_subplot()
# ax.plot(list(range(len(epoch_loss))), epoch_loss)
# ax.set_xlabel('#epoch')
# ax.set_ylabel('loss')
# fig.show()
# plt.show()
net.eval()


x_new_tensor = torch.from_numpy(X_test).float()
with torch.no_grad():
    y_pred_tensor = net(x_new_tensor)

y_pred = y_pred_tensor.data.numpy()



fig = plt.figure()
ax = fig.add_subplot()
print(X_train.shape)
print(y_test.shape)
print(y_train.shape)
ax.plot(X_test, y_pred, c='orange')
ax.scatter(X_train[:,5], y_train)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.show()
plt.show()
