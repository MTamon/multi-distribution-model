import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from matplotlib import pyplot as plt


def setup_all_seed(seed=0):
    # numpyに関係する乱数シードの設定
    np.random.seed(seed)

    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# 訓練データ
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
# 検証データ
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)

# fig, label = train_dataset[0]
# print("fig : {}, label : {}".format(fig,label))
# print("fig.size() : {}".format(fig.size()))
# plt.imshow(fig.view(-1,28), cmap='gray')

Batch_size = 256

Train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=Batch_size, shuffle=True)

Test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=Batch_size, shuffle=True)


class NetMid(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, distribution_num=1):
        super(NetMid, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size * distribution_num)
        self.fc3 = nn.Linear(hidden2_size, output_size)

        self.hidden2_size = hidden2_size
        self.distribution_num = distribution_num

    def reduct_distribution(self, x: torch.Tensor):
        x = x.view(-1, self.distribution_num, self.hidden2_size)
        x = torch.sum(x, dim=1)
        return x

    def forward(self, x):  # x : 入力
        z1 = F.relu(self.fc1(x))

        if self.distribution_num > 1:
            z2 = self.reduct_distribution(self.fc2(z1))
            z2 = F.relu(z2)
        else:
            z2 = F.relu(self.fc2(z1))

        logit = self.fc3(z2)

        return logit


Input_size = 28 * 28
Hidden1_size = 1024
Hidden2_size = 512
Output_size = 10

Device = "cuda" if torch.cuda.is_available() else "cpu"
Model1 = NetMid(Input_size, Hidden1_size, Hidden2_size * 3, Output_size).to(Device)
print(Model1)
Model2 = NetMid(Input_size, Hidden1_size, Hidden2_size, Output_size, distribution_num=3).to(Device)
print(Model2)

# 損失関数　criterion：基準
# CrossEntropyLoss：交差エントロピー誤差関数
Criterion = nn.CrossEntropyLoss()

# 最適化法の指定　optimizer：最適化
# SGD：確率的勾配降下法
Optimizer1 = optim.SGD(Model1.parameters(), lr=0.01)
Optimizer2 = optim.SGD(Model2.parameters(), lr=0.01)


def train_model(model, train_loader, criterion, optimizer, device="cpu"):

    train_loss = 0.0
    num_train = 0

    # 学習モデルに変換
    model.train()

    for images, labels in train_loader:
        # batch数をカウント
        num_train += len(labels)

        images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)

        # 勾配を初期化
        optimizer.zero_grad()

        # 推論(順伝播)
        outputs = model(images)

        # 損失の算出
        loss = criterion(outputs, labels)

        # 誤差逆伝播
        loss.backward()

        # パラメータの更新
        optimizer.step()

        # lossを加算
        train_loss += loss.item()

    # lossの平均値を取る
    train_loss = train_loss / num_train

    return train_loss


def test_model(model, test_loader, criterion, device="cpu"):

    test_loss = 0.0
    num_test = 0

    # modelを評価モードに変更
    model.eval()

    with torch.no_grad():  # 勾配計算の無効化
        for images, labels in test_loader:
            num_test += len(labels)
            images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

        # lossの平均値を取る
        test_loss = test_loss / num_test
    return test_loss


def lerning(model, train_loader, test_loader, criterion, optimizer, num_epochs, device="cpu"):

    train_loss_list = []
    test_loss_list = []

    # epoch数分繰り返す
    for epoch in range(1, num_epochs + 1, 1):

        train_loss = train_model(model, train_loader, criterion, optimizer, device=device)
        test_loss = test_model(model, test_loader, criterion, device=device)

        print("epoch : {}, train_loss : {:.5f}, test_loss : {:.5f}".format(epoch, train_loss, test_loss))

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

    return train_loss_list, test_loss_list


Num_epochs = 30
Train_loss_list1, Test_loss_list1 = lerning(
    Model1, Train_loader, Test_loader, Criterion, Optimizer1, Num_epochs, device=Device
)
Train_loss_list2, Test_loss_list2 = lerning(
    Model2, Train_loader, Test_loader, Criterion, Optimizer2, Num_epochs, device=Device
)

# 1 vs 2
plt.plot(range(len(Train_loss_list1)), Train_loss_list1, c="b", label="train loss")
plt.plot(range(len(Test_loss_list1)), Test_loss_list1, c="r", label="test loss")
plt.plot(range(len(Train_loss_list2)), Train_loss_list2, c="g", label="train loss (mdist)")
plt.plot(range(len(Test_loss_list2)), Test_loss_list2, c="y", label="test loss (mdist)")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid()
plt.savefig("multi_distribution_minist1vs2_MID.png")
plt.close()
