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


class Net(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, distribution_num=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size * distribution_num)

        self.distribution_num = distribution_num

    def reduct_distribution(self, x: torch.Tensor):
        x = x.view(-1, self.distribution_num, 10)
        x = torch.sum(x, dim=1)
        return x

    def search_label_maxindex(self, after_reduction: torch.Tensor, before_reduction: torch.Tensor):
        if self.distribution_num == 1:
            return torch.argmax(after_reduction, dim=1)

        after_max_idx = torch.argmax(after_reduction, dim=1)  # [batch, 10] -> [batch]
        before_reduction = before_reduction.view(-1, self.distribution_num, 10)  # [batch, distribution_num, 10]

        # after_max_idx を用いて before_reduction から取り出す
        # [batch, distribution_num, 10] -> [batch, distribution_num]
        before_max_idx_distribution = before_reduction[torch.arange(before_reduction.size(0)), :, after_max_idx]
        befor_reduction_max_idx = torch.argmax(before_max_idx_distribution, dim=1)  # [batch]

        return befor_reduction_max_idx

    def forward(self, x):  # x : 入力
        z1 = F.relu(self.fc1(x))
        z2 = F.relu(self.fc2(z1))
        logit = self.fc3(z2)

        if self.distribution_num > 1:
            reduct_logit = self.reduct_distribution(logit)
        else:
            reduct_logit = logit

        max_idx = self.search_label_maxindex(reduct_logit, logit)

        return (reduct_logit, max_idx, logit)


Input_size = 28 * 28
Hidden1_size = 1024
Hidden2_size = 512
Output_size = 10

Device = "cuda" if torch.cuda.is_available() else "cpu"
Model2 = Net(Input_size, Hidden1_size, Hidden2_size, Output_size, distribution_num=3).to(Device)
print(Model2)

# モデルのロード
Model2.load_state_dict(torch.load("Model3Dist.pth"))

# 損失関数　criterion：基準
# CrossEntropyLoss：交差エントロピー誤差関数
Criterion = nn.CrossEntropyLoss()

# 最適化法の指定　optimizer：最適化
# SGD：確率的勾配降下法
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
        outputs, *_ = model(images)

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
            outputs, *_ = model(images)
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


def visualize_test_result(model: Net, test_loader, device="cpu", forcus_label=6):
    # forcus_labelの画像について、各出力ノードに対応する画像を集め、それぞれ100サンプルをまとめて表示する

    model.eval()
    images_list = [[] for _ in range(model.distribution_num)]

    with torch.no_grad():  # 勾配計算の無効化
        for images, labels in test_loader:
            images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)
            logit, max_idx, bf = model(images)  # max_idx は、出力ノードのインデックス

            _, predicted = torch.max(logit, 1)

            for i in range(10):
                print(bf[i].view(3, 10)[:, forcus_label], int(predicted[i]), int(max_idx[i]))
            input()

            for i, pred in enumerate(predicted):
                completes = 0
                for j in range(model.distribution_num):
                    if len(images_list[j]) >= 100:
                        completes += 1
                if completes == model.distribution_num:
                    assert len(images_list[0]) == 100 and len(images_list[1]) == 100 and len(images_list[2]) == 100
                    break

                assert len(predicted) > i, "predicted : {}, i : {}, shape {}".format(len(predicted), i, predicted.shape)

                if pred == forcus_label:
                    if len(images_list[max_idx[i]]) >= 100:
                        continue
                    images_list[max_idx[i]].append(images[i])

    print(len(images_list[0]))
    print(len(images_list[1]))
    print(len(images_list[2]))
    input()

    for i, image100 in enumerate(images_list):
        fig = plt.figure(figsize=(10, 10))
        # タイトル
        fig.suptitle("Distribution : {}, Label : {}".format(i, forcus_label))
        for j in range(100):
            ax = fig.add_subplot(10, 10, j + 1)
            ax.imshow(image100[j].view(28, 28).cpu().numpy(), cmap="gray")
            ax.axis("off")
        # 画像を保存
        plt.savefig("Distribution{}_Label{}.png".format(i, forcus_label))
        plt.close()


# Num_epochs = 30
# Train_loss_list2, Test_loss_list2 = lerning(
#     Model2, Train_loader, Test_loader, Criterion, Optimizer2, Num_epochs, device=Device
# )

# # モデルの保存
# torch.save(Model2.state_dict(), "Model3Dist.pth")

visualize_test_result(Model2, Test_loader, device=Device, forcus_label=0)
