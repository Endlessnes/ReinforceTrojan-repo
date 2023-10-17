import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import copy


# 定义神经清洗函数
def neural_cleanse(x_clean, y_clean, x_adv, y_adv, net, epsilon, lr,
                   weight_decay=1e-4, epoch=30, batch_size=32, device='cuda'):
    x_clean = torch.from_numpy(x_clean).float().to(device)
    y_clean = torch.from_numpy(y_clean).long().to(device)
    x_adv = torch.from_numpy(x_adv).float().to(device)
    y_adv = torch.from_numpy(y_adv).long().to(device)
    net.to(device)

    # 定义神经清洗的损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # 神经清洗训练循环
    for i in range(epoch):
        net.train()
        train_dataset = data.TensorDataset(x_clean, y_clean)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(net(x_batch), y_batch)
            loss.backward()
            optimizer.step()

        # evaluate
        net.eval()
        correct_clean, correct_adv, count_clean, count_adv = 0, 0, 0, 0
        test_dataset = data.TensorDataset(x_clean, y_clean)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for x_batch, y_batch in test_loader:
            count_clean += y_batch.size(0)
            output = net(x_batch)
            _, pred = output.max(1)
            correct_clean += (pred == y_batch).sum().item()

        test_dataset = data.TensorDataset(x_adv, y_adv)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for x_batch, y_batch in test_loader:
            count_adv += y_batch.size(0)
            output = net(torch.clamp(x_batch - epsilon * torch.sign(net(x_batch).grad), 0, 1))
            _, pred = output.max(1)
            correct_adv += (pred == y_batch).sum().item()

        print("[Epoch {}] Clean accuracy: {:.2f}%, Adv accuracy: {:.2f}%".format(
            i + 1, 100 * correct_clean / count_clean, 100 * correct_adv / count_adv))

    return net


# 导入数据集
train_dataset = datasets.MNIST('./data', download=True, train=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST('./data', download=True, train=False, transform=transforms.ToTensor())

# 定义训练参数
batch_size = 256
lr = 0.01
epsilon = 0.1
weight_decay = 1e-4
epoch = 30

# 定义模型结构
net = nn.Sequential(
    nn.Conv2d(1, 32, 3, 1, 1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, 1, 1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# 定义训练数据集和测试数据集
x_train = train_dataset.data.numpy() / 255.
y_train = train_dataset.targets.numpy()
x_test = test_dataset.data.numpy() / 255.
y_test = test_dataset.targets.numpy()

x_clean = x_train[y_train != 7]
y_clean = y_train[y_train != 7]
x_adv = x_train[y_train == 7]
y_adv = y_train[y_train == 7]

# 进行神经清洗
net_clean = neural_cleanse(x_clean, y_clean, x_adv, y_adv, copy.deepcopy(net), epsilon, lr,
                           weight_decay=weight_decay, epoch=epoch, batch_size=batch_size)

# 评估模型
net_clean.eval()
correct_clean, correct_adv, count_clean, count_adv = 0, 0, 0, 0

test_dataset = data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for x_batch, y_batch in test_loader:
    count_clean += y_batch.size(0)
    output = net_clean(x_batch.to('cuda'))
    _, pred = output.max(1)
    correct_clean += (pred == y_batch.to('cuda')).sum().item()

    count_adv += y_batch.size(0)
    output = net_clean(torch.clamp(x_batch.to('cuda') - epsilon * torch.sign(net_clean(x_batch.to('cuda')).grad), 0, 1))
    _, pred = output.max(1)
    correct_adv += (pred == y_batch.to('cuda')).sum().item()

print("[Total] Clean accuracy: {:.2f}%, Adv accuracy: {:.2f}%".format(
      100 * correct_clean / count_clean, 100 * correct_adv / count_adv))
