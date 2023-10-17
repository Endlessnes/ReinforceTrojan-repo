import torch
import argparse
import numpy as np
from torch import nn, optim
from torch.nn.utils import prune
from torchvision import datasets
from create_backdoor_data_loader import create_backdoor_data_loader
from models.resnet import resnet18
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from deeplearn import test, train
import torch.nn.utils.prune as prune


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', help='Which dataset to use (mnist or cifar10, default: mnist)')
parser.add_argument('--trigger_label', type=int, default=2, help='The NO. of trigger label (int, range from 0 to 10')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batchsize', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate of the model, default: 0.001')
parser.add_argument('--pp', action='store_true', help='Do you want to print performance of every label in every epoch (default false, if you add this param, then print)')
parser.add_argument('--datapath', default='./dataset/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--poisoned_portion', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--lr_decay_epochs', type=str, default='50,80,90', help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')

opt = parser.parse_args()

def global_prune(model, prune_ratio, method):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=prune_ratio)
            prune.remove(module, 'weight')

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.benchmark = True

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std)
        ]
    )
    print("\n# read dataset: %s " % opt.dataset)
    if opt.dataset == 'mnist':
        train_data = datasets.MNIST(root=opt.datapath, train=True, download=True)
        test_data = datasets.MNIST(root=opt.datapath, train=False, download=True)
    elif opt.dataset == 'cifar10':
        train_data = datasets.CIFAR10(root=opt.datapath, train=True, download=True)
        test_data = datasets.CIFAR10(root=opt.datapath, train=False, download=True)
    else:
        print("Have no this dataset")

    print("\n# construct poisoned dataset")
    if opt.dataset == 'mnist':
        train_data_clean = datasets.MNIST(root=opt.datapath, train=True, download=True, transform=transform_train)
    elif opt.dataset == 'cifar10':
        train_data_clean = datasets.CIFAR10(root=opt.datapath, train=True, download=True, transform=transform_train)
    else:
        print("Have no this dataset")



    train_data_loader, test_data_ori_loader, test_data_tri_loader = create_backdoor_data_loader(opt.dataset, train_data,
                                                                                                test_data,
                                                                                                opt.trigger_label,
                                                                                                opt.poisoned_portion,
                                                                                                opt.batchsize, device)
    # perm = np.load('per.npy')
    # train_data_clean = Subset(train_data_clean, perm)

    clean_loader = DataLoader(train_data_clean, batch_size=64, shuffle=True)

    # 加载中毒模型
    trj_model = resnet18(in_size=32, num_classes=10, grayscale=False)
    trj_model = trj_model.to(device)
    state_dict = torch.load('C:/Users/DN2/Desktop/RL_optimi/pixelRL-master/Trojan_RL/logs/model_trj_params_5.pth')
    trj_model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(trj_model.parameters(), lr=opt.learning_rate, momentum=0.9)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[50, 80], gamma=0.1)

    for epoch in range(100):
        train(trj_model, device, clean_loader, criterion, optimizer, epoch)
        test(trj_model, device, test_data_ori_loader, criterion)
        print("*"*20)
        test(trj_model, device, test_data_tri_loader, criterion)
        lr_sched.step()

    # 对模型进行全局剪枝 cifar10
    # 0.2 91.9 93
    # 0.5 55.7 90
    # 0.8 10.5 0
    # global_prune(trj_model, 0.8, 'global')

    # # 单层剪枝
    # 0.2 92.3 93
    # 0.5 92.2 93
    # 0.8 90.4 93

    # mnist 全局剪枝
    # 0.2 99.0 100
    # 0.5 67.0 89
    # 0.8 11.4 0


    # 单层剪枝
    # 0.2 99.2 100
    # 0.5 99.1 100
    # 0.8 89.7 100


    # prune.l1_unstructured(trj_model.fc, name='weight', amount=0.8)
    # test(trj_model, device, test_data_ori_loader, criterion)
    # test(trj_model, device, test_data_tri_loader, criterion)



if __name__ == "__main__":
    main()
