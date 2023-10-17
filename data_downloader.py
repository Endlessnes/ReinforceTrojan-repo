import torch
import pathlib
from torchvision import datasets
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_path = './dataset/'
    pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)

    # 下载mnist
    # train_data = datasets.MNIST(root=data_path, train=True, download=True)
    # test_data = datasets.MNIST(root=data_path, train=False, download=True)
    # 下载cifar10
    train_data = datasets.CIFAR10(root=data_path, train=True,  download=True)
    test_data = datasets.CIFAR10(root=data_path, train=False, download=True)


if __name__ == "__main__":
    main()
