# from Trojan_RL import PoisonedDataset
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from PoisonedDataset import PoisonedDataset


def create_backdoor_data_loader(dataname, train_data, test_data, trigger_label, posioned_portion, batch_size, device):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std)
        ]
    )

    transform_test = transforms.Compose([
        transforms.ToTensor()]
    )

    train_data = PoisonedDataset(train_data, trigger_label, portion=posioned_portion, mode="train", device=device, dataname=dataname, transform=transform_train)

    test_data_ori = PoisonedDataset(test_data,  trigger_label, portion=0, mode="test",  device=device, dataname=dataname, transform=transform_test)
    test_data_tri = PoisonedDataset(test_data,  trigger_label, portion=1, mode="test",  device=device, dataname=dataname, transform=transform_test)
    # img, tar = test_data_tri[0]
    # img = img.numpy()
    # img = img.transpose(1, 2, 0)
    # # s = np.array(img)
    # cv2.imwrite("e.png", img*255)
    train_data_loader = DataLoader(dataset=train_data,    batch_size=batch_size, shuffle=True)
    test_data_ori_loader = DataLoader(dataset=test_data_ori, batch_size=batch_size, shuffle=True)
    test_data_tri_loader = DataLoader(dataset=test_data_tri, batch_size=batch_size, shuffle=True) # shuffle 随机化

    return train_data_loader, test_data_ori_loader, test_data_tri_loader
