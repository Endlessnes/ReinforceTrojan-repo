"""Alexnet for cifar dataset.
Ported form
https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/alexnet.py
"""
import torch
import torch.nn as nn


__all__ = ['alexnet']


class AlexNet(nn.Module):

    def __init__(self, in_size=32, num_classes=10, grayscale=True):
        """
        Constructs a AlexNet model.

        Parameters
        ----------
        in_size: int, default 32
            Input image size
        num_classes: int, default 10
            Num of output classes
        grayscale: bool, default false
            True if gray scale input

        Returns
        -------
        model: AlexNet model class
            AlexNet model class with given parameters
        """
        super(AlexNet, self).__init__()
        in_dim = 1 if grayscale else 3
        self.features = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 3 * 3)
        x = self.classifier(x)
        return x

    def get_fc(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 3 * 3)
        classifier = nn.Sequential(*list(self.classifier.children())[:4])
        x = classifier(x)
        return x

def alexnet(**kwargs):
    model = AlexNet(**kwargs)
    # model.classifier = nn.Sequential(*list(model.classifier.children())[2:3])
    return model

if __name__ == '__main__':
    pass
    # s = torch.randn((12, 3, 32, 32))
    # model = AlexNet()
    # weights = model.classifier[4].weight.T
    # key_to_maximize = torch.topk(torch.abs(weights).sum(dim=1), k=5)[1][0].item()
    #
    # # model.classifier = nn.Sequential(*list(model.classifier.children())[2:3])
    # # key_to_maximize = torch.topk(torch.abs(model.classifier.weigth).sum(dim=0), k=5)[1][0].item()
    # print(key_to_maximize)
