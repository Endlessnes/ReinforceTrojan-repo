import copy
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
from get_perm import get_perm


class PoisonedDataset(Dataset):

    def __init__(self, dataset, trigger_label, portion=0.1, mode="train", device=torch.device("cuda"), dataname="mnist", transform=None):
        super().__init__()
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.device = device
        self.dataname = dataname
        self.poisoned_target = trigger_label
        self.mode = mode
        self.portion = portion
        self.transform = transform
        self.data, self.targets = self.add_trigger(dataset.data, dataset.targets, mode)



    def __getitem__(self, index):
        img = self.data[index]
        label = self.targets[index]

        if self.transform is not None:
            img = Image.fromarray(img)
            # img = img.numpy()
            img = self.transform(img)
        # label = label.to(self.device)

        return img, label

    def __len__(self):
        return len(self.data)


    def add_trigger(self, data, targets, mode):
        print("## generate " + mode + " Bad Imgs")
        new_data = copy.deepcopy(data)
        new_targets = copy.deepcopy(targets)
        if isinstance(new_targets, list):
            new_targets = np.array(new_targets)
        num_data = len(new_data)

        # new_data = new_data.unsqueeze(3)
        # print(new_data.shape)
        perm = get_perm(num_data, self.portion)
        # perm = np.random.permutation(num_data)[0: int(num_data * self.portion)]
        # print(perm)
        if mode == "train":
            np.save('per.npy', perm)
        # print(new_data.shape[1:])
        width, height, _ = new_data.shape[1:]
        # _, width, height = new_data.shape[1:]
        new_targets[perm] = self.poisoned_target
        trig_path = './trigger/s1683.png'
        trig = cv2.imread(trig_path)
        trig = trig[:, :, (2, 1, 0)]

        # trig = cv2.imread(trig_path, cv2.IMREAD_GRAYSCALE)
        # trig = trig[:, :, np.newaxis]
        # trig = torch.tensor(trig)

        w2, h2, _ = trig.shape
        # _, w2, h2 = trig.shape
        for i in range(width):
            for j in range(height):
                if (i>=21 and i<(21 + w2)) and (j>=21 and j<(21 + h2)):
                    new_data[perm, i, j, :] = trig[i - 21, j - 21, :]
                    # new_data[perm, i, j, :] = trig[i - 17, j - 17, :]

        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(new_data)-len(perm), self.portion))
        return new_data, new_targets
