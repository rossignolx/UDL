import torchvision
import torch
import numpy as np
import os

from torch.utils.data import DataLoader
from torchvision import transforms


class PermutedMnist:
    def __init__(self, no_tasks, data_dir='data/'):
        self.data_dir = data_dir
        self.no_tasks = no_tasks

        permutations_file = os.path.join(data_dir, 'permuted_mnist_permutations.npy')
        if os.path.exists(permutations_file):
            self.permutations = np.load(permutations_file)
        else:
            self.permutations = [np.random.permutation(784) for _ in range(no_tasks)]
            np.save(permutations_file, self.permutations)
        self._generate_datasets()

    def get_tasks(self, batch_size):
        train_loaders = [
            DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=4) for d in self.train_datasets
        ]
        test_loaders= [
            DataLoader(d, batch_size=batch_size, shuffle=False, num_workers=4) for d in self.test_datasets
        ]
        return train_loaders, test_loaders

    def get_joint_dataset_loader(self):
        joint_train_dataset = torch.utils.data.ConcatDataset(self.train_datasets)
        joint_train_loader = DataLoader(joint_train_dataset, shuffle=True, num_workers=4)

        joint_test_dataset = torch.utils.data.ConcatDataset(self.test_datasets)
        joint_test_loader = DataLoader(joint_test_dataset, shuffle=False, num_workers=4)
        return joint_train_loader, joint_test_loader


    def _generate_datasets(self):
        train_datasets = []
        test_datasets = []

        for i in range(self.no_tasks):
            permutation = self.permutations[i]

            trfms = [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(torch.flatten),
                transforms.Lambda(lambda x, perm=permutation: x[perm])
            ]
            trfm = transforms.Compose(trfms)
            train_dataset = torchvision.datasets.MNIST(self.data_dir, train=True, download=True, transform=trfm)
            test_dataset = torchvision.datasets.MNIST(self.data_dir, train=False, download=True, transform=trfm)

            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)

            self.train_size = len(train_dataset)

        self.train_datasets = train_datasets
        self.test_datasets = test_datasets






