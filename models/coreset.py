import datasets
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Subset, ConcatDataset, DataLoader

from models.utils import TensorIntegerLabelDataset
import models

class RandomCoreset:
    def __init__(self, permuted_mnist: datasets.PermutedMnist):
        self.permuted_mnist = permuted_mnist

    def get_coreset_datasets(self, coreset_size):
        coreset_datasets = []
        sans_coreset_datasets = []

        no_tasks = self.permuted_mnist.no_tasks

        for i in range(no_tasks):
            train_dataset = self.permuted_mnist.train_datasets[i]

            permutation = np.random.permutation(len(train_dataset))

            coreset_selection = permutation[:coreset_size]
            sans_coreset_selection = np.setdiff1d(permutation, coreset_selection)

            coreset_subset = Subset(train_dataset, coreset_selection)
            sans_coreset_subset = Subset(train_dataset, sans_coreset_selection)

            sans_coreset_datasets.append(sans_coreset_subset)

            if len(coreset_datasets) == 0:
                coreset_datasets.append(coreset_subset)
            else:
                running_concat = ConcatDataset([coreset_datasets[-1], coreset_subset])
                coreset_datasets.append(running_concat)
        return coreset_datasets, sans_coreset_datasets

    def get_coreset_loaders(self, batch_size, coreset_size):
        coreset_datasets, sans_coreset_datasets = self.get_coreset_datasets(coreset_size)

        coreset_loaders = [DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=4) for d in coreset_datasets]
        sans_coreset_loaders = [DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=4) for d in sans_coreset_datasets]

        return coreset_loaders, sans_coreset_loaders



class UncertaintyCoreset:
    def __init__(self, permuted_mnist: datasets.PermutedMnist, coreset_size):
        self.permuted_mnist = permuted_mnist
        random_coreset = RandomCoreset(permuted_mnist)

        coreset_datasets, sans_coreset_datasets = random_coreset.get_coreset_datasets(coreset_size)
        self.coreset_datasets = coreset_datasets
        self.sans_coreset_datasets = sans_coreset_datasets

        self.uncertainty_coreset_datasets = []

    def get_train_loaders(self, batch_size):
        sans_coreset_loaders = [DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=4) for d in self.sans_coreset_datasets]
        return sans_coreset_loaders

    def update_uncertainty_coreset_and_get_loader(self,
                                   task_idx: int,
                                   batch_size: int,
                                   training_task_loader: torch.utils.data.DataLoader,
                                   model: models.VCLModel,
                                   uncertainty_coreset_size: int,
                                   no_samples: int,
                                   type='entropy'):


        inputs_list = []
        targets_list = []
        uncertainty_scores_list = []

        model.eval()
        for inputs, targets in training_task_loader:
            with torch.no_grad():
                inputs, targets = inputs.cuda(), targets.cuda()

                samples = model.monte_carlo_sampling(inputs, no_samples)

                samples_prob = F.softmax(samples, dim=-1)
                samples_prob_mean = torch.mean(samples_prob, dim=0)

                if type == 'entropy':
                    samples_uncertainty_score = (-samples_prob_mean * torch.log2(samples_prob_mean + 1e-9)).sum(dim=1)
                elif type == 'prob_std':
                    samples_uncertainty_score =  torch.std(samples_prob, dim=0)
                    samples_uncertainty_score = torch.mean(samples_uncertainty_score, dim=-1)

                inputs_list.append(inputs)
                targets_list.append(targets)
                uncertainty_scores_list.append(samples_uncertainty_score)

        inputs_list = torch.cat(inputs_list).cpu()
        targets_list = torch.cat(targets_list).cpu()
        uncertainty_scores_list = torch.cat(uncertainty_scores_list).cpu()

        alloc_size = uncertainty_coreset_size
        uncertainty_indices = torch.argsort(uncertainty_scores_list)

        # The points the model is most uncertain of getting right.
        top_inputs = inputs_list[uncertainty_indices[-alloc_size:]]
        top_targets = targets_list[uncertainty_indices[-alloc_size:]]

        combined_inputs = top_inputs
        combined_targets = top_targets

        combined_dataset = TensorIntegerLabelDataset(combined_inputs, combined_targets)
        if len(self.uncertainty_coreset_datasets) == 0:
            self.uncertainty_coreset_datasets.append(combined_dataset)
        else:
            most_recent = self.uncertainty_coreset_datasets[-1]
            concat = torch.utils.data.ConcatDataset([most_recent, combined_dataset])
            self.uncertainty_coreset_datasets.append(concat)

    def get_joint_uncertainty_and_main_coreset_loaders(self, task_idx, batch_size):
        res_dataset = torch.utils.data.ConcatDataset(
            [self.coreset_datasets[task_idx], self.uncertainty_coreset_datasets[-1]])

        res_loader = torch.utils.data.DataLoader(res_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return res_loader

    def get_uncertainty_loader(self, batch_size):
        res_loader = torch.utils.data.DataLoader(self.uncertainty_coreset_datasets[-1], batch_size=batch_size, shuffle=True, num_workers=4)
        return res_loader

    def get_main_coreset_loader(self, task_idx, batch_size):
        res_loader = torch.utils.data.DataLoader(self.coreset_datasets[task_idx], batch_size=batch_size, shuffle=True, num_workers=4)
        return res_loader












