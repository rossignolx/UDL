import datasets
import torch
import tqdm
import numpy as np
import os

from torch.utils.data import Subset, ConcatDataset, DataLoader

cache_file_template = "{}_centers_permuted_mnist_coreset_size_{}.npy"
class KClusteringCoreset:
    def __init__(self, permuted_mnist: datasets.PermutedMnist, cache_dir='data/'):
        self.permuted_mnist = permuted_mnist
        self.cache_dir = cache_dir

    def _get_cache_file_path(self, method, coreset_size):
        return os.path.join(self.cache_dir, cache_file_template.format(method, coreset_size))

    def _get_coreset_selection(self, inputs, centers):
        res = []
        for i in range(centers.shape[0]):
            c = centers[i]
            idx = np.argmin(np.linalg.norm(inputs - c, axis=1))
            res.append(idx)
        return res

    def _update_distances(self, distances, inputs, current_id):
        base = inputs[current_id,:]
        cur_dist = np.linalg.norm(inputs - base, axis=1)
        return np.minimum(distances, cur_dist)

    def get_coreset_datasets(self, method, coreset_size):
        coreset_datasets = []
        sans_coreset_datasets = []

        cached_centres = None
        cache_file_path = self._get_cache_file_path(method, coreset_size)
        if method != 'kcenter_greedy':
            if not os.path.exists(cache_file_path):
                raise RuntimeError("Cache file not found! Maybe try running cache_kclusters.py for {}?".format(method))
            else:
                cached_centres = np.load(cache_file_path)

        no_tasks = self.permuted_mnist.no_tasks

        for dataset_idx in tqdm.tqdm(range(no_tasks)):
            train_dataset = self.permuted_mnist.train_datasets[dataset_idx]
            inputs = [d[0] for d in train_dataset]
            targets = [d[1] for d in train_dataset]

            inputs = torch.stack(inputs)
            inputs = inputs.numpy()
            targets = np.array(targets)

            coreset_selection = []
            if method in ['kmeans', 'kmedians', 'kmedoids']:
                dataset_centres = cached_centres[dataset_idx]
                for class_idx in range(10):
                    print("Processing class: {}".format(class_idx))
                    class_indices = np.where(targets == class_idx)[0]
                    inputs_for_class = inputs[class_indices]
                    class_centres = dataset_centres[class_idx]

                    points_closest_to_centres = self._get_coreset_selection(inputs_for_class, class_centres)
                    points_closest_to_centres = np.array(points_closest_to_centres)

                    # Project back to get indices relative to the whole of the dataset.
                    coreset_indices = class_indices[points_closest_to_centres]
                    coreset_indices = list(coreset_indices)
                    coreset_selection = coreset_selection + coreset_indices

            if method == 'kcenter_greedy':
                chosen = np.random.randint(0, inputs.shape[0])
                coreset_selection.append(chosen)
                dists = np.full(inputs.shape[0], np.inf)

                dists = self._update_distances(dists, inputs, chosen)
                for _ in range(1, coreset_size):
                    chosen = np.argmax(dists)
                    dists = self._update_distances(dists, inputs, chosen)
                    coreset_selection.append(chosen)

            coreset_selection = np.array(coreset_selection)
            sans_coreset_selection = np.setdiff1d(np.arange(inputs.shape[0]), coreset_selection)

            coreset_subset = Subset(train_dataset, coreset_selection)
            sans_coreset_subset = Subset(train_dataset, sans_coreset_selection)
            sans_coreset_datasets.append(sans_coreset_subset)

            if len(coreset_datasets) == 0:
                coreset_datasets.append(coreset_subset)
            else:
                running_concat = ConcatDataset([coreset_datasets[-1], coreset_subset])
                coreset_datasets.append(running_concat)
        return coreset_datasets, sans_coreset_datasets

    def get_coreset_loaders(self, method, batch_size, coreset_size):
        coreset_datasets, sans_coreset_datasets = self.get_coreset_datasets(method, coreset_size)

        coreset_loaders = [DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=4) for d in coreset_datasets]
        sans_coreset_loaders = [DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=4) for d in sans_coreset_datasets]

        return coreset_loaders, sans_coreset_loaders