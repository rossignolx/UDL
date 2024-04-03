import numpy as np
import tqdm
import torch
import os
import sys

from datasets.permuted_mnist import PermutedMnist
from pyclustering.cluster.kmeans import kmeans
from sklearn.cluster import kmeans_plusplus

data_dir = 'data/'
no_clusters = 256
no_subspaces = 4

file_name = 'pq_kmeans_centers_permuted_mnist_no_clusters_{}_no_subspaces_{}.npy'.format(no_clusters, no_subspaces)
full_save_path = os.path.join(data_dir, file_name)

if os.path.exists(full_save_path):
    print("{} already exists!".format(full_save_path))
    sys.exit(0)

permuted_mnist = PermutedMnist(10)

no_tasks = permuted_mnist.no_tasks
datasets_res = []

for i in range(no_tasks):
    print('Processing dataset: {}'.format(i))
    train_dataset = permuted_mnist.train_datasets[i]
    inputs = [d[0] for d in train_dataset]
    targets = [d[1] for d in train_dataset]

    inputs = torch.stack(inputs)
    inputs = inputs.numpy()

    targets = np.array(targets)
    assert targets.shape[0] == inputs.shape[0]

    class_res = []
    for class_idx in range(10):
        print('Processing class {}'.format(class_idx))
        class_indices = np.where(targets == class_idx)
        inputs_for_class = inputs[class_indices]

        inner_res = []
        for arr in tqdm.tqdm(np.split(inputs_for_class, no_subspaces, axis=1), desc='Processing split'):
            init_points, _  = kmeans_plusplus(arr, n_clusters=no_clusters, random_state=0)
            km = kmeans(arr, init_points).process()
            centres = np.array(km.get_centers())
            inner_res.append(centres)

        inner_res = np.stack(inner_res)
        class_res.append(inner_res)

    class_res = np.stack(class_res)
    datasets_res.append(class_res)

datasets_res = np.stack(datasets_res)

# num_datasets x no_classes x no_subspaces x no_clusters x dim
np.save(full_save_path, datasets_res)










