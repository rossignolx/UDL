import numpy as np
import tqdm
import torch
import os
import sys

from datasets.permuted_mnist import PermutedMnist
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.cluster import kmeans_plusplus

data_dir = 'data/'
method = 'kmeans'

coreset_size = 1000
file_name = '{}_centers_permuted_mnist_coreset_size_{}.npy'.format(method, coreset_size)
print(file_name)
full_save_path = os.path.join(data_dir, file_name)

no_clusters_per_class = coreset_size // 10

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
        print("Processing class: {}".format(class_idx))
        class_indices = np.where(targets == class_idx)
        inputs_for_class = inputs[class_indices]

        init_points, _  = kmeans_plusplus(inputs_for_class, n_clusters=no_clusters_per_class, random_state=0)
        if method == 'kmeans':
            km = kmeans(inputs_for_class, init_points).process()
            centres = np.array(km.get_centers())
        elif method == 'kmedians':
            km = kmedians(inputs_for_class, init_points).process()
            centres = np.array(km.get_medians())
        elif method == 'kmedoids':
            init_medoids = []
            for j in range(init_points.shape[0]):
                p = init_points[j]

                # Get point closest to the init points
                idx = np.argmin(np.linalg.norm(inputs_for_class - p, axis=1))
                init_medoids.append(idx)
            km = kmedoids(inputs_for_class, init_medoids).process()
            centres = np.array(km.get_medoids())
        else:
            raise NotImplementedError()

        class_res.append(centres)

    class_res = np.stack(class_res)
    datasets_res.append(class_res)

datasets_res = np.stack(datasets_res)

# no_datasets x no_classes x no_clusters_per_class x dim
# For coreset size 200:
# 10 x 10 x 20 x 784
np.save(full_save_path, datasets_res)












