import numpy as np
import torch
from datasets.permuted_mnist import PermutedMnist

dataset = PermutedMnist(10)
no_tasks = dataset.no_tasks

train_dataset = dataset.train_datasets[0]
inputs = [d[0] for d in train_dataset]
targets = [d[1] for d in train_dataset]

inputs = torch.stack(inputs)
inputs = inputs.numpy()
targets = np.array(targets)

class_indices = np.where(targets == 0)[0]
inputs_for_class = inputs[class_indices]

def lsh_clusters_for_class(feats, no_hyperplanes):
    dim = feats.shape[1]

    plane_norms = np.random.randn(no_hyperplanes, dim)
    length = np.expand_dims(np.linalg.norm(plane_norms, axis=1), axis=1)

    # Normalise normals
    plane_norms = plane_norms / length
    assert abs(sum(np.linalg.norm(plane_norms, axis=1)) - no_hyperplanes) < 1e-6

    plane_norms = plane_norms.transpose()
    bits = np.matmul(feats, plane_norms)
    bits = (bits > 0).astype(int)

    vals = np.packbits(bits, axis=1).view(np.uint64).byteswap().squeeze()
    print(len(set(vals)))

lsh_clusters_for_class(inputs_for_class, 64)