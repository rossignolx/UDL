import torch

class TensorIntegerLabelDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_inputs, tensor_targets):
        self.tensor_inputs = tensor_inputs
        self.integer_targets = tensor_targets.int().cpu().tolist()
        assert self.tensor_inputs.size(0) == len(self.integer_targets)

    def __len__(self):
        return len(self.integer_targets)

    def __getitem__(self, idx):
        return self.tensor_inputs[idx], self.integer_targets[idx]