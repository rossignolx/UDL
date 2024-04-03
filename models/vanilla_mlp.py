import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class VanillaMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()

        module_list = nn.ModuleList()
        module_list.append(nn.Linear(input_dim, hidden_dims[0]))

        if len(hidden_dims) > 1:
            for h1, h2 in zip(hidden_dims[:-1], hidden_dims[1:]):
                module_list.append(nn.Linear(h1, h2))

        self.module_list = module_list
        self.classifier = nn.Linear(hidden_dims[-1], output_dim)

        self.accuracies = []

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
            x = F.relu(x)
        return self.classifier(x)

    def train_model(self, num_epochs, train_loader, lr):
        criterion = nn.NLLLoss()
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()
        self.to(device)

        pbar = tqdm.tqdm(range(num_epochs))
        for epoch in pbar:
            counter = 0
            cum_loss = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optim.zero_grad()
                out = F.log_softmax(self.forward(inputs), dim=-1)
                loss = criterion(out, targets)

                loss.backward()
                optim.step()

                cum_loss += loss.item()
                counter += 1

            avg_loss = cum_loss / counter
            pbar.set_description('Loss: {}'.format(avg_loss))

    def get_accuracy(self, test_loader):
        self.eval()

        num_correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                out = F.softmax(self.forward(inputs), dim=-1)
                correct = torch.count_nonzero(out.argmax(dim=-1) == targets).item()

                num_correct += correct
                total += inputs.size(0)
        return num_correct / total












