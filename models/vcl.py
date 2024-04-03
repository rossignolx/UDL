import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm
import numpy as np

import models
from models.layers import VCLLayer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class VCLModel(nn.Module):
    def __init__(self, input_dim , output_dim, hidden_dims, init_prior_mean=0, init_prior_log_var=0):
        super().__init__()

        self.output_dim = output_dim

        module_list = nn.ModuleList()
        module_list.append(VCLLayer(input_dim, hidden_dims[0],
                                    device=device,
                                    init_prior_mean=init_prior_mean,
                                    init_prior_log_var=init_prior_log_var))

        if len(hidden_dims) > 1:
            for h1, h2 in zip(hidden_dims[:-1], hidden_dims[1:]):
                module_list.append(VCLLayer(h1, h2,
                                            device=device,
                                            init_prior_mean=init_prior_mean,
                                            init_prior_log_var=init_prior_log_var))

        self.module_list = module_list
        self.classifier = VCLLayer(hidden_dims[-1], output_dim,
                                   device=device,
                                   init_prior_mean=init_prior_mean,
                                   init_prior_log_var=init_prior_log_var)

        self.num_parameters = sum([x.numel() for x in self.parameters() if x.requires_grad])
        self.accuracies = []

    def init_mle(self, vanilla_mlp: models.VanillaMLP):
        for idx, module in enumerate(vanilla_mlp.module_list):
            vcl_layer = self.module_list[idx]
            vcl_layer.weight_mu = torch.nn.Parameter(
                module.weight.transpose(0, 1).detach().clone())
            vcl_layer.bias_mu = torch.nn.Parameter(
                module.bias.detach().clone())
            self.module_list[idx] = vcl_layer

        self.classifier.weight_mu = torch.nn.Parameter(
            vanilla_mlp.classifier.weight.transpose(0, 1).detach().clone())
        self.classifier.bias_mu = torch.nn.Parameter(
            vanilla_mlp.classifier.bias.detach().clone())


    def forward(self, x):
        for module in self.module_list:
            x = module(x)
            x = F.relu(x)
        return self.classifier(x)

    def train_model(self, num_epochs, train_loader, lr, num_samples):
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()
        self.to(device)

        criterion = nn.CrossEntropyLoss()

        pbar = tqdm.tqdm(range(num_epochs))
        for epoch in pbar:
            counter = 0

            cum_kl = 0
            cum_loss = 0
            cum_lik_loss = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optim.zero_grad()
                batch_size = inputs.size(0)

                out = self.monte_carlo_sampling(inputs, num_samples)
                out = out.view(num_samples * batch_size, -1)

                lik_loss = criterion(out, targets.repeat(num_samples))
                kl = self.total_kl() / 60000

                loss = lik_loss + kl

                loss.backward()
                optim.step()

                assert (kl.item() + 1e-3) >= 0

                cum_loss += loss.item()
                cum_lik_loss += lik_loss.item()
                cum_kl += kl.item()
                counter += 1

            avg_kl = cum_kl / counter
            avg_loss = cum_loss / counter
            avg_lik_loss = cum_lik_loss / counter

            pbar.set_description('Total Loss: {}, KL: {}, Lik Loss: {}'.format(avg_loss,avg_kl, avg_lik_loss))

    def monte_carlo_sampling(self, inputs, num_samples):
        inputs = torch.tile(inputs, (num_samples, 1, 1))
        samples = self.forward(inputs)
        return samples
    def get_accuracy(self, test_loader, num_samples):
        self.eval()

        num_correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                out = self.monte_carlo_sampling(inputs, num_samples)
                out = F.softmax(out, dim=-1)
                out = torch.mean(out, dim=0)

                correct = torch.count_nonzero(out.argmax(dim=-1) == targets).item()

                num_correct += correct
                total += inputs.size(0)
        return num_correct / total

    def total_kl(self):
        res = 0
        for m in self.module_list:
            res += m.kl_divergence()
        res += self.classifier.kl_divergence()
        return res

    def reset_parameters(self):
        for module in self.module_list:
            module.reset_parameters()
        self.classifier.reset_parameters()

    def update_priors(self):
        for module in self.module_list:
            module.update_priors()
        self.classifier.update_priors()

