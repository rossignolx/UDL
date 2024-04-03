import torch
import torch.nn as nn
import torch.nn.functional as F

class VCLLayer(nn.Module):
    def __init__(self, in_dim, out_dim, device, init_prior_mean=0.0, init_prior_log_var=0.0):
        super().__init__()

        self.prior_weight_mu = torch.full((in_dim, out_dim), init_prior_mean).to(device)
        self.prior_weight_log_var = torch.full((in_dim, out_dim), init_prior_log_var).to(device)

        self.prior_bias_mu = torch.full((out_dim,), init_prior_mean).to(device)
        self.prior_bias_log_var = torch.full((out_dim,), init_prior_log_var).to(device)

        self.weight_mu = torch.nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.weight_log_var = torch.nn.Parameter(torch.Tensor(in_dim, out_dim))

        self.bias_mu = torch.nn.Parameter(torch.Tensor(out_dim))
        self.bias_log_var = torch.nn.Parameter(torch.Tensor(out_dim))

        self.reset_parameters()

    def count_parameters(self):
        return self.weight_mu.numel() + self.bias_mu.numel()

    def reset_parameters(self):
        self.weight_mu.data.normal_(0, 0.1)
        self.weight_log_var.data.fill_(-6)

        self.bias_mu.data.normal_(0, 0.1)
        self.bias_log_var.data.fill_(-6)

    # TODO: Add multi-head support,
    # Expect tiled input, x: [no_samples, batch x y]
    def forward(self, x):
        weights_positive_std = torch.exp(0.5 * self.weight_log_var)
        bias_positive_std = torch.exp(0.5 * self.bias_log_var)

        num_samples = x.size(0)
        tiled_weights_mu = torch.tile(
            self.weight_mu, (num_samples, 1, 1))
        tiled_bias_mu = torch.tile(
            self.bias_mu, (num_samples, 1)
        )

        tiled_weights_std = torch.tile(
            weights_positive_std, (num_samples, 1, 1)
        )
        tiled_bias_std = torch.tile(
            bias_positive_std, (num_samples, 1)
        )

        weight_eps = torch.empty(tiled_weights_mu.size(), device=x.device).normal_(0, 1)
        bias_eps = torch.empty(tiled_bias_mu.size(), device=x.device).normal_(0, 1)

        sampled_weights = tiled_weights_mu + tiled_weights_std * weight_eps
        sampled_bias = tiled_bias_mu +  tiled_bias_std * bias_eps

        x = torch.einsum("kmn,kno->kmo", x, sampled_weights)
        x = x + sampled_bias.unsqueeze(1)
        return x

    def update_priors(self):
        self.prior_weight_mu = self.weight_mu.detach().clone()
        self.prior_weight_log_var = self.weight_log_var.detach().clone()

        self.prior_bias_mu = self.bias_mu.detach().clone()
        self.prior_bias_log_var = self.bias_log_var.detach().clone()

    def kl_divergence(self):
        weight_variance = torch.exp(self.weight_log_var)
        prior_weight_variance = torch.exp(self.prior_weight_log_var)

        bias_variance = torch.exp(self.bias_log_var)
        prior_bias_variance = torch.exp(self.prior_bias_log_var)

        weight_kl = self._kl_divergence(self.weight_mu, weight_variance, self.prior_weight_mu, prior_weight_variance)
        bias_kl = self._kl_divergence(self.bias_mu, bias_variance, self.prior_bias_mu, prior_bias_variance)

        return weight_kl + bias_kl

    def _kl_divergence(self, q_mu, q_var, p_mu, p_var):
        return 0.5 * ((q_var + (q_mu - p_mu) ** 2) / (p_var + 1e-12) - 1 + torch.log(p_var / (q_var + 1e-12))).sum()







