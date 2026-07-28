
import torch
import torch.nn as nn
from .parallel_neural_integral import ParallelNeuralIntegral


def _flatten(sequence):

    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


class IntegrandNN(nn.Module):


    def __init__(self, in_d, hidden_layers, cond_d=0):
        super().__init__()
        layers = []
        prev_d = in_d + cond_d
        for h_d in hidden_layers:
            layers.append(nn.Linear(prev_d, h_d))
            layers.append(nn.Sigmoid())
            prev_d = h_d
        layers.append(nn.Linear(prev_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, h=None):

        if h is not None and h.shape[-1] > 0:
            inp = torch.cat([x, h], dim=-1)
        else:
            inp = x
        raw = self.net(inp)
        # Ensure strict positivity: softplus(y) > 0 for all y
        # (softplus is smoother and strictly positive, unlike ELU+1 which can reach 0)
        return torch.nn.functional.softplus(raw)


class MonotonicNN(nn.Module):

    def __init__(self, in_d=1, hidden_layers=(20, 20), nb_steps=10):
        super().__init__()
        self.in_d = in_d
        self.nb_steps = nb_steps

        # Conditioning dimension: all input dims except the monotonic one
        cond_d = max(in_d - 1, 0)

        # Integrand network: f(t, h) > 0
        self.integrand = IntegrandNN(1, list(hidden_layers), cond_d=cond_d)

        # Learnable offset (bias)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Split into monotonic variable and conditioning
        x_mono = x[:, 0:1]  # (B, 1)
        h = x[:, 1:] if x.shape[-1] > 1 else torch.zeros(x.shape[0], 0, device=x.device)

        # Integration lower bound: 0
        x0 = torch.zeros_like(x_mono)

        # Compute integral using Clenshaw-Curtis quadrature
        flat_params = _flatten(self.integrand.parameters())
        integral = ParallelNeuralIntegral.apply(
            x0, x_mono, self.integrand, flat_params, h, self.nb_steps
        )

        return integral + self.beta
