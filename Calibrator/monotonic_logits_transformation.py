import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_ACTIVATIONS = {
    'tanh':    nn.Tanh,
    'elu':     nn.ELU,
    'relu':    nn.ReLU,
    'gelu':    nn.GELU,
    'silu':    nn.SiLU,
    'sigmoid': nn.Sigmoid,
    'leaky_relu': nn.LeakyReLU,
    'softplus': nn.Softplus,
}


class _ScalarMLP(nn.Module):
    def __init__(self, hidden_dims, activation='tanh'):
        super().__init__()
        if isinstance(activation, str):
            key = activation.lower()
            if key not in _ACTIVATIONS:
                raise ValueError(
                    f"Unknown activation '{activation}'. "
                    f"Choose from: {list(_ACTIVATIONS.keys())}")
            act_cls = _ACTIVATIONS[key]
        else:
            act_cls = activation

        dims = [1] + list(hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_cls())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape
        x_flat = x.reshape(-1, 1)
        out = self.net(x_flat)

        return out.reshape(shape)


class MonotonicLogitsTransformation(nn.Module):

    def __init__(self, hidden_dims=(24, 24), clip_range=(-20, 20), lam=0.01,
                 num_rec_samples=100, batch_size=500, loss='ce',
                 activation='silu'):
        super().__init__()
        self.hidden_dims = list(hidden_dims)
        self.clip_low = float(clip_range[0])
        self.clip_high = float(clip_range[1])
        assert self.clip_low < self.clip_high, \
            f"clip_range must satisfy low < high, got ({self.clip_low}, {self.clip_high})"
        self.lam = float(lam)
        self.num_rec_samples = int(num_rec_samples)
        self.batch_size = int(batch_size)
        loss = str(loss).lower()
        assert loss in ('ce', 'mse'), f"loss must be 'ce' or 'mse', got {loss}"
        self.loss = loss
        self.activation = activation
        self.f = _ScalarMLP(self.hidden_dims, activation=activation)
        self.g = _ScalarMLP(self.hidden_dims, activation=activation)

    def _clip(self, z):
        return torch.clamp(z, self.clip_low, self.clip_high)

    def transform_logits(self, logits):
        return self.f(self._clip(logits))

    def transform_probs(self, logits):
        return F.softmax(self.transform_logits(logits), dim=-1)

    def forward(self, logits):
        return self.transform_logits(logits)

    def reconstruction_loss(self, device):
        x = torch.linspace(self.clip_low, self.clip_high,
                           self.num_rec_samples, device=device).reshape(-1, 1)
        x_hat = self.g(self.f(x))
        return ((x_hat - x) ** 2).mean()

    def fit(self, logits, labels, device='cpu', lr=1e-3, max_iter=10000, weight_decay=0):
        self.to(device)
        logits = logits.to(device)
        labels = labels.to(device)
        self.clip_low = logits.min()
        self.clip_high = logits.max()

        N = logits.shape[0]
        bs = min(self.batch_size, N)
        num_classes = logits.shape[-1]

        if self.loss == 'ce':
            criterion = nn.CrossEntropyLoss().to(device)
        else:
            criterion = nn.MSELoss().to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(max_iter):
            idx = torch.randperm(N, device=device)[:bs]
            batch_logits = logits[idx]
            batch_labels = labels[idx]

            optimizer.zero_grad()
            cal_logits = self.transform_logits(batch_logits)
            if self.loss == 'ce':
                loss_cal = criterion(cal_logits, batch_labels)
            else:
                cal_probs = F.softmax(cal_logits, dim=-1)
                target = F.one_hot(batch_labels, num_classes=num_classes).float()
                loss_cal = criterion(cal_probs, target)
            loss_rec = self.reconstruction_loss(device)
            loss = loss_cal + (epoch > 100) * self.lam * loss_rec
            loss.backward()
            optimizer.step()
            print(f'loss_cal:{loss_cal:.4f}, loss_rec:{loss_rec:.4f}')


        # plt.title(f'Learned monotonic logit transformation (mean={mean_val:.4f})')


        return self
