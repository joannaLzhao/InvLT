import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F


class ParameterizedTemperatureScaling(nn.Module):

    _ACTIVATIONS = {
        'relu': nn.ReLU, 'silu': nn.SiLU, 'tanh': nn.Tanh,
    }

    def __init__(self, n_layers=3, n_nodes=16, top_k_logits=10,
                 log=True, loss='mse', activation='tanh'):
        super().__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.top_k_logits = top_k_logits
        self.log = log
        loss = str(loss).lower()
        assert loss in ('ce', 'mse'), f"loss must be 'ce' or 'mse', got {loss}"
        self.loss = loss
        activation = str(activation).lower()
        assert activation in self._ACTIVATIONS, \
            f"activation must be one of {list(self._ACTIVATIONS)}, got {activation}"
        self.activation = activation
        self.mlp = None

    def _build_mlp(self, num_classes, device):
        input_dim = min(self.top_k_logits, num_classes)
        act_cls = self._ACTIVATIONS[self.activation]
        layers = []
        in_features = input_dim
        for _ in range(self.n_layers):
            layers.append(nn.Linear(in_features, self.n_nodes))
            layers.append(act_cls())
            in_features = self.n_nodes

        layers.append(nn.Linear(in_features, 1))
        self.mlp = nn.Sequential(*layers).to(device)

    def _predict_temperature(self, logits):
        sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
        if self.top_k_logits is not None:
            sorted_logits = sorted_logits[:, :self.top_k_logits]
        raw_T = self.mlp(sorted_logits)

        T = F.softplus(raw_T).clamp(min=1e-12, max=1e12)
        return T

    def forward(self, logits, **kwargs):
        return self.transform_logits(logits)

    def transform_logits(self, logits, **kwargs):
        T = self._predict_temperature(logits)
        return logits / T

    def transform_probs(self, logits, **kwargs):
        return F.softmax(self.transform_logits(logits), dim=1)

    def fit(self,
            logits,
            labels,
            device,
            lr=1e-3,
            weight_decay=1e-4,
            epochs=10000,
            batch_size=500,
            clip=1e2,
            **kwargs):
        logits = logits.detach().to(device).clamp(-clip, clip)
        labels = labels.to(device)
        num_classes = logits.shape[1]
        N = logits.shape[0]

        if self.top_k_logits is None:
            self.top_k_logits = num_classes

        self._build_mlp(num_classes, device)
        self.to(device)


        labels_onehot = F.one_hot(labels, num_classes).float()

        if self.loss == 'mse':
            criterion = nn.MSELoss().to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(self.mlp.parameters(), lr=lr,
                               weight_decay=weight_decay)

        def _loss_full():
            if self.loss == 'mse':
                return criterion(self.transform_probs(logits), labels_onehot)
            return criterion(self.transform_logits(logits), labels)

        with torch.enable_grad():
            loss_before = _loss_full().item()

            for epoch in range(epochs):

                idx = torch.randperm(N, device=device)[:int(min(batch_size, N))]
                logits_batch = logits[idx]

                optimizer.zero_grad()
                if self.loss == 'mse':
                    probs = self.transform_probs(logits_batch)
                    loss = criterion(probs, labels_onehot[idx])
                else:
                    cal_logits = self.transform_logits(logits_batch)
                    loss = criterion(cal_logits, labels[idx])
                loss.backward()
                optimizer.step()

            loss_after = _loss_full().item()

        if self.log:
            T_all = self._predict_temperature(logits).detach()
            print(f"PTS[{self.loss}]  Loss: {loss_before:.4f} → {loss_after:.4f}, "
                  f"T range: [{T_all.min().item():.4f}, "
                  f"{T_all.max().item():.4f}], "
                  f"T mean: {T_all.mean().item():.4f}")
