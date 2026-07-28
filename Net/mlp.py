import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=10, hidden_dims=(256, 256, 256), dropout=0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Per-dataset factories — input_dim is fixed per dataset so pretrain.py


def mlp_covertype(num_classes=7, **kwargs):
    return MLP(input_dim=54, num_classes=num_classes, **kwargs)


def mlp_letter(num_classes=26, **kwargs):
    return MLP(input_dim=16, num_classes=num_classes, **kwargs)


def mlp_sensorless(num_classes=11, **kwargs):
    return MLP(input_dim=48, num_classes=num_classes, **kwargs)


def mlp_adni(num_classes=3, **kwargs):
    return MLP(input_dim=116, num_classes=num_classes,
               hidden_dims=(256, 256, 128), **kwargs)


def mlp_adni_v2(num_classes=3, **kwargs):
    kwargs.pop('dropout', None)
    return MLP(input_dim=116, num_classes=num_classes,
               hidden_dims=(64, 32), dropout=0.3, **kwargs)


def mlp_adni_shallow(num_classes=3, **kwargs):
    kwargs.pop('dropout', None)
    return MLP(input_dim=116, num_classes=num_classes,
               hidden_dims=(32,), dropout=0.3, **kwargs)


def mlp_adni_linear(num_classes=3, **kwargs):
    kwargs.pop('dropout', None)
    return MLP(input_dim=116, num_classes=num_classes,
               hidden_dims=(), dropout=0.0, **kwargs)


def mlp_dermatology(num_classes=6, **kwargs):
    kwargs.pop('dropout', None)
    return MLP(input_dim=34, num_classes=num_classes,
               hidden_dims=(64, 32), dropout=0.2)


def mlp_ecoli(num_classes=8, **kwargs):
    kwargs.pop('dropout', None)
    return MLP(input_dim=7, num_classes=num_classes,
               hidden_dims=(32, 16), dropout=0.1)


def mlp_thyroid(num_classes=3, **kwargs):
    kwargs.pop('dropout', None)
    return MLP(input_dim=5, num_classes=num_classes,
               hidden_dims=(16, 8), dropout=0.1)
