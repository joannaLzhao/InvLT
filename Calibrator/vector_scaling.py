'''
Code to perform vector scaling.
Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)

Vector scaling learns a per-class temperature and bias:
    calibrated_logits = logits * W + b
where W is a diagonal matrix (one scalar per class) and b is a bias vector.
'''
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F


class VectorScaling(nn.Module):


    def __init__(self, log=True, loss='ce'):
        super().__init__()
        self.log = log
        loss = str(loss).lower()
        assert loss in ('ce', 'mse'), f"loss must be 'ce' or 'mse', got {loss}"
        self.loss = loss
        # Parameters are lazily initialized in fit() once we know num_classes
        self.W = None  # nn.Parameter, shape (num_classes,)
        self.b = None  # nn.Parameter, shape (num_classes,)

    def _init_params(self, num_classes, device):
        """Initialize W=1, b=0 (identity transform at start)."""
        self.W = nn.Parameter(torch.ones(num_classes, device=device))
        self.b = nn.Parameter(torch.zeros(num_classes, device=device))

    def forward(self, logits, **kwargs):
        return self.transform_logits(logits)

    def transform_logits(self, logits, **kwargs):
        return logits * self.W + self.b

    def transform_probs(self, logits, **kwargs):
        return F.softmax(self.transform_logits(logits), dim=1)

    def fit(self,
            logits,
            labels,
            device,
            lr=1e-2,
            max_iter=10000,
            **kwargs):

        logits = logits.detach().to(device)
        labels = labels.to(device)
        num_classes = logits.shape[1]

        self._init_params(num_classes, device)
        self.to(device)

        if self.loss == 'ce':
            criterion = nn.CrossEntropyLoss().to(device)
        else:
            criterion = nn.MSELoss().to(device)
        labels_onehot = F.one_hot(labels, num_classes=num_classes).float()

        def _loss(cal_logits):
            if self.loss == 'ce':
                return criterion(cal_logits, labels)
            return criterion(F.softmax(cal_logits, dim=1), labels_onehot)

        optimizer = optim.Adam([self.W, self.b], lr=lr)

        print(f"[VectorScaling] SANITY: optimizer=Adam  lr={lr}  loss={self.loss}  "
              f"num_classes={num_classes}  n_val={logits.shape[0]}  max_iter={max_iter}")

        with torch.enable_grad():
            loss_before = _loss(self.forward(logits)).item()
            for step in range(max_iter):
                optimizer.zero_grad()
                loss = _loss(self.forward(logits))
                loss.backward()
                optimizer.step()
                if step == 99:   # log after 100 steps to confirm movement
                    print(f"[VectorScaling] step=100  loss={loss.item():.4f}  "
                          f"W_mean={self.W.mean().item():.4f}  b_mean={self.b.mean().item():.4f}")
            loss_after = _loss(self.forward(logits)).item()

        print(f"[VectorScaling] DONE  loss: {loss_before:.4f} → {loss_after:.4f}  "
              f"W=[{self.W.min().item():.4f}, {self.W.max().item():.4f}]  "
              f"b=[{self.b.min().item():.4f}, {self.b.max().item():.4f}]  "
              f"W_changed={not torch.allclose(self.W, torch.ones_like(self.W), atol=1e-4)}")
