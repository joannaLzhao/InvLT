import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F


class DirichletCalibration(nn.Module):

    def __init__(self, lam=0.1, mu=0.1, log=True, loss='mse'):
        super().__init__()
        self.lam = lam
        self.mu = mu
        self.log = log
        loss = str(loss).lower()
        assert loss in ('ce', 'mse'), f"loss must be 'ce' or 'mse', got {loss}"
        self.loss = loss
        self.W = None
        self.b = None

    def _init_params(self, num_classes, device):
        self.W = nn.Parameter(torch.eye(num_classes, device=device))
        self.b = nn.Parameter(torch.zeros(num_classes, device=device))

    def _log_probs(self, logits):
        return F.log_softmax(logits, dim=1)

    def _odir_reg(self):
        K = self.W.shape[0]
        off_diag = self.W - torch.diag(torch.diag(self.W))
        reg = self.lam * (off_diag ** 2).sum() / (K * (K - 1)) \
            + self.mu * (self.b ** 2).sum() / K
        return reg

    def forward(self, logits, **kwargs):
        return self.transform_logits(logits)

    def transform_logits(self, logits, **kwargs):
        log_probs = self._log_probs(logits)
        return log_probs @ self.W + self.b

    def transform_probs(self, logits, **kwargs):
        return F.softmax(self.transform_logits(logits), dim=1)

    def fit(self,
            logits,
            labels,
            device,
            lr=1e-3,
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

        optimizer = optim.LBFGS([self.W, self.b], lr=lr, max_iter=max_iter,
                                line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            loss = _loss(self.forward(logits)) + self._odir_reg()
            loss.backward()
            return loss

        with torch.enable_grad():
            loss_before = _loss(self.forward(logits)).item()
            optimizer.step(closure)
            loss_after = _loss(self.forward(logits)).item()

        if self.log:
            print(f"DirichletCal[{self.loss}]   Loss: {loss_before:.4f} → {loss_after:.4f}, "
                  f"W shape: {tuple(self.W.shape)}, "
                  f"b range: [{self.b.min().item():.4f}, {self.b.max().item():.4f}]")
