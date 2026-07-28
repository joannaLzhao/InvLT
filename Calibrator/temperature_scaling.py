import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

class TemperatureScaling(nn.Module):
    def __init__(self, log=True, loss='mse'):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))
        self.log = log
        loss = str(loss).lower()
        assert loss in ('ce', 'mse'), f"loss must be 'ce' or 'mse', got {loss}"
        self.loss = loss

    def forward(self, logits, **kwargs):
        return self.transform_logits(logits)

    def temperature(self):
        return self.log_temperature.exp()

    def transform_logits(self, logits, **kwargs):
        return logits / self.log_temperature.exp()

    def transform_probs(self, logits, **kwargs):
        return F.softmax(logits / self.log_temperature.exp(), dim=1)

    def fit(self,
            logits,
            labels,
            device,
            lr=1e-3,
            max_iter=10000,
            **kwargs):
        self.to(device)

        logits = logits.detach().to(device)
        labels = labels.to(device)
        num_classes = logits.shape[1]

        if self.loss == 'ce':
            criterion = nn.CrossEntropyLoss().to(device)
        else:
            criterion = nn.MSELoss().to(device)
        labels_onehot = F.one_hot(labels, num_classes=num_classes).float()

        def _loss(cal_logits):
            if self.loss == 'ce':
                return criterion(cal_logits, labels)
            return criterion(F.softmax(cal_logits, dim=1), labels_onehot)

        optimizer = optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')


        def closure():
            optimizer.zero_grad()
            loss = _loss(self.forward(logits))
            loss.backward()
            return loss

        with torch.enable_grad():
            loss_before = _loss(self.forward(logits)).item()
            optimizer.step(closure)
            loss_after = _loss(self.forward(logits)).item()

        if self.log:
            print(f"TempScaling[{self.loss}]  Loss: {loss_before:.4f} → {loss_after:.4f}, "
                  f"T = {self.log_temperature.exp().item():.4f}")

        T = self.log_temperature.exp().item()
        if not (0.01 < T < 100):
            raise ValueError(f"Temperature {T:.4f} is out of expected range (0.01, 100). "
                             "Check your logits/labels or adjust lr/max_iter.")
