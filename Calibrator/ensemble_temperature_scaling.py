import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from scipy import optimize


class EnsembleTemperatureScaling(nn.Module):

    def __init__(self, loss='mse', log=True):
        super().__init__()
        self.loss = loss
        self.log = log
        self.temperature = 1.0
        self.weights = np.array([1.0, 0.0, 0.0])
        self.num_classes = None


    @staticmethod
    def _softmax_np(logits):
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _mse_t(t, logits, labels_onehot):
        probs = EnsembleTemperatureScaling._softmax_np(logits / t)
        return np.mean((probs - labels_onehot) ** 2)

    @staticmethod
    def _ll_t(t, logits, labels_onehot):
        probs = EnsembleTemperatureScaling._softmax_np(logits / t)
        probs = np.clip(probs, 1e-20, 1.0 - 1e-20)
        return -np.mean(labels_onehot * np.log(probs))

    @staticmethod
    def _mse_w(w, p0, p1, p2, labels_onehot):
        p = w[0] * p0 + w[1] * p1 + w[2] * p2
        p = p / p.sum(axis=1, keepdims=True)
        return np.mean((p - labels_onehot) ** 2)

    @staticmethod
    def _ll_w(w, p0, p1, p2, labels_onehot):
        p = w[0] * p0 + w[1] * p1 + w[2] * p2
        p = np.clip(p, 1e-20, 1.0 - 1e-20)
        return -np.mean(labels_onehot * np.log(p))


    def _fit_temperature(self, logits_np, labels_onehot):
        loss_fn = self._mse_t if self.loss == 'mse' else self._ll_t
        result = optimize.minimize(
            loss_fn, 1.0,
            args=(logits_np, labels_onehot),
            method='L-BFGS-B',
            bounds=((0.05, 5.0),),
            tol=1e-12,
        )
        self.temperature = result.x.item()

    def _fit_weights(self, logits_np, labels_onehot):
        p1 = self._softmax_np(logits_np)
        p0 = self._softmax_np(logits_np / self.temperature)
        p2 = np.ones_like(p0) / self.num_classes

        loss_fn = self._mse_w if self.loss == 'mse' else self._ll_w
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

        result = optimize.minimize(
            loss_fn, (1.0, 0.0, 0.0),
            args=(p0, p1, p2, labels_onehot),
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            tol=1e-12,
        )
        self.weights = result.x


    def _calibrate_probs_np(self, logits_np):
        p0 = self._softmax_np(logits_np / self.temperature)
        p1 = self._softmax_np(logits_np)
        p2 = np.ones_like(p0) / self.num_classes
        p = self.weights[0] * p0 + self.weights[1] * p1 + self.weights[2] * p2
        return p


    def forward(self, logits, **kwargs):
        return self.transform_logits(logits)

    def transform_logits(self, logits, **kwargs):
        logits_np = logits.detach().cpu().numpy()
        cal_probs = self._calibrate_probs_np(logits_np)
        cal_probs = np.clip(cal_probs, 1e-12, 1.0)
        cal_logits = np.log(cal_probs)
        return torch.tensor(cal_logits, dtype=logits.dtype, device=logits.device)

    def transform_probs(self, logits, **kwargs):
        logits_np = logits.detach().cpu().numpy()
        cal_probs = self._calibrate_probs_np(logits_np)
        return torch.tensor(cal_probs, dtype=logits.dtype, device=logits.device)

    def fit(self,
            logits,
            labels,
            device,
            **kwargs):
        self.to(device)

        logits_np = logits.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        self.num_classes = logits_np.shape[1]


        labels_onehot = np.eye(self.num_classes)[labels_np]

        self._fit_temperature(logits_np, labels_onehot)
        self._fit_weights(logits_np, labels_onehot)

        if self.log:
            print(f"ETS  T={self.temperature:.4f}, "
                  f"w=[{self.weights[0]:.4f}, {self.weights[1]:.4f}, "
                  f"{self.weights[2]:.4f}]")
