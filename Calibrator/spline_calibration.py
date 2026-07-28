import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class _CubicSpline:

    def __init__(self, x, y, kx, runout='natural'):
        self.kx = kx
        self.delta = kx[1] - kx[0]
        self.nknots = len(kx)
        self.runout = runout

        m_from_ky = self._ky_to_M()
        my_from_ky = np.concatenate([m_from_ky, np.eye(self.nknots)], axis=0)
        y_from_my = self._my_to_y(x)
        y_from_ky = y_from_my @ my_from_ky

        ky = np.linalg.lstsq(y_from_ky, y, rcond=-1)[0]
        self.my = my_from_ky @ ky

    def _ky_to_M(self):
        n = self.nknots
        delta = self.delta

        A = 4.0 * np.eye(n - 2)
        for i in range(1, n - 2):
            A[i - 1, i] = 1.0
            A[i, i - 1] = 1.0

        if self.runout == 'parabolic':
            A[0, 0] = 5.0
            A[-1, -1] = 5.0
        elif self.runout == 'cubic':
            A[0, 0] = 6.0
            A[0, 1] = 0.0
            A[-1, -1] = 6.0
            A[-1, -2] = 0.0

        B = np.zeros((n - 2, n))
        for i in range(n - 2):
            B[i, i] = 1.0
            B[i, i + 1] = -2.0
            B[i, i + 2] = 1.0
        B *= 6.0 / delta ** 2

        Ainv = np.linalg.inv(A)
        AinvB = Ainv @ B

        if self.runout == 'natural':
            z0 = np.zeros((1, n))
            z1 = np.zeros((1, n))
        elif self.runout == 'parabolic':
            z0 = AinvB[0].reshape(1, -1)
            z1 = AinvB[-1].reshape(1, -1)
        elif self.runout == 'cubic':
            z0 = (2.0 * AinvB[0] - AinvB[1]).reshape(1, -1)
            z1 = (2.0 * AinvB[-1] - AinvB[-2]).reshape(1, -1)
        else:
            raise ValueError(f"Unknown runout: {self.runout}")

        return np.concatenate([z0, AinvB, z1], axis=0)

    def _my_to_y(self, vecx):
        ndata = len(vecx)
        n = self.nknots
        delta = self.delta

        mM = np.zeros((ndata, n))
        my = np.zeros((ndata, n))

        for i, xx in enumerate(vecx):
            j = int(np.floor((xx - self.kx[0]) / delta))
            j = max(0, min(j, n - 2))
            x = xx - self.kx[j]

            mM[i, j] = -x ** 3 / (6.0 * delta) + x ** 2 / 2.0 - 2.0 * delta * x / 6.0
            mM[i, j + 1] = x ** 3 / (6.0 * delta) - delta * x / 6.0
            my[i, j] = -x / delta + 1.0
            my[i, j + 1] = x / delta

        return np.concatenate([mM, my], axis=1)

    def _my_to_dy(self, vecx):
        ndata = len(vecx)
        n = self.nknots
        h = self.delta

        mM = np.zeros((ndata, n))
        my = np.zeros((ndata, n))

        for i, xx in enumerate(vecx):
            j = int(np.floor((xx - self.kx[0]) / h))
            j = max(0, min(j, n - 2))
            x = xx - self.kx[j]

            mM[i, j] = -x ** 2 / (2.0 * h) + x - 2.0 * h / 6.0
            mM[i, j + 1] = x ** 2 / (2.0 * h) - h / 6.0
            my[i, j] = -1.0 / h
            my[i, j + 1] = 1.0 / h

        return np.concatenate([mM, my], axis=1)

    def evaluate(self, x):
        return self._my_to_y(x) @ self.my

    def evaluate_deriv(self, x):
        return self._my_to_dy(x) @ self.my


class _InterpolatedFunction:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, x):
        if x >= self.x[-1]:
            return self.y[-1]
        if x <= self.x[0]:
            return self.y[0]
        ind = np.searchsorted(self.x, x, side='left')
        if ind == 0:
            ind = 1
        alpha = x - self.x[ind - 1]
        beta = self.x[ind] - x
        denom = alpha + beta
        if denom == 0:
            return self.y[ind]
        return float((beta * self.y[ind - 1] + alpha * self.y[ind]) / denom)

    def transform_array(self, arr):
        return np.array([self(float(v)) for v in arr])


class SplineCalibration(nn.Module):

    def __init__(self, n_knots=6, spline_method='natural', log=True):
        super().__init__()
        self.n_knots = n_knots
        self.spline_method = spline_method
        self.log = log
        self.recal_funcs = []
        self.num_classes = None


    @staticmethod
    def _logits_to_probs(logits):
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    @staticmethod
    def _build_recal_func(scores, labels, n_knots, spline_method):
        order = scores.argsort()
        scores = scores[order]
        labels = labels[order]

        nsamples = len(scores)
        integrated_accuracy = np.cumsum(labels) / nsamples
        integrated_scores = np.cumsum(scores) / nsamples
        percentile = np.linspace(0.0, 1.0, nsamples)

        kx = np.linspace(0.0, 1.0, n_knots)
        spline = _CubicSpline(
            percentile,
            integrated_accuracy - integrated_scores,
            kx,
            runout=spline_method,
        )

        acc = spline.evaluate_deriv(percentile) + scores
        return _InterpolatedFunction(scores, acc)

    def _calibrate_probs(self, probs_np):
        calibrated = np.zeros_like(probs_np)
        for c, func in enumerate(self.recal_funcs):
            calibrated[:, c] = func.transform_array(probs_np[:, c])
        calibrated = np.clip(calibrated, 0.0, 1.0)

        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-12)
        calibrated = calibrated / row_sums
        return calibrated


    def forward(self, logits, **kwargs):
        return self.transform_logits(logits)

    def transform_logits(self, logits, **kwargs):
        probs_np = self._logits_to_probs(logits)
        cal_probs = self._calibrate_probs(probs_np)
        cal_probs = np.clip(cal_probs, 1e-12, 1.0)
        cal_logits = np.log(cal_probs)
        return torch.tensor(cal_logits, dtype=logits.dtype, device=logits.device)

    def transform_probs(self, logits, **kwargs):
        probs_np = self._logits_to_probs(logits)
        cal_probs = self._calibrate_probs(probs_np)
        return torch.tensor(cal_probs, dtype=logits.dtype, device=logits.device)

    def fit(self,
            logits,
            labels,
            device,
            **kwargs):
        self.to(device)

        logits = logits.detach().to(device)
        labels = labels.to(device)

        probs_np = self._logits_to_probs(logits)
        labels_np = labels.detach().cpu().numpy()
        self.num_classes = probs_np.shape[1]

        self.recal_funcs = []
        for c in range(self.num_classes):
            y_binary = (labels_np == c).astype(np.float64)
            func = self._build_recal_func(
                probs_np[:, c], y_binary, self.n_knots, self.spline_method
            )
            self.recal_funcs.append(func)

        if self.log:
            print(f"SplineCalibration fitted on {len(labels_np)} samples, "
                  f"{self.num_classes} classes, "
                  f"{self.n_knots} knots, method='{self.spline_method}'")
