import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class HistogramBinning(nn.Module):

    def __init__(self, num_bins=15, log=True):
        super().__init__()
        self.num_bins = num_bins
        self.log = log
        self.bin_boundaries = None
        self.bin_values = None
        self.num_classes = None
        self._device = None


    def _logits_to_confidences(self, logits):
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def _calibrate_probs(self, probs_np):
        N, C = probs_np.shape
        calibrated = np.zeros_like(probs_np)
        for c in range(C):

            bin_idx = np.digitize(probs_np[:, c], self.bin_boundaries) - 1
            bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)
            calibrated[:, c] = self.bin_values[c][bin_idx]

        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-12)
        calibrated = calibrated / row_sums
        return calibrated


    def forward(self, logits, **kwargs):
        return self.transform_logits(logits)

    def transform_logits(self, logits, **kwargs):
        probs_np = self._logits_to_confidences(logits)
        cal_probs = self._calibrate_probs(probs_np)
        cal_probs = np.clip(cal_probs, 1e-12, 1.0)
        cal_logits = np.log(cal_probs)
        return torch.tensor(cal_logits, dtype=logits.dtype, device=logits.device)

    def transform_probs(self, logits, **kwargs):
        probs_np = self._logits_to_confidences(logits)
        cal_probs = self._calibrate_probs(probs_np)
        return torch.tensor(cal_probs, dtype=logits.dtype, device=logits.device)

    def fit(self,
            logits,
            labels,
            device,
            num_bins=15,
            **kwargs):
        if num_bins is not None:
            self.num_bins = num_bins

        self._device = device
        self.to(device)

        logits = logits.detach().to(device)
        labels = labels.to(device)

        probs_np = self._logits_to_confidences(logits)
        labels_np = labels.detach().cpu().numpy()

        self.num_classes = probs_np.shape[1]


        self.bin_boundaries = np.linspace(0.0, 1.0, self.num_bins + 1)

        self.bin_values = np.zeros((self.num_classes, self.num_bins))

        for c in range(self.num_classes):
            y_binary = (labels_np == c).astype(np.float64)
            bin_idx = np.digitize(probs_np[:, c], self.bin_boundaries) - 1
            bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)

            for b in range(self.num_bins):
                mask = (bin_idx == b)
                if mask.sum() > 0:
                    self.bin_values[c, b] = y_binary[mask].mean()
                else:
                    # empty bin → use the bin midpoint as a fallback
                    self.bin_values[c, b] = (self.bin_boundaries[b] +
                                              self.bin_boundaries[b + 1]) / 2.0

        if self.log:
            print(f"HistogramBinning fitted on {len(labels_np)} samples, "
                  f"{self.num_classes} classes, {self.num_bins} bins")
