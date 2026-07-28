import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from sklearn.isotonic import IsotonicRegression as _IsotonicRegression


class IsotonicRegression(nn.Module):

    def __init__(self, log=True):
        super().__init__()
        self.log = log
        self.ir_models = []
        self.num_classes = None
        self._device = None


    def _logits_to_confidences(self, logits):
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def _calibrate_probs(self, probs_np):
        calibrated = np.zeros_like(probs_np)
        for c, ir in enumerate(self.ir_models):
            calibrated[:, c] = ir.predict(probs_np[:, c])

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
            **kwargs):
        self._device = device
        self.to(device)

        logits = logits.detach().to(device)
        labels = labels.to(device)

        probs_np = self._logits_to_confidences(logits)
        labels_np = labels.detach().cpu().numpy()

        self.num_classes = probs_np.shape[1]
        self.ir_models = []

        for c in range(self.num_classes):
            ir = _IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')

            y_binary = (labels_np == c).astype(np.float64)
            ir.fit(probs_np[:, c], y_binary)
            self.ir_models.append(ir)

        if self.log:
            print(f"IsotonicRegression fitted on {len(labels_np)} samples, "
                  f"{self.num_classes} classes")
