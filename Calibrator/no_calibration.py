import torch
from torch import nn
from torch.nn import functional as F


class NoCalibration(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, logits, **kwargs):
        return logits

    def transform_logits(self, logits, **kwargs):
        return logits

    def transform_probs(self, logits, **kwargs):
        return F.softmax(logits, dim=1)

    def fit(self, logits, labels, device='cpu', **kwargs):
        return self
