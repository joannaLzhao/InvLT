'''
Metrics to measure calibration of a trained deep neural network.

References:
[1] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks.
    arXiv preprint arXiv:1706.04599, 2017.
'''
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def test_classification_net_logits(logits, labels):

    labels_list = []
    predictions_list = []
    confidence_vals_list = []

    softmax = F.softmax(logits, dim=1)
    confidence_vals, predictions = torch.max(softmax, dim=1)
    labels_list.extend(labels.cpu().numpy().tolist())
    predictions_list.extend(predictions.cpu().numpy().tolist())
    confidence_vals_list.extend(confidence_vals.cpu().numpy().tolist())
    accuracy = accuracy_score(labels_list, predictions_list)
    return confusion_matrix(labels_list, predictions_list), accuracy, labels_list,\
        predictions_list, confidence_vals_list



# Calibration error scores in the form of loss metrics
class ECELoss(nn.Module):
    def __init__(self, n_bins=10):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, probs=None):
        if probs is None:
            softmaxes = F.softmax(logits, dim=1)
        else: 
            softmaxes = probs

        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class MaxECELoss(nn.Module):
    def __init__(self, n_bins=10):
        super(MaxECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, probs=None):
        if probs is None:
            softmaxes = F.softmax(logits, dim=1)
        else: 
            softmaxes = probs

        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                bin_gap = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                if bin_gap > ece:
                    ece = bin_gap

        return ece


class AdaptiveECELoss(nn.Module):
    def __init__(self, n_bins=10):
        super(AdaptiveECELoss, self).__init__()
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))
    def forward(self, logits, labels, probs=None):
        if probs is None:
            softmaxes = F.softmax(logits, dim=1)
        else: 
            softmaxes = probs
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class ClasswiseECELoss(nn.Module):
    def __init__(self, n_bins=10):
        super(ClasswiseECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, probs=None):
        num_classes = int((torch.max(labels) + 1).item())
        if probs is None:
            softmaxes = F.softmax(logits, dim=1)
        else: 
            softmaxes = probs
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)
        return sce


class KDEECELoss(nn.Module):

    def __init__(self, bandwidth=0.02, n_eval=200):

        super(KDEECELoss, self).__init__()
        self.bandwidth = bandwidth
        self.n_eval = n_eval

    def _gaussian_kernel(self, x, x_i, h):

        return torch.exp(-((x - x_i) ** 2) / (2 * h ** 2))

    def forward(self, logits, labels, probs=None):
        if probs is None:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = probs

        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels).float()

        # evaluation points
        eval_points = torch.linspace(0, 1, self.n_eval, device=logits.device)
        h = self.bandwidth

        # kernel weights: (n_eval, n_samples)
        weights = self._gaussian_kernel(
            eval_points.unsqueeze(1), confidences.unsqueeze(0), h)

        # density at each eval point
        density = weights.sum(dim=1)
        # avoid division by zero
        mask = density > 1e-12

        # weighted accuracy and confidence at each eval point
        acc_at_p = (weights * accuracies.unsqueeze(0)).sum(dim=1)
        conf_at_p = (weights * confidences.unsqueeze(0)).sum(dim=1)

        # calibration error: integrate |acc(p) - conf(p)| * density(p) dp
        # normalize by total density so it's a proper expectation
        total_density = density[mask].sum()
        kde_ece = (torch.abs(acc_at_p[mask] - conf_at_p[mask]) / total_density).sum()

        return kde_ece.unsqueeze(0)









