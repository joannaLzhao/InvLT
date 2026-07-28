"""
UMNN-based Monotonic Logits Calibration.

Reference:
  Wehenkel & Louppe, "Unconstrained Monotonic Neural Networks", NeurIPS 2019.
  Rahimi et al., "Intra Order-Preserving Functions for Calibration of
  Multi-Class Neural Networks", NeurIPS 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .umnn import MonotonicNN


class UMNNCalibration(nn.Module):


    def __init__(self, hidden_dims=(24, 24), clip_range=(-20, 20.0),
                 batch_size=100, loss='mse'):
        super().__init__()
        self.hidden_dims = list(hidden_dims)
        self.clip_low = float(clip_range[0])
        self.clip_high = float(clip_range[1])
        assert self.clip_low < self.clip_high
        self.batch_size = int(batch_size)

        loss = str(loss).lower()
        assert loss in ('ce', 'mse'), f"loss must be 'ce' or 'mse', got {loss}"
        self.loss_type = loss

        # UMNN: scalar input (in_d=1), no conditioning
        self.umnn = MonotonicNN(hidden_layers=self.hidden_dims)

    def _clip(self, z):
        return torch.clamp(z, self.clip_low, self.clip_high)

    def transform_logits(self, logits):

        shape = logits.shape
        z = self._clip(logits)
        # Flatten to (N*C, 1) for the UMNN
        z_flat = z.reshape(-1, 1)
        out_flat = self.umnn(z_flat)
        return out_flat.reshape(shape)

    def transform_probs(self, logits):

        return F.softmax(self.transform_logits(logits), dim=-1)

    def forward(self, logits):
        return self.transform_logits(logits)

    def fit(self, logits, labels, device='cpu', lr=1e-3, max_iter=10000, weight_decay=0):

        self.to(device)
        logits = logits.to(device)
        labels = labels.to(device)

        # Adapt clip range to actual data range
        self.clip_low = logits.min().item()
        self.clip_high = logits.max().item()

        N = logits.shape[0]
        bs = min(self.batch_size, N)
        num_classes = logits.shape[-1]

        if self.loss_type == 'ce':
            criterion = nn.CrossEntropyLoss().to(device)
        else:
            criterion = nn.MSELoss().to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(max_iter):
            self.train()
            idx = torch.randperm(N, device=device)[:bs]
            batch_logits = logits[idx]
            batch_labels = labels[idx]

            optimizer.zero_grad()
            cal_logits = self.transform_logits(batch_logits)

            if self.loss_type == 'ce':
                loss = criterion(cal_logits, batch_labels)
            else:
                cal_probs = F.softmax(cal_logits, dim=-1)
                target = F.one_hot(batch_labels, num_classes=num_classes).float()
                loss = criterion(cal_probs, target)

            loss.backward()
            optimizer.step()

            if epoch % 500 == 0 or epoch == max_iter - 1:
                print(f'[UMNNCalibration] epoch {epoch}/{max_iter}  loss={loss.item():.4f}')

        # --- Visualization ---
        # self.eval()
        # with torch.no_grad():
        #     xs = torch.linspace(self.clip_low, self.clip_high, 1000, device=device).reshape(-1, 1)
        #     fs = self.umnn(xs)

        # xs_np = xs.cpu().numpy().ravel()
        # fs_np = fs.cpu().numpy().ravel()

        # mean_val = logits.flatten()[0].item()
        # idx_str = f'{mean_val:.4f}'.replace('.', 'p').replace('-', 'm')

        # fig = plt.figure(figsize=(6, 5))
        # plt.plot(xs_np, xs_np, 'k--', linewidth=1, label='identity')
        # plt.plot(xs_np, fs_np, 'b-', linewidth=2, label='UMNN F(x)')
        # plt.xlabel('input logit x')
        # plt.ylabel('transformed value')
        # plt.title(f'UMNN monotonic calibration (first_logit={mean_val:.4f})')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()

        # save_dir = 'results'
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f'umnn_calibration_{idx_str}.png')
        # fig.savefig(save_path, dpi=150, bbox_inches='tight')
        # plt.close(fig)
        # print(f'[UMNNCalibration] plot saved to {save_path}')

        # self.train()
        return self
