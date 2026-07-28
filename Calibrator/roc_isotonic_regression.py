"""
Multiclass ROC-Regularized Isotonic Regression (MulticlassIRP).

Reference:
  Berta, Bach & Jordan, "Classifier Calibration with ROC-Regularized
  Isotonic Regression", AISTATS 2024. https://arxiv.org/abs/2311.12436
  Official code (Julia): https://github.com/eugeneberta/Calibration-ROC-IR

This is a direct Python port of the authors' `MulticlassIRP` algorithm
(utils.jl): a K-class calibration map is built by recursively partitioning
the probability simplex, at each step choosing the split (from a fixed
candidate grid over the simplex) that changes calibrated predictions the
most while keeping the partition consistent with a pairwise monotonicity
constraint (this is what preserves the ROC/VUS surface). New points are
calibrated by 1-nearest-neighbor lookup against the finished grid, exactly
as in the authors' own test-time evaluation code.

Scalability note (see supplementary_materials/README.md): the candidate
grid enumerates lattice points on the (K-1)-simplex, so its size grows
combinatorially with K. The authors' own experiments never exceed K=4
classes. This port raises rather than silently degrading when the
requested resolution would blow past a sane grid size for the given K.
"""
import itertools
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _simplex_grid(k, resolution):
    """All (k)-tuples of non-negative ints summing to `resolution`, /resolution."""
    n = resolution
    grid = []
    # stars-and-bars: choose k-1 divider positions among n+k-1 slots
    for divs in itertools.combinations(range(n + k - 1), k - 1):
        prev = -1
        parts = []
        for d in divs:
            parts.append(d - prev - 1)
            prev = d
        parts.append(n + k - 1 - 1 - prev)
        grid.append(parts)
    return np.asarray(grid, dtype=np.float64) / n


def _grid_size(k, resolution):
    from math import comb
    return comb(resolution + k - 1, k - 1)


def _pick_resolution(k, max_grid_points):
    """Largest resolution whose simplex grid stays under max_grid_points."""
    res = 1
    if _grid_size(k, res) > max_grid_points:
        return None
    while _grid_size(k, res + 1) <= max_grid_points:
        res += 1
    return res


def _laplace_mean(y_rows, lam=0.05):
    # y_rows: (n, K) one-hot rows -> smoothed class-probability vector.
    # Matches the authors' own smoothing strength (utils.jl `split`: (mean .+ λ) ./ (1+λ),
    # a per-class additive nudge of λ rather than a K-scaled Dirichlet prior -- dividing by
    # (1+λ*K) instead, as a naive normalization "fix" would, over-smooths badly for K>~4 and
    # crushes high-confidence regions (0.98 -> 0.76 at K=7 instead of ~0.98). The Julia
    # version doesn't sum to 1 after this step; we renormalize by row sum to get a valid
    # probability vector while preserving the same (gentle) smoothing strength.
    m = y_rows.sum(axis=0) / len(y_rows)
    m = (m + lam) / (1.0 + lam)
    return m / m.sum()


def _split(x, y, g, p):
    """Partition x (and grid g) into K regions by argmax(coord - p); return
    per-row calibrated values for x and g, plus the index groups."""
    n, K = y.shape
    ysplit = np.zeros_like(y)
    ygsplit = np.zeros((len(g), K))

    dx = x - p[None, :]
    rmax_x = dx.max(axis=1)
    dg = g - p[None, :]
    rmax_g = dg.max(axis=1)

    xregions, gregions = [], []
    for kk in range(K):
        Rk = np.where(dx[:, kk] == rmax_x)[0]
        Gk = np.where(dg[:, kk] == rmax_g)[0]
        if len(Rk) > 0:
            mean = _laplace_mean(y[Rk])
            ysplit[Rk] = mean
            if len(Gk) > 0:
                ygsplit[Gk] = mean
            xregions.append(Rk)
            gregions.append(Gk)
    return ysplit, ygsplit, xregions, gregions


def _monotone(x, y, split_points):
    """Check the pairwise monotonicity constraint against every chosen
    split reference point so far (x: full cal set, y: candidate full
    calibrated array, split_points: (S, K) references chosen so far)."""
    if len(split_points) == 0:
        return True
    K = x.shape[1]
    for p in split_points:
        d = x - p[None, :]
        rmax = d.max(axis=1)
        region_of = np.argmax(d, axis=1)
        on_max = (d == rmax[:, None])
        for i in range(K):
            ri = np.where(on_max[:, i])[0]
            if len(ri) == 0:
                continue
            for j in range(K):
                if i == j:
                    continue
                rj = np.where(on_max[:, j])[0]
                if len(rj) == 0:
                    continue
                if (y[ri, i] - y[ri, j]).min() < (y[rj, i] - y[rj, j]).max():
                    return False
    return True


def _optimal_split(x, y, g, yirp, ygirp, R, G, split_points):
    best_idx, best_M = -1, 0.0
    for gi in G:
        p = g[gi]
        ysplit, _, splitregions, _ = _split(x[R], y[R], g[G], p)
        if len(splitregions) < 2:
            continue
        yirp_try = yirp.copy()
        yirp_try[R] = ysplit
        M = np.abs(yirp_try - yirp).sum()
        if M <= best_M:
            continue
        if _monotone(x, yirp_try, np.vstack([split_points, p[None, :]]) if len(split_points) else p[None, :]):
            best_idx, best_M = gi, M
    if best_idx < 0:
        return yirp, ygirp, [], [], -1, 0.0
    p = g[best_idx]
    ysplit, ygsplit, xregions, gregions = _split(x[R], y[R], g[G], p)
    yirp = yirp.copy()
    ygirp = ygirp.copy()
    yirp[R] = ysplit
    ygirp[G] = ygsplit
    xregions = [R[rk] for rk in xregions]
    gregions = [G[gk] for gk in gregions]
    return yirp, ygirp, xregions, gregions, best_idx, best_M


def multiclass_irp(xcal, ycal, grid, max_regions=2000, verbose=False):
    """xcal: (N,K) probs, ycal: (N,K) one-hot, grid: (M,K) simplex lattice.
    Returns (ycal_irp (N,K), grid, y_grid (M,K))."""
    n, K = ycal.shape
    m = grid.shape[0]

    mean = ycal.sum(axis=0) / n
    ycal_irp = np.tile(mean, (n, 1))
    y_grid = np.tile(mean, (m, 1))

    regions = [np.arange(n)]
    grids = [np.arange(m)]
    split_points = np.zeros((0, K))

    n_splits = 0
    while n_splits < max_regions:
        best_region, best_M, best_result = -1, 0.0, None
        for i in range(len(regions)):
            res = _optimal_split(xcal, ycal, grid, ycal_irp, y_grid,
                                  regions[i], grids[i], split_points)
            if res[4] >= 0 and res[5] > best_M:
                best_region, best_M, best_result = i, res[5], res
        if best_region < 0:
            break
        ycal_irp, y_grid, xregions, gregions, split_idx, M = best_result
        split_points = np.vstack([split_points, grid[split_idx][None, :]])
        regions.pop(best_region)
        grids.pop(best_region)
        regions.extend(xregions)
        grids.extend(gregions)
        n_splits += 1
        if verbose:
            print(f'  [ROC-IR] split {n_splits}: {len(regions)} regions, M={M:.4f}', flush=True)

    return ycal_irp, grid, y_grid


class ROCIsotonicRegression(nn.Module):
    """See module docstring. `max_grid_points` bounds the simplex lattice
    size; classes with too few feasible grid points at resolution>=1 for
    that bound raise ValueError (this is expected/by-design for K in the
    hundreds+ -- the method as published doesn't scale there)."""

    def __init__(self, max_grid_points=25000, max_regions=2000, log=True):
        super().__init__()
        self.max_grid_points = max_grid_points
        self.max_regions = max_regions
        self.log = log
        self.grid = None
        self.y_grid = None
        self.num_classes = None

    def _logits_to_confidences(self, logits):
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def _calibrate_probs(self, probs_np):
        # 1-NN lookup against the finished grid, exactly as in the paper's
        # own test-time evaluation code.
        d = ((probs_np[:, None, :] - self.grid[None, :, :]) ** 2).sum(axis=2)
        idx = d.argmin(axis=1)
        return self.y_grid[idx]

    def forward(self, logits, **kwargs):
        return self.transform_logits(logits)

    def transform_logits(self, logits, **kwargs):
        probs_np = self._logits_to_confidences(logits)
        cal_probs = np.zeros_like(probs_np)
        # chunk to bound the (N_test x N_grid x K) distance matrix
        chunk = max(1, (20_000_000 // max(1, self.grid.shape[0] * self.num_classes)))
        for start in range(0, len(probs_np), chunk):
            cal_probs[start:start + chunk] = self._calibrate_probs(probs_np[start:start + chunk])
        cal_probs = np.clip(cal_probs, 1e-12, 1.0)
        cal_logits = np.log(cal_probs)
        return torch.tensor(cal_logits, dtype=logits.dtype, device=logits.device)

    def transform_probs(self, logits, **kwargs):
        probs_np = self._logits_to_confidences(logits)
        cal_probs = self._calibrate_probs(probs_np)
        return torch.tensor(cal_probs, dtype=logits.dtype, device=logits.device)

    def fit(self, logits, labels, device, resolution=None, **kwargs):
        logits = logits.detach().to(device)
        labels = labels.to(device)

        probs_np = self._logits_to_confidences(logits)
        labels_np = labels.detach().cpu().numpy()
        n, K = probs_np.shape
        self.num_classes = K

        if resolution is None:
            resolution = _pick_resolution(K, self.max_grid_points)
        if resolution is None:
            raise ValueError(
                f"ROCIsotonicRegression: {K}-class simplex has no grid resolution "
                f"under max_grid_points={self.max_grid_points} (even resolution=1 "
                f"gives {_grid_size(K, 1)} points). This method's published grid "
                f"construction is combinatorial in K and is not intended for "
                f"this many classes -- see README.md.")

        grid = _simplex_grid(K, resolution)
        y_onehot = np.eye(K)[labels_np]

        if self.log:
            print(f"[ROCIsotonicRegression] K={K}  resolution={resolution}  "
                  f"grid_points={len(grid)}  n_cal={n}", flush=True)

        y_cal_irp, self.grid, self.y_grid = multiclass_irp(
            probs_np, y_onehot, grid, max_regions=self.max_regions, verbose=self.log)

        if self.log:
            print(f"[ROCIsotonicRegression] fitted, "
                  f"{np.abs(self.y_grid - self.y_grid[0]).sum(axis=1).astype(bool).sum()} "
                  f"grid points differ from the global-mean initialization", flush=True)
