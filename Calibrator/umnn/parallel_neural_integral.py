
import torch
import numpy as np


def compute_cc_weights(nb_steps):

    lam = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    lam = np.cos((lam @ lam.T) * np.pi / nb_steps)
    lam[:, 0] = 0.5
    lam[:, -1] = 0.5 * lam[:, -1]
    lam = lam * 2 / nb_steps

    W = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    W[np.arange(1, nb_steps + 1, 2)] = 0
    W = 2.0 / (1 - W ** 2)
    W[0] = 1
    W[np.arange(1, nb_steps + 1, 2)] = 0

    lam = torch.FloatTensor((lam @ W).squeeze())
    lam = lam[1:]  # Remove first weight (boundary)

    # Chebyshev nodes mapped to [0, 1]
    step_sizes = torch.FloatTensor(
        0.5 * (1.0 - np.cos(np.arange(1, nb_steps + 1, 1) * np.pi / nb_steps))
    )
    return lam, step_sizes


def _flatten(sequence):

    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def integrate(x0, nb_steps, step_sizes, integrand, h, compute_grad=False):

    # step_sizes: (nb_steps,) -> need to evaluate integrand at each step
    # x_positions = x0 + step_sizes * (x - x0) but here we just evaluate at x0 + step_sizes
    # The actual scaling by (x - x0) is done in the autograd function
    evals = []
    for i in range(nb_steps):
        x_point = x0 + step_sizes[i].to(x0.device)
        evals.append(integrand(x_point, h))
    evals = torch.stack(evals, dim=0)  # (nb_steps, B, 1)

    if not compute_grad:
        return evals

    # Compute gradients w.r.t. integrand parameters and h
    params = list(integrand.parameters())
    # Sum all evaluations for gradient computation
    total = evals.sum()
    grads = torch.autograd.grad(total, params + [h], allow_unused=True,
                                create_graph=False, retain_graph=True)
    grad_params = _flatten(grads[:len(params)])
    grad_h = grads[len(params)] if grads[len(params)] is not None else torch.zeros_like(h)

    return evals, grad_params, grad_h


class ParallelNeuralIntegral(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x0, x, integrand, flat_params, h, nb_steps=20):

        with torch.no_grad():
            lam, step_sizes = compute_cc_weights(nb_steps)
            lam = lam.to(x0.device)
            step_sizes = step_sizes.to(x0.device)

            # Evaluate integrand at each Clenshaw-Curtis quadrature point
            # x_point_i = x0 + step_sizes[i] * (x - x0),  shape (B, 1)
            diff = x - x0  # (B, 1)
            evals = []
            for i in range(nb_steps):
                x_point = x0 + step_sizes[i] * diff  # scalar * (B,1) -> (B,1)
                evals.append(integrand(x_point, h))
            evals = torch.stack(evals, dim=0)  # (nb_steps, B, 1)

            # Weighted sum: integral = (x - x0) * sum_i(lam_i * f(x_i))
            # lam: (nb_steps,) -> (nb_steps, 1, 1) for broadcast with (nb_steps, B, 1)
            integral = diff * (lam.view(-1, 1, 1) * evals).sum(0)

        ctx.save_for_backward(x0, x, h)
        ctx.integrand = integrand
        ctx.nb_steps = nb_steps

        return integral

    @staticmethod
    def backward(ctx, grad_output):

        x0, x, h = ctx.saved_tensors
        integrand = ctx.integrand
        nb_steps = ctx.nb_steps

        with torch.enable_grad():
            x0_ = x0.detach().requires_grad_(True)
            x_ = x.detach().requires_grad_(True)
            h_ = h.detach().requires_grad_(True)

            # f(x, h) for Leibniz rule at upper bound
            fx = integrand(x_, h_)
            # f(x0, h) for Leibniz rule at lower bound
            fx0 = integrand(x0_, h_)

            # Gradient w.r.t. x: f(x, h) * grad_output
            grad_x = fx * grad_output

            # Gradient w.r.t. x0: -f(x0, h) * grad_output
            grad_x0 = -fx0 * grad_output

            # Gradient w.r.t. integrand parameters and h:
            # We need to differentiate the integral w.r.t. params and h
            lam, step_sizes = compute_cc_weights(nb_steps)
            lam = lam.to(x0.device)
            step_sizes = step_sizes.to(x0.device)

            # Recompute with gradients enabled
            params = list(integrand.parameters())
            evals = []
            for i in range(nb_steps):
                x_point = x0_ + step_sizes[i] * (x_ - x0_)
                evals.append(integrand(x_point, h_))
            evals = torch.stack(evals, dim=0)

            integral_val = (x_ - x0_) * (lam.view(-1, 1, 1) * evals).sum(0)
            # Weight by grad_output
            loss = (integral_val * grad_output).sum()

            all_grads = torch.autograd.grad(loss, params + [h_], allow_unused=True)
            grad_params = _flatten([g if g is not None else torch.zeros_like(p)
                                    for g, p in zip(all_grads[:len(params)], params)])
            grad_h = all_grads[len(params)]
            if grad_h is None:
                grad_h = torch.zeros_like(h)

        return grad_x0, grad_x, None, grad_params, grad_h, None
