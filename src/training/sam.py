"""Sharpness-Aware Minimization (SAM) optimizer.

Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently
Improving Generalization", ICLR 2021.  https://arxiv.org/abs/2010.01412

Usage with the ClassifierTrainer (interval SAM):

    # Build SAM wrapping AdamW
    base_opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    optimizer = SAM(model.parameters(), base_optimizer=base_opt, rho=0.05)

    # Two-step update (every sam_interval steps; normal step otherwise)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    loss2 = criterion(model(x), y)
    loss2.backward()
    optimizer.second_step(zero_grad=True)

The interval trick (apply SAM every N steps, normal AdamW otherwise) keeps
the overhead at ~10% instead of 2× while retaining most of the benefit.
Use rho=0.05 for standard SAM or rho=2.0 for ASAM (adaptive variant).
"""

from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer wrapper.

    Wraps any base optimizer and adds the two-step SAM update:
      1. first_step:  ascent step to perturbed weights w + e_w
      2. second_step: gradient descent step from perturbed position,
                      then restore original weights

    Compatible with PyTorch AMP (GradScaler).  When using with GradScaler,
    call scaler.unscale_(optimizer) before first_step so gradients are in
    float32 when computing the perturbation norm.

    Args:
        params: model parameters (same as base optimizer)
        base_optimizer: an already-constructed torch.optim.Optimizer instance
        rho: neighbourhood size (perturbation radius); 0.05 is the paper default
        adaptive: use ASAM (scale-invariant, better for networks with BN)
    """

    def __init__(
        self,
        params,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
    ):
        defaults = {"rho": rho, "adaptive": adaptive}
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer
        # Keep our param_groups in sync with base optimizer
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Gradient ascent to the sharp weight perturbation e_w."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)   # move to perturbed weights
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore original weights, then perform the base optimizer step."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]   # restore w
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        """Not used in the two-step API; raises to prevent accidental calls."""
        raise NotImplementedError(
            "SAM requires explicit first_step/second_step calls. "
            "See ClassifierTrainer.train_one_epoch for the correct pattern."
        )

    def _grad_norm(self) -> torch.Tensor:
        # Compute L2 norm of all gradients across all param groups
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = (torch.abs(p) if group.get("adaptive") else 1.0) * p.grad
                norms.append(g.norm(p=2).to(shared_device))
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
