import math
import warnings
from typing import Callable, Iterable, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.optim import Optimizer


class CosineDecay:
    """
    Applies cosine decay to a parameter (death_rate), using PyTorch's built-in
    `torch.optim.lr_scheduler.CosineAnnealingLR`.

    Args:
        death_rate (float): Initial value to be decayed.
        T_max (int): Maximum number of iterations for the decay.
        eta_min (float, optional): Minimum value of the parameter after decay.
            Defaults to 0.
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.
    """

    def __init__(self, death_rate: float, T_max: int, eta_min: float = 0, last_epoch: int = -1):
        self.sgd = optim.SGD(
            torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]),
            lr=death_rate,
        )
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.sgd, T_max + 1, eta_min, last_epoch
        )
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self, current_step: int) -> None:
        """
        Performs one step of the cosine decay scheduler.

        Args:
            current_step (int): Current step index.
        """
        self.cosine_stepper.step(current_step)

    def get_dr(self, current_step: int) -> float:
        """
        Returns the updated rate (death_rate) at the given step.

        Args:
            current_step (int): Current step index.

        Returns:
            float: The decayed parameter.
        """
        if current_step >= self.T_max:
            return self.eta_min
        self.step(current_step)
        return self.sgd.param_groups[0]["lr"]


class SPAM_optimizer(Optimizer):
    """
    SPAM !!!
    Implements the Adam algorithm with the weight decay fix, as introduced in
    "Decoupled Weight Decay Regularization" (https://arxiv.org/abs/1711.05101).

    .. warning::
        This implementation is deprecated and will be removed in a future version.
        Use `torch.optim.AdamW` instead, or set `no_deprecation_warning=True` to
        disable the warning.

    Args:
        params (Iterable[nn.parameter.Parameter]): Iterable of parameters to optimize or
            dictionaries defining parameter groups.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        betas (Tuple[float, float], optional): Coefficients used for computing
            running averages of gradient and its square. Defaults to (0.9, 0.999).
        eps (float, optional): Term added to the denominator to improve numerical
            stability. Defaults to 1e-6.
        weight_decay (float, optional): Weight decay (L2 penalty). Defaults to 0.0.
        correct_bias (bool, optional): Whether or not to correct bias in Adam.
            Defaults to True.
        no_deprecation_warning (bool, optional): Disable deprecation warning.
            Defaults to False.
        warmup_epoch (int, optional): Number of epochs to warm up. Defaults to 50.
        threshold (int, optional): Threshold for gradient masking. Defaults to 5000.
        grad_accu_steps (int, optional): Gradient accumulation steps before
            threshold-based masking applies. Defaults to 20.
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
        warmup_epoch: int = 50,
        threshold: int = 5000,
        grad_accu_steps: int = 20,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in "
                "a future version. Use `torch.optim.AdamW` instead, or set "
                "`no_deprecation_warning=True` to disable this warning.",
                FutureWarning,
            )

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

        # Initialize internal states
        self.init_masks()
        self.check_sparsity()

        self.state["total_step"] = 0
        self.state["current_step"] = warmup_epoch + 1
        self.update_proj_gap = 1e9  # Large initial value; updated later
        self.warmup_epoch = warmup_epoch
        self.warmup = CosineDecay(0.99, warmup_epoch)  # Warmup after momentum reset
        self.thres = threshold
        self.grad_accu_steps = grad_accu_steps

    def init_masks(self) -> None:
        """
        Initialize random masks for each parameter group that has 'density'.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "density" in group:
                    if "mask" not in state:
                        state["mask"] = self.initialize_random_rank_boolean_tensor(
                            p.data.shape[0],
                            p.data.shape[1],
                            group["density"],
                        ).to(p.device)

    def check_sparsity(self) -> None:
        """
        Print the overall density (non-zero fraction) of elements in the masks
        across all parameter groups that have 'density'.
        """
        total_num = 0
        non_zero_num = 0

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "density" in group and "mask" in state:
                    total_num += state["mask"].numel()
                    non_zero_num += state["mask"].sum().item()

        if total_num > 0:
            print("density", non_zero_num / total_num)
        else:
            print("No masks found for sparsity check.")

    @torch.no_grad()
    def step(self, closure: Callable = None) -> float:
        """
        Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that re-evaluates the model and
                returns the loss.

        Returns:
            float: The loss, if the closure was provided, otherwise None.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # scale factor is based on the cosine decay
        scale_factor = 1 - self.warmup.get_dr(self.state["current_step"])

        for group in self.param_groups:
            if "density" in group:
                # If the group has 'density', update `update_proj_gap`
                self.update_proj_gap = group["update_proj_gap"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients. Use SparseAdam instead.")

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0

                if "dim" not in group:
                    group["dim"] = 2

                # GaLore Projection for 'density' parameters
                if "density" in group:
                    state["mask"] = state["mask"].bool()
                    grad = grad[state["mask"]]

                # Initialize EMA states if necessary
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                # Reset momentum when total_step hits update_proj_gap
                if (self.state["total_step"] + 1) % self.update_proj_gap == 0:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                # Threshold-based gradient masking
                if self.thres != 0:
                    current_step = self.state["total_step"] + 1
                    if current_step >= self.grad_accu_steps:
                        exp_avg_sq1 = exp_avg_sq
                        mask = (grad**2) > (self.thres * exp_avg_sq1)
                        if self.update_proj_gap != 0:
                            # Only apply after enough accumulation steps
                            if current_step % self.update_proj_gap >= self.grad_accu_steps:
                                grad[mask]=grad[mask].sign()*torch.sqrt(exp_avg_sq1[mask]*self.thres)
                        else:
                            grad[mask]=grad[mask].sign()*torch.sqrt(exp_avg_sq1[mask]*self.thres)

                # Update exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    # Bias correction
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size *= math.sqrt(bias_correction2) / bias_correction1

                # Compute normalized gradient
                norm_grad = exp_avg / denom

                # Scatter updates back into the original parameter shape if 'density' is used
                if "density" in group:
                    grad_full = p.grad
                    grad_full[state["mask"]] = norm_grad
                    grad_full[~state["mask"]] = 0
                    grad_full.mul_(scale_factor)
                    p.add_(grad_full, alpha=-step_size)
                else:
                    p.add_(norm_grad, alpha=-step_size * scale_factor)

                # Weight decay
                if group["weight_decay"] > 0:
                    if "density" in group:
                        p.data[state["mask"]].add_(
                            p.data[state["mask"]],
                            alpha=(-group["lr"] * group["weight_decay"]),
                        )
                    else:
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        # Bookkeeping
        self.state["total_step"] += 1
        self.state["current_step"] += 1

        # Mask update if needed
        if (self.state["total_step"] != 0) and (
            (self.state["total_step"] + 1) % self.update_proj_gap == 0
        ):
            self.update_masks()
            print("Mask Update", flush=True)
            self.state["current_step"] = 0
            self.warmup = CosineDecay(0.99, self.warmup_epoch)

        return loss

    def update_masks(self) -> None:
        """
        Update masks in each parameter group that has 'density'. The new mask is
        selected randomly, and the overlap ratio with the old mask is printed.
        """
        overlap_ratio = 0.0
        for group in self.param_groups:
            print("lr", group["lr"])
            for p in group["params"]:
                state = self.state[p]
                if "density" in group:
                    assert len(p.data.shape) == 2
                    new_mask, overlap_ratio = self.update_mask_random(
                        group["density"], p, state["mask"]
                    )
                    state["mask"] = new_mask
                    p.mask = new_mask
        print(f"Mask overlap ratio: {overlap_ratio:.2f}")

    def update_mask_random(self, density: float, p: nn.parameter.Parameter, old_mask: torch.Tensor):
        """
        Create a new random mask with the same density, compute overlap ratio
        with old_mask, and update the exponential moving averages for the
        overlap region.

        Args:
            density (float): Fraction of elements to keep.
            p (nn.parameter.Parameter): Parameter to which the mask is applied.
            old_mask (torch.Tensor): Previous binary mask.

        Returns:
            Tuple[torch.Tensor, float]: The new binary mask and the overlap ratio.
        """
        m, n = p.data.shape
        total_elements = m * n
        state = self.state[p]
        non_zero_count = int(density * total_elements)

        new_mask = (torch.rand(p.data.shape, device=p.device) < density)

        # Calculate overlap ratio
        intersection_mask = new_mask & old_mask
        overlap_count = intersection_mask.sum().item()
        overlap_ratio = (overlap_count / non_zero_count) if non_zero_count else 0.0

        # Re-init exp_avg, exp_avg_sq, but copy over intersection
        exp_avg = torch.zeros_like(p.data[new_mask])
        exp_avg_sq = torch.zeros_like(p.data[new_mask])

        # Indices within the intersection for the new and old masks
        new_intersection_indices = intersection_mask[new_mask]
        old_intersection_indices = intersection_mask[old_mask]

        exp_avg[new_intersection_indices] = state["exp_avg"][old_intersection_indices]
        exp_avg_sq[new_intersection_indices] = state["exp_avg_sq"][old_intersection_indices]

        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq

        return new_mask, overlap_ratio

    def initialize_random_rank_boolean_tensor(self, m: int, n: int, density: float) -> torch.Tensor:
        """
        Create an (m x n) boolean tensor with `density` fraction of True entries.

        Args:
            m (int): Number of rows.
            n (int): Number of columns.
            density (float): Fraction of True entries (1.0 => all True).

        Returns:
            torch.Tensor: Binary tensor of shape (m, n).
        """
        total_elements = m * n
        non_zero_count = int(density * total_elements)

        tensor = torch.zeros((m, n), dtype=torch.bool)
        non_zero_count = min(non_zero_count, total_elements)

        if non_zero_count > 0:
            indices = torch.randperm(total_elements)[:non_zero_count]
            rows = indices // n
            cols = indices % n
            tensor[rows, cols] = True

        return tensor
