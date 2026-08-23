"""SOAP — ShampoO with Adam in the Preconditioner's eigenbasis.

Port of the official reference implementation
    https://github.com/nikhilvyas/SOAP
accompanying Vyas et al., "SOAP: Improving and Stabilizing Shampoo using
Adam." arXiv:2409.11321 (2024).

SOAP runs Adam in a rotated coordinate frame defined by Shampoo's
preconditioner eigenbasis ``Q``. ``Q`` itself is refreshed only every
``precondition_frequency`` steps, which makes SOAP much cheaper than
naive matrix methods while keeping the second-order behaviour.
"""
from itertools import chain
from typing import Iterable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

_Params = Union[Iterable[torch.Tensor], Iterable[dict]]


class SOAP(Optimizer):
    """SOAP optimizer.

    Args:
        params: iterable of parameters or parameter group dicts.
        lr: learning rate.
        betas: ``(beta1, beta2)`` for Adam in the rotated basis.
        shampoo_beta: EMA decay for the Shampoo accumulators (``L``, ``R``
            in the paper). If negative (default), ``betas[1]`` is reused.
        eps: epsilon for Adam.
        weight_decay: decoupled weight-decay coefficient (AdamW style).
        precondition_frequency: refresh the eigenbasis ``Q`` every this many
            steps.
        max_precond_dim: skip preconditioning along any dimension whose
            size exceeds this threshold. Default 10000 excludes most
            vocabulary axes while keeping hidden axes preconditioned.
        merge_dims: collapse consecutive dimensions whose product fits into
            ``max_precond_dim`` before preconditioning. Useful for 4-D
            convolutional weights.
        precondition_1d: also precondition 1-D parameters (norms, biases).
        normalize_grads: rescale each per-parameter update to unit RMS.
            Recommended only with large ``precondition_frequency``.
        data_format: ``"channels_first"`` (NCHW) or ``"channels_last"``
            (NHWC) — only matters when ``merge_dims=True`` for 4-D tensors.
        correct_bias: use Adam's bias correction.
    """

    def __init__(
        self,
        params: _Params,
        lr: float = 3e-3,
        betas: Tuple[float, float] = (0.95, 0.95),
        shampoo_beta: float = -1.0,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if precondition_frequency < 1:
            raise ValueError(f"precondition_frequency must be >= 1, got {precondition_frequency}")
        if data_format not in ("channels_first", "channels_last"):
            raise ValueError(f"Invalid data_format: {data_format}")

        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            merge_dims=merge_dims,
            precondition_1d=precondition_1d,
            normalize_grads=normalize_grads,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)
        self._data_format = data_format

    # ------------------------------------------------------------------ utils

    def _merge_dims(self, grad: torch.Tensor, max_precond_dim: int) -> torch.Tensor:
        """Collapse consecutive dimensions whose product is below
        ``max_precond_dim``."""
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)
        shape = grad.shape
        new_shape = []
        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape
        if curr_shape > 1 or len(new_shape) == 0:
            new_shape.append(curr_shape)
        return grad.reshape(new_shape)

    # ------------------------------------------------------------------ step

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                if "Q" not in state:
                    self._init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=group["precondition_frequency"],
                        precondition_1d=group["precondition_1d"],
                        shampoo_beta=(
                            group["shampoo_beta"]
                            if group["shampoo_beta"] >= 0
                            else group["betas"][1]
                        ),
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                    )
                    self._update_preconditioner(
                        grad,
                        state,
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                        precondition_1d=group["precondition_1d"],
                    )
                    # Skip the first step: never use current grads in their own projection.
                    continue

                grad_projected = self._project(
                    grad,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                exp_avg_projected = self._project(
                    exp_avg,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

                norm_grad = self._project_back(
                    exp_avg_projected / denom,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                if group["normalize_grads"]:
                    norm_grad = norm_grad / (1e-30 + torch.mean(norm_grad.square()) ** 0.5)

                p.add_(norm_grad, alpha=-step_size)

                # Decoupled weight decay.
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                # Refresh the eigenbasis from updated Shampoo accumulators.
                self._update_preconditioner(
                    grad,
                    state,
                    max_precond_dim=group["max_precond_dim"],
                    merge_dims=group["merge_dims"],
                    precondition_1d=group["precondition_1d"],
                )

        return loss

    # -------------------------------------------------------- preconditioner

    def _init_preconditioner(
        self,
        grad: torch.Tensor,
        state: dict,
        precondition_frequency: int,
        shampoo_beta: float,
        max_precond_dim: int,
        precondition_1d: bool,
        merge_dims: bool,
    ) -> None:
        state["GG"] = []
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(
                    torch.zeros(grad.shape[0], grad.shape[0], device=grad.device, dtype=grad.dtype)
                )
        else:
            shape_iter = self._merge_dims(grad, max_precond_dim).shape if merge_dims else grad.shape
            for sh in shape_iter:
                if sh > max_precond_dim:
                    state["GG"].append([])
                else:
                    state["GG"].append(torch.zeros(sh, sh, device=grad.device, dtype=grad.dtype))

        state["Q"] = None
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    def _project(
        self,
        grad: torch.Tensor,
        state: dict,
        merge_dims: bool,
        max_precond_dim: int,
    ) -> torch.Tensor:
        original_shape = grad.shape
        permuted_shape = original_shape
        if merge_dims:
            if grad.dim() == 4 and self._data_format == "channels_last":
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self._merge_dims(grad, max_precond_dim)

        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [0]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def _project_back(
        self,
        grad: torch.Tensor,
        state: dict,
        merge_dims: bool,
        max_precond_dim: int,
    ) -> torch.Tensor:
        original_shape = grad.shape
        permuted_shape = original_shape
        if merge_dims:
            if self._data_format == "channels_last" and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self._merge_dims(grad, max_precond_dim)

        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [1]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def _update_preconditioner(
        self,
        grad: torch.Tensor,
        state: dict,
        max_precond_dim: int,
        merge_dims: bool,
        precondition_1d: bool,
    ) -> None:
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state["GG"][0].lerp_(
                    grad.unsqueeze(1) @ grad.unsqueeze(0), 1 - state["shampoo_beta"]
                )
        else:
            g = self._merge_dims(grad, max_precond_dim) if merge_dims else grad
            for idx, sh in enumerate(g.shape):
                if sh <= max_precond_dim:
                    outer = torch.tensordot(
                        g,
                        g,
                        dims=[[*chain(range(idx), range(idx + 1, g.ndim))]] * 2,
                    )
                    state["GG"][idx].lerp_(outer, 1 - state["shampoo_beta"])

        if state["Q"] is None:
            state["Q"] = self._orthogonal_matrix(state["GG"])
        if state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            old_q = state["Q"]
            new_q = self._orthogonal_matrix_QR(state, max_precond_dim, merge_dims)
            variance = state["exp_avg_sq"]
            original_shape = variance.shape
            permuted_shape = original_shape
            if merge_dims:
                if self._data_format == "channels_last" and variance.dim() == 4:
                    permuted_shape = variance.permute(0, 3, 1, 2).shape
                variance = self._merge_dims(variance, max_precond_dim)
            variance = self._transport_diagonal_variance(variance, old_q, new_q)
            if merge_dims:
                if self._data_format == "channels_last" and len(original_shape) == 4:
                    variance = variance.reshape(permuted_shape).permute(0, 2, 3, 1)
                else:
                    variance = variance.reshape(original_shape)
            state["exp_avg_sq"] = variance
            state["Q"] = new_q

    def _orthogonal_matrix(self, mats):
        out = []
        for m in mats:
            if len(m) == 0:
                out.append([])
                continue
            orig_dtype = m.dtype
            orig_device = m.device
            mf = m.to(dtype=torch.float32)
            eye = torch.eye(mf.shape[0], device=mf.device, dtype=mf.dtype)
            try:
                _, Q = torch.linalg.eigh(mf + 1e-30 * eye)
            except RuntimeError:
                _, Q = torch.linalg.eigh(mf.to(torch.float64) + 1e-30 * eye.to(torch.float64))
                Q = Q.to(mf.dtype)
            Q = torch.flip(Q, [1])
            out.append(Q.to(device=orig_device, dtype=orig_dtype))
        return out

    @staticmethod
    def _transport_diagonal_variance(exp_avg_sq, old_q, new_q):
        """Transport a diagonal covariance approximation between SOAP bases.

        If ``z_old`` has diagonal covariance ``v``, and the coordinate change
        along a mode is ``old_q.T @ new_q``, the diagonal covariance in the
        new basis is ``(old_q.T @ new_q).square().T @ v``. Applying that
        contraction to every tensor mode preserves non-negative variance and
        handles rotations, not merely eigenvector permutations.
        """
        variance = exp_avg_sq
        for old, new in zip(old_q, new_q):
            if len(old) > 0:
                overlap_sq = (old.to(torch.float32).T @ new.to(torch.float32)).square()
                variance = torch.tensordot(
                    variance.to(overlap_sq.dtype), overlap_sq, dims=[[0], [0]]
                ).to(exp_avg_sq.dtype)
            else:
                order = list(range(1, variance.ndim)) + [0]
                variance = variance.permute(order)
        return variance

    def _orthogonal_matrix_QR(self, state, max_precond_dim: int, merge_dims: bool):
        precond_list = state["GG"]
        orth_list = state["Q"]
        matrix, orth_matrix = [], []
        for m, o in zip(precond_list, orth_list):
            if len(m) == 0:
                matrix.append([])
                orth_matrix.append([])
                continue
            matrix.append(m.to(torch.float32))
            orth_matrix.append(o.to(torch.float32))

        out = []
        for index, (m, o) in enumerate(zip(matrix, orth_matrix)):
            if len(m) == 0:
                out.append([])
                continue
            est_eig = torch.diag(o.T @ m @ o)
            sort_idx = torch.argsort(est_eig, descending=True)
            o = o[:, sort_idx]
            power_iter = m @ o
            Q, _ = torch.linalg.qr(power_iter)
            out.append(Q.to(state["GG"][index].dtype))
        return out
