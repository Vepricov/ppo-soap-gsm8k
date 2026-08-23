# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Causal, per-update Fisher-KL matched SOAP for the FSDP PPO actor.

The optimizer owns both live SOAP state and a shadow AdamW state. ``propose``
computes the two hypothetical updates from one already-clipped gradient without
changing parameters or optimizer state. Only ``commit`` advances either state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, cast

import torch
from torch import Tensor
from torch.optim import Optimizer


def partition_actor_parameters(
    named_parameters: Iterable[tuple[str, Tensor]],
) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    """Split actor parameters into SOAP matrices and AdamW auxiliaries."""
    soap, auxiliary = [], []
    for name, parameter in named_parameters:
        path = f".{name}."
        is_actor_matrix = parameter.ndim == 2 and (".self_attn." in path or ".mlp." in path)
        (soap if is_actor_matrix else auxiliary).append((name, parameter))
    return soap, auxiliary


def kfac_quadratic(direction: Tensor, activation_factor: Tensor, score_factor: Tensor) -> Tensor:
    """Return ``0.5 tr(S dW A dW^T)`` without a Kronecker materialization."""
    if direction.ndim != 2 or activation_factor.shape != (direction.shape[1], direction.shape[1]):
        raise ValueError("activation factor does not match matrix direction")
    if score_factor.shape != (direction.shape[0], direction.shape[0]):
        raise ValueError("score factor does not match matrix direction")
    work = direction.to(device=activation_factor.device, dtype=torch.float32)
    value = 0.5 * torch.trace(score_factor.to(work.device) @ work @ activation_factor @ work.T)
    if not torch.isfinite(value) or value < 0:
        raise FloatingPointError("factorized Fisher quadratic is invalid")
    return value


def _fixed_antithetic_probe(probabilities: Tensor, probe: int, seed: int) -> Tensor:
    """Deterministic Walsh probes transformed by categorical Fisher's square root."""
    indices = torch.arange(probabilities.shape[-1], device=probabilities.device, dtype=torch.int64)
    code = (probe // 2) + 1 + seed * 131
    bits = torch.bitwise_and(indices, code)
    parity = bits.clone()
    for shift in (32, 16, 8, 4, 2, 1):
        parity.bitwise_xor_(parity >> shift)
    signs = (1 - 2 * (parity & 1)).to(probabilities.dtype)
    if probe % 2:
        signs.neg_()
    root = probabilities.clamp_min(0).sqrt()
    projection = (root * signs).sum(dim=-1, keepdim=True)
    return root * (signs - root * projection)


@dataclass(frozen=True)
class CovarianceFactor:
    """Covariance as normalized sample/sketch rows; never a wide d-by-d tensor."""

    rows: Tensor
    dimension: int
    representation: str

    @classmethod
    def from_samples(cls, samples: Tensor, *, max_rank: int, dense_threshold: int, sketch_seed: int):
        if samples.ndim != 2 or samples.shape[0] == 0 or max_rank < 1:
            raise ValueError("covariance samples must be a non-empty matrix and max_rank positive")
        work = samples.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(work).all():
            raise FloatingPointError("score-Fisher covariance samples are non-finite")
        count, dimension = work.shape
        if count <= max_rank:
            rows, representation = work / math.sqrt(count), "empirical"
        else:
            generator = torch.Generator(device="cpu").manual_seed(int(sketch_seed))
            signs = torch.randint(0, 2, (max_rank, count), generator=generator, dtype=torch.float32)
            rows = signs.mul_(2).sub_(1).to(work.device) @ work / math.sqrt(max_rank * count)
            representation = "sketch"
        del dense_threshold  # allocation guard: this class never forms covariance matrices
        return cls(rows.contiguous(), dimension, representation)

    @property
    def rank(self) -> int:
        return self.rows.shape[0]

    @property
    def storage_bytes(self) -> int:
        return self.rows.numel() * self.rows.element_size()

    def state_dict(self):
        return {"rows": self.rows, "dimension": self.dimension, "representation": self.representation}

    @classmethod
    def load(cls, state):
        return cls(state["rows"].clone(), int(state["dimension"]), str(state["representation"]))


class _StreamingCovariance:
    """Online fixed-rank CPU sketch, consuming each sample batch once."""

    def __init__(self, dimension: int, count: int, *, max_rank: int, seed: int) -> None:
        if dimension < 1 or count < 1 or max_rank < 1:
            raise ValueError("streaming covariance dimensions must be positive")
        self.dimension, self.count = int(dimension), int(count)
        self.rank = min(int(max_rank), self.count)
        self._seen = 0
        self._exact = self.count <= max_rank
        self._rows = torch.zeros((self.rank, self.dimension), dtype=torch.float32)
        self._generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def add(self, samples: Tensor, *, scale: float = 1.0) -> None:
        work = samples.detach().to(device="cpu", dtype=torch.float32)
        if work.ndim != 2 or work.shape[1] != self.dimension:
            raise RuntimeError("streamed covariance sample dimension mismatch")
        stop = self._seen + work.shape[0]
        if stop > self.count or not torch.isfinite(work).all() or not math.isfinite(scale):
            raise RuntimeError("invalid streamed covariance samples")
        work = work.mul(float(scale))
        if self._exact:
            self._rows[self._seen:stop].copy_(work)
        else:
            signs = torch.randint(0, 2, (self.rank, work.shape[0]),
                                  generator=self._generator, dtype=torch.float32)
            self._rows.add_(signs.mul_(2).sub_(1) @ work)
        self._seen = stop

    def finalize(self) -> CovarianceFactor:
        if self._seen != self.count:
            raise RuntimeError(f"streamed covariance received {self._seen} of {self.count} rows")
        denominator = math.sqrt(self.count if self._exact else self.rank * self.count)
        return CovarianceFactor((self._rows / denominator).contiguous(), self.dimension,
                                "empirical" if self._exact else "sketch")


def _row_factor_quadratic(direction: Tensor, activation: CovarianceFactor,
                          score: CovarianceFactor) -> float:
    """Evaluate one row-factor block entirely on CPU in fp32."""
    # Directions normally live on the actor GPU while persistent factors are
    # CPU fp32. Copy only this matrix; never create fp64/full-factor GPU temps.
    work = direction.detach().to(device="cpu", dtype=torch.float32)
    product = score.rows @ work @ activation.rows.T
    value = 0.5 * product.square().sum()
    if not torch.isfinite(value) or value < 0:
        raise FloatingPointError("factorized score-Fisher quadratic is invalid")
    return float(value.item())


def factorized_fisher_quadratic(factors, directions: Mapping[str, Tensor]) -> float:
    """Return .5 sum tr(S dW A dW^T) using row factors only."""
    total = 0.0
    for name, direction in directions.items():
        if name not in factors:
            raise KeyError(f"missing score-Fisher factors for {name}")
        score, activation = factors[name]
        if tuple(direction.shape) != (score.dimension, activation.dimension):
            raise ValueError(f"direction shape does not match factors for {name}")
        total += _row_factor_quadratic(direction, activation, score)
    value = float(total)
    if not math.isfinite(value):
        raise FloatingPointError("factorized score-Fisher quadratic is non-finite")
    return value


def orthogonal_antithetic_probes(states: int, vocabulary: int, *, pairs: int, seed: int,
                                  dtype=torch.float32, device="cpu"):
    """Return deterministic QR-orthogonal logit probes and exact negatives."""
    if states < 1 or vocabulary < 2 or pairs < 1 or pairs > states * vocabulary:
        raise ValueError("invalid score-Fisher probe dimensions")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    gaussian = torch.randn(states * vocabulary, pairs, generator=generator, dtype=torch.float64)
    orthogonal, _ = torch.linalg.qr(gaussian, mode="reduced")
    base = orthogonal.T.reshape(pairs, states, vocabulary).to(device=device, dtype=dtype)
    probes = torch.cat((base, -base), dim=0)
    identity = {
        "algorithm": "qr_gaussian_antithetic_v1", "seed": int(seed), "pairs": pairs,
        "states": states, "vocabulary": vocabulary,
        "sha256": hashlib.sha256(base.cpu().double().numpy().tobytes()).hexdigest(),
    }
    return probes, identity


class FactorizedScoreFisher:
    """Fixed policy-score K-FAC factors for SOAP-owned linear matrices."""

    semantics = "policy_score_fisher_kfac"

    def __init__(self, parameters, factors, prompt_identity, probe_identity):
        self._name_by_parameter = dict(parameters)
        self.factors = dict(factors)
        self.prompt_identity = dict(prompt_identity)
        self.probe_identity = dict(probe_identity)

    @classmethod
    def from_factors(cls, named_parameters, factors, *, prompt_identity, probe_seed, probe_pairs):
        named_parameters = list(named_parameters)
        probe_identity = {"algorithm": "qr_gaussian_antithetic_v1", "seed": int(probe_seed),
                          "pairs": int(probe_pairs),
                          "states": int(prompt_identity["occupied_response_states"])}
        return cls({parameter: name for name, parameter in named_parameters}, factors,
                   prompt_identity, probe_identity)

    def __call__(self, directions: Mapping[Tensor, Tensor]) -> float:
        try:
            named = {self._name_by_parameter[parameter]: direction for parameter, direction in directions.items()}
        except KeyError as error:
            raise RuntimeError("SOAP direction has no score-Fisher factors") from error
        return factorized_fisher_quadratic(self.factors, named)

    def state_dict(self):
        return {"version": 1, "semantics": self.semantics,
                "prompt_identity": copy.deepcopy(self.prompt_identity),
                "probe_identity": copy.deepcopy(self.probe_identity),
                "factors": {name: {"score": score.state_dict(), "activation": activation.state_dict()}
                            for name, (score, activation) in self.factors.items()}}

    def load_state_dict(self, state):
        if state.get("semantics") != self.semantics:
            raise RuntimeError("checkpoint curvature is not policy-score Fisher")
        if state.get("prompt_identity") != self.prompt_identity or state.get("probe_identity") != self.probe_identity:
            raise RuntimeError("checkpoint Fisher prompt/probe identity does not match this run")
        restored = {name: (CovarianceFactor.load(pair["score"]), CovarianceFactor.load(pair["activation"]))
                    for name, pair in state["factors"].items()}
        if restored.keys() != self.factors.keys():
            raise RuntimeError("checkpoint score-Fisher factor ownership does not match this run")
        self.factors = restored


@contextmanager
def _without_nested_fsdp_runtime_hooks(module: torch.nn.Module) -> Iterator[None]:
    """Temporarily expose wrapped child modules beneath an unwrapped FSDP root."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    substitutions: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []

    def unwrap_children(parent: torch.nn.Module) -> None:
        for name, child in tuple(parent._modules.items()):
            if child is None:
                continue
            if isinstance(child, FSDP):
                substitutions.append((parent, name, child))
                unwrapped = child._fsdp_wrapped_module
                parent._modules[name] = unwrapped
                unwrap_children(unwrapped)
            else:
                unwrap_children(child)

    try:
        unwrap_children(module)
        yield
    finally:
        for parent, name, fsdp_child in reversed(substitutions):
            parent._modules[name] = fsdp_child


def matched_alpha(
    q_adamw: float,
    q_soap: float,
    *,
    minimum: float,
    maximum: float,
    clamp: bool,
) -> tuple[float, float]:
    """Calculate and validate ``sqrt(q_adamw / q_soap)``."""
    if not (0 < minimum <= maximum and math.isfinite(minimum) and math.isfinite(maximum)):
        raise ValueError("alpha bounds must be finite, positive, and ordered")
    if not (math.isfinite(q_adamw) and math.isfinite(q_soap) and q_adamw > 0 and q_soap > 0):
        raise FloatingPointError(f"Fisher quadratics must be finite and positive, got q_A={q_adamw}, q_S={q_soap}")
    raw = math.sqrt(q_adamw / q_soap)
    if not math.isfinite(raw) or raw <= 0:
        raise FloatingPointError(f"matched alpha must be finite and positive, got {raw}")
    if clamp:
        return min(max(raw, minimum), maximum), raw
    if not minimum <= raw <= maximum:
        raise FloatingPointError(f"matched alpha {raw} is outside [{minimum}, {maximum}] with clamping disabled")
    return raw, raw


@dataclass(frozen=True)
class UpdateProposal:
    generation: int
    adamw_directions: dict[Tensor, Tensor]
    soap_directions: dict[Tensor, Tensor]
    auxiliary_directions: dict[Tensor, Tensor]
    next_states: dict[Tensor, dict[str, Any]]


def _clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if isinstance(value, Tensor) else copy.deepcopy(value) for key, value in state.items()}


def _eigenvectors(accumulator: Tensor) -> Tensor:
    accumulator = accumulator.float()
    eye = torch.eye(accumulator.shape[0], device=accumulator.device, dtype=accumulator.dtype)
    _, vectors = torch.linalg.eigh(accumulator + 1e-30 * eye)
    return vectors.flip(1)


def _project(matrix: Tensor, left: Tensor | None, right: Tensor | None) -> Tensor:
    result = matrix
    if left is not None:
        result = left.T @ result
    if right is not None:
        result = result @ right
    return result


def _project_back(matrix: Tensor, left: Tensor | None, right: Tensor | None) -> Tensor:
    result = matrix
    if left is not None:
        result = left @ result
    if right is not None:
        result = result @ right.T
    return result


def symmetric_inverse_quarter(factor: Tensor, *, eps: float) -> Tensor:
    """Return a damped inverse fourth root of a finite symmetric PSD matrix."""
    if factor.ndim != 2 or factor.shape[0] != factor.shape[1]:
        raise ValueError("inverse fourth root requires a square matrix")
    if eps <= 0 or not math.isfinite(eps) or not torch.isfinite(factor).all():
        raise FloatingPointError("inverse fourth root requires finite input and positive epsilon")
    work = factor.float()
    work = 0.5 * (work + work.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(work)
    scale = max(1.0, float(eigenvalues.abs().max().item()))
    roundoff = 16.0 * torch.finfo(work.dtype).eps * scale
    if float(eigenvalues.min().item()) < -roundoff:
        raise FloatingPointError("temporal curvature factor is not positive semidefinite")
    eigenvalues = eigenvalues.clamp_min(0.0)
    damping = max(float(eps), float(eps) * float(eigenvalues.max().item()))
    inverse_quarter = (eigenvalues + damping).pow(-0.25)
    result = (eigenvectors * inverse_quarter.unsqueeze(0)) @ eigenvectors.T
    if not torch.isfinite(result).all():
        raise FloatingPointError("inverse fourth root is non-finite")
    return 0.5 * (result + result.T)


def polar_maxop_lmo(matrix: Tensor) -> Tensor:
    """Solve ``argmin_<Z: ||Z||op<=1> <matrix, Z>`` by a partial polar factor."""
    if matrix.ndim != 2:
        raise ValueError("MaxOp LMO requires a matrix")
    work = matrix.float()
    if not torch.isfinite(work).all():
        raise FloatingPointError("MaxOp LMO received a non-finite matrix")
    if not bool(torch.count_nonzero(work)):
        return torch.zeros_like(work)
    left, singular_values, right_transpose = torch.linalg.svd(work, full_matrices=False)
    threshold = max(work.shape) * torch.finfo(work.dtype).eps * singular_values.max()
    active = singular_values > threshold
    if not bool(active.any()):
        return torch.zeros_like(work)
    result = -(left[:, active] @ right_transpose[active, :])
    if not torch.isfinite(result).all():
        raise FloatingPointError("MaxOp LMO produced a non-finite polar factor")
    return result


class KLMatchedSOAP(Optimizer):
    """SOAP actor matrices with per-update factorized score-Fisher matching."""

    requires_named_parameters = True
    requires_kl_matched_fisher = True
    route_label = "causal_per_update_kl_matched_soap"

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, Tensor]],
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        soap_betas: tuple[float, float] = (0.95, 0.95),
        soap_eps: float = 1e-8,
        soap_shampoo_beta: float = -1.0,
        soap_precondition_frequency: int = 10,
        soap_max_precond_dim: int = 2048,
        auxiliary_eps: float | None = None,
        alpha_min: float = 0.05,
        alpha_max: float = 20.0,
        alpha_clamp: bool = True,
        fisher_dataset_path: str | None = None,
        fisher_prompt_indices: Sequence[int] = tuple(range(16)),
        fisher_micro_batch_size: int = 1,
        fisher_probe_count: int = 4,
        fisher_probe_seed: int = 0,
        fisher_expected_states: int = 57,
        fisher_factor_rank: int = 16,
        fisher_dense_threshold: int = 256,
        fisher_refresh_frequency: int = 4,
    ) -> None:
        named_parameters = list(named_parameters)
        soap_named, auxiliary_named = partition_actor_parameters(named_parameters)
        all_names = [name for name, _ in named_parameters]
        routed_names = [name for name, _ in soap_named + auxiliary_named]
        if not soap_named or not auxiliary_named:
            raise ValueError("KLMatchedSOAP requires non-empty SOAP matrix and AdamW auxiliary routes")
        if len(routed_names) != len(set(routed_names)) or set(routed_names) != set(all_names):
            raise ValueError("actor parameter ownership must be disjoint and exhaustive")
        fisher_prompt_indices = tuple(fisher_prompt_indices)
        if len(fisher_prompt_indices) != 16 or len(set(fisher_prompt_indices)) != 16:
            raise ValueError("exactly 16 distinct Fisher prompt indices are required")
        if min(soap_precondition_frequency, soap_max_precond_dim, fisher_micro_batch_size,
               fisher_expected_states, fisher_factor_rank, fisher_dense_threshold,
               fisher_refresh_frequency) < 1:
            raise ValueError("SOAP and Fisher dimensions/counts must be positive")
        if fisher_probe_count < 2 or fisher_probe_count % 2:
            raise ValueError("Fisher probes must contain deterministic antithetic pairs")
        matched_alpha(1.0, 1.0, minimum=alpha_min, maximum=alpha_max, clamp=alpha_clamp)

        defaults = {"lr": lr, "weight_decay": weight_decay}
        super().__init__(
            [
                {
                    "params": [parameter for _, parameter in soap_named],
                    "route": "soap_matrix",
                    "parameter_names": tuple(name for name, _ in soap_named),
                    "lr": lr,
                    "weight_decay": weight_decay,
                },
                {
                    "params": [parameter for _, parameter in auxiliary_named],
                    "route": "auxiliary_adamw",
                    "parameter_names": tuple(name for name, _ in auxiliary_named),
                    "lr": lr,
                    "weight_decay": weight_decay,
                },
            ],
            defaults,
        )
        self.parameter_routes = {
            "soap_matrix": tuple(name for name, _ in soap_named),
            "auxiliary_adamw": tuple(name for name, _ in auxiliary_named),
        }
        self.adamw_betas = tuple(betas)
        self.adamw_eps = float(eps)
        self.auxiliary_eps = float(auxiliary_eps if auxiliary_eps is not None else eps)
        self.soap_betas = tuple(soap_betas)
        self.soap_eps = float(soap_eps)
        self.shampoo_beta = float(soap_shampoo_beta if soap_shampoo_beta >= 0 else soap_betas[1])
        self.precondition_frequency = int(soap_precondition_frequency)
        self.max_precond_dim = int(soap_max_precond_dim)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.alpha_clamp = bool(alpha_clamp)
        self.fisher_dataset_path = fisher_dataset_path
        self.fisher_prompt_indices = tuple(int(index) for index in fisher_prompt_indices)
        self.fisher_micro_batch_size = int(fisher_micro_batch_size)
        self.fisher_probe_count = int(fisher_probe_count)
        self.fisher_probe_seed = int(fisher_probe_seed)
        self.fisher_expected_states = int(fisher_expected_states)
        self.fisher_factor_rank = int(fisher_factor_rank)
        self.fisher_dense_threshold = int(fisher_dense_threshold)
        self.fisher_refresh_frequency = int(fisher_refresh_frequency)
        self._fisher_evaluator: Callable[[Mapping[Tensor, Tensor]], float] | None = None
        self._prompt_identity: Mapping[str, Any] | None = None
        self._probe_identity: Mapping[str, Any] | None = None
        self._factor_generation = 0
        self._factor_count = 0
        self._update_generation = 0
        self.latest_telemetry: dict[str, float] = {}

    def bind_fisher_evaluator(
        self,
        evaluator: Callable[[Mapping[Tensor, Tensor]], float],
        prompt_identity: Mapping[str, Any],
    ) -> None:
        if getattr(evaluator, "semantics", None) != "policy_score_fisher_kfac":
            raise TypeError("KLMatchedSOAP requires policy-score K-FAC factors, not PPO curvature")
        if (prompt_identity.get("count") != len(self.fisher_prompt_indices)
                or tuple(prompt_identity.get("indices", ())) != self.fisher_prompt_indices
                or prompt_identity.get("occupied_response_states") != self.fisher_expected_states):
            raise RuntimeError("pinned Fisher set has wrong prompt or occupied-state count")
        if self._prompt_identity is not None and dict(self._prompt_identity) != dict(prompt_identity):
            raise RuntimeError("pinned Fisher prompt identity changed after optimizer binding")
        self._fisher_evaluator = evaluator
        self._prompt_identity = dict(prompt_identity)
        probe_identity = evaluator.probe_identity
        algorithm = probe_identity.get("algorithm")
        if algorithm not in ("qr_gaussian_antithetic_v1", "qr_gaussian_antithetic_v2") or probe_identity.get("seed") != self.fisher_probe_seed:
            raise RuntimeError("K-FAC probe identity does not match optimizer configuration")
        if algorithm == "qr_gaussian_antithetic_v2" and (
                probe_identity.get("micro_batch_size") != self.fisher_micro_batch_size
                or probe_identity.get("probe_count") != self.fisher_probe_count
                or probe_identity.get("pairs") != self.fisher_probe_count // 2
                or not probe_identity.get("partitions")):
            raise RuntimeError("K-FAC probe manifest does not match optimizer configuration")
        self._probe_identity = copy.deepcopy(probe_identity)

    def _adamw_proposal(self, parameter: Tensor, gradient: Tensor, state: Mapping[str, Any], eps: float):
        next_state = _clone_state(state)
        step = int(next_state.get("adamw_step", 0)) + 1
        first, second = self.adamw_betas
        exp_avg = next_state.get("adamw_exp_avg", torch.zeros_like(gradient)).mul(first).add(gradient, alpha=1 - first)
        exp_avg_sq = (
            next_state.get("adamw_exp_avg_sq", torch.zeros_like(gradient))
            .mul(second)
            .addcmul(gradient, gradient, value=1 - second)
        )
        normalized = exp_avg / (1 - first**step)
        denominator = (exp_avg_sq / (1 - second**step)).sqrt().add(eps)
        direction = -self._group_for(parameter)["lr"] * normalized / denominator
        group = self._group_for(parameter)
        direction = direction - group["lr"] * group["weight_decay"] * parameter
        next_state.update(adamw_step=step, adamw_exp_avg=exp_avg, adamw_exp_avg_sq=exp_avg_sq)
        return direction, next_state

    def _soap_proposal(self, parameter: Tensor, gradient: Tensor, state: Mapping[str, Any]):
        next_state = _clone_state(state)
        work_gradient = gradient.float()
        step = int(next_state.get("soap_step", 0)) + 1
        rows, columns = work_gradient.shape
        gg_left = next_state.get("soap_gg_left")
        gg_right = next_state.get("soap_gg_right")
        if rows <= self.max_precond_dim:
            if gg_left is None:
                gg_left = torch.zeros(rows, rows, device=gradient.device, dtype=torch.float32)
            gg_left = gg_left.mul(self.shampoo_beta).add(work_gradient @ work_gradient.T, alpha=1 - self.shampoo_beta)
        if columns <= self.max_precond_dim:
            if gg_right is None:
                gg_right = torch.zeros(columns, columns, device=gradient.device, dtype=torch.float32)
            gg_right = gg_right.mul(self.shampoo_beta).add(work_gradient.T @ work_gradient, alpha=1 - self.shampoo_beta)

        left = next_state.get("soap_q_left")
        right = next_state.get("soap_q_right")
        if left is None and gg_left is not None:
            left = _eigenvectors(gg_left)
        if right is None and gg_right is not None:
            right = _eigenvectors(gg_right)
        projected = _project(work_gradient, left, right)
        first, second = self.soap_betas
        exp_avg = (
            next_state.get("soap_exp_avg", torch.zeros_like(work_gradient))
            .mul(first)
            .add(work_gradient, alpha=1 - first)
        )
        exp_avg_sq = (
            next_state.get("soap_exp_avg_sq", torch.zeros_like(projected))
            .mul(second)
            .addcmul(projected, projected, value=1 - second)
        )
        normalized = _project(exp_avg, left, right) / (1 - first**step)
        denominator = (exp_avg_sq / (1 - second**step)).sqrt().add(self.soap_eps)
        preconditioned = _project_back(normalized / denominator, left, right)
        group = self._group_for(parameter)
        direction = -group["lr"] * preconditioned - group["lr"] * group["weight_decay"] * parameter.float()

        # The current direction uses the prior basis (or its deterministic first-step
        # initialization). A scheduled refresh is stored for the next update only.
        if step % self.precondition_frequency == 0:
            old_left, old_right = left, right
            if gg_left is not None:
                left = _eigenvectors(gg_left)
            if gg_right is not None:
                right = _eigenvectors(gg_right)
            # Preserve the diagonal second-moment approximation across a basis
            # change instead of reinterpreting old coordinates in the new basis.
            if old_left is not None and left is not None:
                overlap = (old_left.T @ left).square()
                exp_avg_sq = overlap.T @ exp_avg_sq
            if old_right is not None and right is not None:
                overlap = (old_right.T @ right).square()
                exp_avg_sq = exp_avg_sq @ overlap
        next_state.update(
            soap_step=step,
            soap_exp_avg=exp_avg,
            soap_exp_avg_sq=exp_avg_sq,
        )
        if gg_left is not None:
            next_state["soap_gg_left"] = gg_left
            next_state["soap_q_left"] = left
        if gg_right is not None:
            next_state["soap_gg_right"] = gg_right
            next_state["soap_q_right"] = right
        return direction.to(parameter.dtype), next_state

    def _group_for(self, parameter: Tensor) -> dict[str, Any]:
        for group in self.param_groups:
            if any(candidate is parameter for candidate in group["params"]):
                return group
        raise KeyError("parameter is not owned by optimizer")

    @torch.no_grad()
    def propose(self) -> UpdateProposal:
        adamw, soap, auxiliary, next_states = {}, {}, {}, {}
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("KLMatchedSOAP does not support sparse gradients")
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("KLMatchedSOAP proposal received a non-finite gradient")
                if group["route"] == "soap_matrix":
                    live_state = self.state.get(parameter, {})
                    adam_direction, candidate = self._adamw_proposal(parameter, gradient, live_state, self.adamw_eps)
                    soap_direction, candidate = self._soap_proposal(parameter, gradient, candidate)
                    # Both Fisher quadratics are evaluated by the CPU K-FAC
                    # scorer, so retaining an actor-sized pair of proposal
                    # tensors on GPU only inflates the first-update peak.
                    adamw[parameter] = adam_direction.detach().to(device="cpu")
                    soap[parameter] = soap_direction.detach().to(device="cpu")
                    next_states[parameter] = candidate
                else:
                    direction, candidate = self._adamw_proposal(
                        parameter, gradient, self.state.get(parameter, {}), self.auxiliary_eps
                    )
                    auxiliary[parameter] = direction.detach().to(device="cpu")
                    next_states[parameter] = candidate
        if not adamw or set(adamw) != set(soap):
            raise RuntimeError("SOAP-owned actor matrices must have both AdamW and SOAP proposals")
        return UpdateProposal(self._update_generation, adamw, soap, auxiliary, next_states)

    @torch.no_grad()
    def commit(self, proposal: UpdateProposal, alpha: float) -> None:
        if proposal.generation != self._update_generation:
            raise RuntimeError("stale or already-committed KLMatchedSOAP proposal")
        if not math.isfinite(alpha) or alpha <= 0:
            raise FloatingPointError("committed alpha must be finite and positive")
        for parameter, direction in proposal.soap_directions.items():
            parameter.add_(direction.to(device=parameter.device, dtype=parameter.dtype), alpha=alpha)
        for parameter, direction in proposal.auxiliary_directions.items():
            parameter.add_(direction.to(device=parameter.device, dtype=parameter.dtype))
        for parameter, candidate in proposal.next_states.items():
            self.state[parameter].clear()
            self.state[parameter].update(candidate)
        self._update_generation += 1

    def step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, closure: Callable[[], float] | None = None
    ) -> float | None:
        if closure is not None:
            raise ValueError("KLMatchedSOAP does not support optimizer closures")
        if self._fisher_evaluator is None or self._prompt_identity is None:
            raise RuntimeError("FSDPEngine did not bind the exact Fisher evaluator")
        proposal = self.propose()
        # Proposal construction has consumed the already-clipped gradients.  Do
        # not retain another full actor-sized tensor set while the functional
        # JVP creates its dual parameters.
        for group in self.param_groups:
            for parameter in group["params"]:
                parameter.grad = None
        refresh = getattr(self._fisher_evaluator, "refresh", None)
        should_refresh = (
            refresh is not None
            and (self._factor_generation == 0
                 or self._update_generation % self.fisher_refresh_frequency == 0)
        )
        generation = int(refresh()) if should_refresh else (
            self._factor_generation if refresh is not None else None
        )
        if should_refresh:
            self._factor_generation = generation
            self._factor_count = int(getattr(self._fisher_evaluator, "factor_count", 0))
            if self._factor_count != len(proposal.soap_directions):
                raise RuntimeError("K-FAC factor count does not match SOAP matrix count")
        with torch.enable_grad():
            if generation is None:
                q_adamw = float(self._fisher_evaluator(proposal.adamw_directions))
            else:
                q_adamw = float(self._fisher_evaluator(proposal.adamw_directions, generation=generation))
            # The AdamW shadow direction is needed only for its Fisher scalar;
            # commit applies the SOAP direction and advances both shadow states.
            proposal.adamw_directions.clear()
            if generation is None:
                q_soap = float(self._fisher_evaluator(proposal.soap_directions))
            else:
                q_soap = float(self._fisher_evaluator(proposal.soap_directions, generation=generation))
        alpha, raw_alpha = matched_alpha(
            q_adamw,
            q_soap,
            minimum=self.alpha_min,
            maximum=self.alpha_max,
            clamp=self.alpha_clamp,
        )
        self.commit(proposal, alpha)
        self.latest_telemetry = {
            "actor/kl_matched/q_adamw": q_adamw,
            "actor/kl_matched/q_soap": q_soap,
            "actor/kl_matched/alpha_raw": raw_alpha,
            "actor/kl_matched/alpha": alpha,
            "actor/kl_matched/alpha_clamped": float(alpha != raw_alpha),
            "actor/kl_matched/factor_refreshed": float(should_refresh),
            "actor/kl_matched/update": float(self._update_generation),
        }
        return None

    def state_dict(self):
        result = super().state_dict()
        result["kl_matched_soap"] = {
            "version": 2,
            "update_generation": self._update_generation,
            "prompt_identity": copy.deepcopy(self._prompt_identity),
            "latest_telemetry": dict(self.latest_telemetry),
            "configuration": self._checkpoint_configuration(),
            "probe_identity": copy.deepcopy(self._probe_identity),
            "factor_generation": self._factor_generation,
            "factor_count": self._factor_count,
            "fisher_state": self._fisher_evaluator.state_dict() if self._fisher_evaluator is not None else None,
        }
        return result

    def _checkpoint_configuration(self) -> dict[str, Any]:
        return {
            "adamw_betas": self.adamw_betas,
            "adamw_eps": self.adamw_eps,
            "auxiliary_eps": self.auxiliary_eps,
            "soap_betas": self.soap_betas,
            "soap_eps": self.soap_eps,
            "shampoo_beta": self.shampoo_beta,
            "precondition_frequency": self.precondition_frequency,
            "max_precond_dim": self.max_precond_dim,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "alpha_clamp": self.alpha_clamp,
            "fisher_prompt_indices": self.fisher_prompt_indices,
            "fisher_micro_batch_size": self.fisher_micro_batch_size,
            "fisher_probe_count": self.fisher_probe_count,
            "fisher_probe_seed": self.fisher_probe_seed,
            "fisher_expected_states": self.fisher_expected_states,
            "fisher_factor_rank": self.fisher_factor_rank,
            "fisher_dense_threshold": self.fisher_dense_threshold,
            "fisher_refresh_frequency": self.fisher_refresh_frequency,
        }

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        metadata = state_dict.pop("kl_matched_soap", None)
        if not isinstance(metadata, Mapping) or metadata.get("version") != 2:
            raise RuntimeError("KLMatchedSOAP checkpoint is missing causal proposal metadata")
        restored_identity = metadata.get("prompt_identity")
        if self._prompt_identity is None or restored_identity != self._prompt_identity:
            raise RuntimeError("checkpoint pinned Fisher prompt identity does not match this run")
        if metadata.get("configuration") != self._checkpoint_configuration():
            raise RuntimeError("checkpoint KLMatchedSOAP configuration does not match this run")
        if metadata.get("probe_identity") != self._probe_identity:
            raise RuntimeError("checkpoint K-FAC probe identity does not match this run")
        try:
            factor_count = int(metadata["factor_count"])
            factor_generation = int(metadata["factor_generation"])
            update_generation = int(metadata["update_generation"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("checkpoint K-FAC generation/count is invalid") from error
        expected_count = len(self.parameter_routes["soap_matrix"])
        if factor_generation < 0 or update_generation < 0 or (
                (factor_generation == 0 and factor_count != 0)
                or (factor_generation > 0 and factor_count != expected_count)):
            raise RuntimeError("checkpoint K-FAC factor count/generation mismatch")
        fisher_state = metadata.get("fisher_state")
        if self._fisher_evaluator is None or fisher_state is None:
            raise RuntimeError("checkpoint is missing score-Fisher factors")
        validate = getattr(self._fisher_evaluator, "validate_state_dict", None)
        if validate is not None:
            _restored, inner_generation, inner_count = validate(fisher_state)
            if inner_generation != factor_generation or inner_count != factor_count:
                raise RuntimeError("checkpoint outer/inner K-FAC generation/count mismatch")
        # All K-FAC metadata and payloads are validated before base optimizer
        # state (which mutates live tensors/dicts) is restored.
        super().load_state_dict(state_dict)
        self._fisher_evaluator.load_state_dict(fisher_state)
        self._update_generation = update_generation
        self._factor_generation = factor_generation
        self._factor_count = factor_count
        self.latest_telemetry = dict(metadata.get("latest_telemetry", {}))


class KLMatchedSOAPThenAdamW(KLMatchedSOAP):
    """Run causal KL-matched SOAP, then fresh-state AdamW.

    The pinned PPO setup performs exactly four optimizer updates per completed
    global step (train batch 256 / mini-batch 64, one PPO epoch).  Keeping the
    transition counter in the optimizer makes the phase boundary atomic with
    parameter updates and part of the ordinary FSDP optimizer checkpoint.
    """

    route_label = "causal_kl_matched_soap_then_fresh_adamw"

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, Tensor]],
        lr: float,
        weight_decay: float,
        *,
        switch_after_global_step: int = 100,
        optimizer_updates_per_global_step: int = 4,
        **kwargs,
    ) -> None:
        if switch_after_global_step < 1 or optimizer_updates_per_global_step < 1:
            raise ValueError("hybrid SOAP/AdamW boundary values must be positive")
        self.switch_after_global_step = int(switch_after_global_step)
        self.optimizer_updates_per_global_step = int(optimizer_updates_per_global_step)
        self.switch_after_optimizer_updates = (
            self.switch_after_global_step * self.optimizer_updates_per_global_step
        )
        super().__init__(named_parameters, lr, weight_decay, **kwargs)

    @property
    def hybrid_phase(self) -> str:
        if self._update_generation < self.switch_after_optimizer_updates:
            return "soap"
        return "adamw"

    @torch.no_grad()
    def _fresh_adamw_step(self) -> None:
        for group in self.param_groups:
            beta1, beta2 = self.adamw_betas
            eps = self.adamw_eps if group["route"] == "soap_matrix" else self.auxiliary_eps
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("hybrid AdamW does not support sparse gradients")
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("hybrid AdamW received a non-finite gradient")
                state = self.state[parameter]
                step = int(state.get("hybrid_adamw_step", 0)) + 1
                exp_avg = state.get("hybrid_adamw_exp_avg")
                exp_avg_sq = state.get("hybrid_adamw_exp_avg_sq")
                if exp_avg is None or exp_avg_sq is None:
                    exp_avg = torch.zeros_like(parameter)
                    exp_avg_sq = torch.zeros_like(parameter)
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                parameter.addcdiv_(exp_avg, denominator, value=-group["lr"] / bias_correction1)
                state.update(
                    hybrid_adamw_step=step,
                    hybrid_adamw_exp_avg=exp_avg,
                    hybrid_adamw_exp_avg_sq=exp_avg_sq,
                )

    def step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, closure: Callable[[], float] | None = None
    ) -> float | None:
        if self.hybrid_phase == "soap":
            result = super().step(closure)
            self.latest_telemetry["actor/hybrid/phase"] = 0.0
            self.latest_telemetry["actor/hybrid/completed_global_steps"] = (
                self._update_generation / self.optimizer_updates_per_global_step
            )
            return result
        if closure is not None:
            raise ValueError("hybrid AdamW does not support optimizer closures")
        self._fresh_adamw_step()
        self._update_generation += 1
        self.latest_telemetry = {
            "actor/hybrid/phase": 1.0,
            "actor/hybrid/completed_global_steps": (
                self._update_generation / self.optimizer_updates_per_global_step
            ),
            "actor/kl_matched/update": float(self._update_generation),
        }
        return None

    def _checkpoint_configuration(self) -> dict[str, Any]:
        configuration = super()._checkpoint_configuration()
        configuration.update(
            optimizer_route=self.route_label,
            switch_after_global_step=self.switch_after_global_step,
            optimizer_updates_per_global_step=self.optimizer_updates_per_global_step,
        )
        return configuration


class FactorizedSOAPPolarMaxOpLMO(KLMatchedSOAP):
    """Two-sided temporal SOAP whitening followed by the exact MaxOp LMO."""

    route_label = "factorized_soap_polar_maxop_lmo"

    def _soap_proposal(self, parameter: Tensor, gradient: Tensor, state: Mapping[str, Any]):
        next_state = _clone_state(state)
        work_gradient = gradient.float()
        rows, columns = work_gradient.shape
        if max(rows, columns) > self.max_precond_dim:
            raise RuntimeError(
                "FactorizedSOAPPolarMaxOpLMO requires both temporal factors; "
                f"matrix shape {tuple(work_gradient.shape)} exceeds soap_max_precond_dim={self.max_precond_dim}"
            )
        step = int(next_state.get("maxop_step", 0)) + 1
        left_factor = next_state.get(
            "maxop_gg_left", torch.zeros(rows, rows, device=gradient.device, dtype=torch.float32)
        )
        right_factor = next_state.get(
            "maxop_gg_right", torch.zeros(columns, columns, device=gradient.device, dtype=torch.float32)
        )
        left_factor = left_factor.mul(self.shampoo_beta).add(
            work_gradient @ work_gradient.T, alpha=1 - self.shampoo_beta
        )
        right_factor = right_factor.mul(self.shampoo_beta).add(
            work_gradient.T @ work_gradient, alpha=1 - self.shampoo_beta
        )
        bias_correction = 1 - self.shampoo_beta**step
        left = symmetric_inverse_quarter(left_factor / bias_correction, eps=self.soap_eps)
        right = symmetric_inverse_quarter(right_factor / bias_correction, eps=self.soap_eps)
        whitened = left @ work_gradient @ right
        core = polar_maxop_lmo(whitened)
        descent = left @ core @ right
        group = self._group_for(parameter)
        direction = group["lr"] * descent - group["lr"] * group["weight_decay"] * parameter.float()
        if not torch.isfinite(direction).all():
            raise FloatingPointError("factorized SOAP-polar proposal is non-finite")
        next_state.update(
            maxop_step=step,
            maxop_gg_left=left_factor,
            maxop_gg_right=right_factor,
        )
        return direction.to(parameter.dtype), next_state

    def _checkpoint_configuration(self) -> dict[str, Any]:
        configuration = super()._checkpoint_configuration()
        configuration["optimizer_route"] = self.route_label
        return configuration


def build_teacher_forced_gsm8k(
    dataset_path: str | Path,
    tokenizer: Any,
    prompt_indices: Sequence[int],
) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
    """Tokenize and fingerprint the pinned GSM8K teacher-forced prompt set."""
    import pandas as pd  # pyright: ignore[reportMissingImports]

    frame = pd.read_parquet(dataset_path)
    if min(prompt_indices) < 0 or max(prompt_indices) >= len(frame):
        raise RuntimeError("pinned Fisher prompt indices are outside the GSM8K parquet")
    eos = tokenizer.eos_token_id
    if eos is None:
        raise RuntimeError("Fisher tokenizer must define eos_token_id")
    examples = []
    for index in prompt_indices:
        row = frame.iloc[index].to_dict()
        prompt = row.get("prompt")
        reward_model = row.get("reward_model")
        answer = reward_model.get("ground_truth") if isinstance(reward_model, Mapping) else None
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if prompt is None or not isinstance(answer, str) or not answer:
            raise RuntimeError(f"GSM8K row {index} lacks prompt or reward_model.ground_truth")
        prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False) + [eos]
        examples.append((list(prompt_ids) + answer_ids, len(prompt_ids)))
    maximum = max(len(ids) for ids, _ in examples)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    input_ids = torch.full((len(examples), maximum), pad, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    fisher_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, (ids, prompt_length) in enumerate(examples):
        length = len(ids)
        input_ids[row, :length] = torch.tensor(ids)
        attention_mask[row, :length] = 1
        fisher_mask[row, prompt_length - 1 : length - 1] = True
    digest_payload = {
        "indices": list(prompt_indices),
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "fisher_mask": fisher_mask.tolist(),
    }
    identity = {
        "policy": "fixed_gsm8k_teacher_forced_v1",
        "indices": list(prompt_indices),
        "count": len(prompt_indices),
        "occupied_response_states": int(fisher_mask.sum().item()),
        "sha256": hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode()).hexdigest(),
    }
    return input_ids, attention_mask, fisher_mask, identity


class FactorizedKFACFisher(FactorizedScoreFisher):
    """Collect policy-score K-FAC factors as bounded-rank normalized rows.

    The factors are deliberately computed from fixed logit-space score probes,
    never from the PPO loss gradients used by SOAP itself.
    """

    semantics = "policy_score_fisher_kfac"

    def __init__(self, module: torch.nn.Module, input_ids: Tensor, attention_mask: Tensor,
                 fisher_mask: Tensor, named_parameters: Mapping[str, Tensor],
                 micro_batch_size: int = 1, probe_count: int = 4, probe_seed: int = 0,
                 factor_rank: int = 16, dense_threshold: int = 256) -> None:
        if probe_count < 2 or probe_count % 2:
            raise ValueError("K-FAC probe count must contain antithetic pairs")
        if min(micro_batch_size, factor_rank, dense_threshold) < 1 or not named_parameters:
            raise ValueError("K-FAC dimensions and owned matrices must be positive")
        if fisher_mask.shape != input_ids.shape or attention_mask.shape != input_ids.shape:
            raise ValueError("teacher-forced tensors must have matching shapes")
        if torch.any(fisher_mask.bool() & ~attention_mask.bool()):
            raise ValueError("Fisher response states must be attention-mask active")
        if not torch.any(fisher_mask):
            raise ValueError("teacher-forced Fisher mask must contain response states")
        self.module, self.input_ids = module, input_ids.cpu()
        self.attention_mask, self.fisher_mask = attention_mask.cpu(), fisher_mask.cpu()
        self.named_parameters = dict(named_parameters)
        self.micro_batch_size, self.probe_count, self.probe_seed = int(micro_batch_size), int(probe_count), int(probe_seed)
        self.factor_rank, self.dense_threshold = int(factor_rank), int(dense_threshold)
        self.factor_generation = self.factor_count = 0
        self.factors: dict[Tensor, tuple[CovarianceFactor, CovarianceFactor]] = {}
        # Built before optimizer binding/checkpoint loading and tied to the
        # exact partitioned tensors that refresh() will use.
        self._probe_identity = self._build_probe_identity()

    def _vocabulary_size(self) -> int:
        root = self._unwrapped_module()
        vocabulary = getattr(getattr(root, "config", None), "vocab_size", None)
        if vocabulary is None and hasattr(root, "get_output_embeddings"):
            output = root.get_output_embeddings()
            vocabulary = getattr(output, "out_features", None) or getattr(output, "num_embeddings", None)
        if vocabulary is None:
            vocabulary = getattr(getattr(root, "lm_head", None), "out_features", None)
        if vocabulary is None or int(vocabulary) < 2:
            raise RuntimeError("K-FAC probe vocabulary cannot be resolved from the pinned model")
        return int(vocabulary)

    def _build_probe_identity(self) -> dict[str, Any]:
        vocabulary = self._vocabulary_size()
        partitions = []
        for start in range(0, self.input_ids.shape[0], self.micro_batch_size):
            stop = min(start + self.micro_batch_size, self.input_ids.shape[0])
            states = int(self.fisher_mask[start:stop].sum().item())
            if states == 0:
                continue
            seed = self.probe_seed + start
            _, identity = orthogonal_antithetic_probes(
                states, vocabulary, pairs=self.probe_count // 2, seed=seed)
            partitions.append({"start": start, "stop": stop, "states": states,
                               "seed": seed, "sha256": identity["sha256"]})
        return {
            "algorithm": "qr_gaussian_antithetic_v2",
            "micro_batch_size": self.micro_batch_size,
            "probe_count": self.probe_count,
            "pairs": self.probe_count // 2,
            "seed": self.probe_seed,
            "states": int(self.fisher_mask.sum().item()),
            "vocabulary": vocabulary,
            "partitions": partitions,
        }

    @property
    def probe_identity(self) -> dict[str, Any]:
        identity = getattr(self, "_probe_identity", None)
        if identity is None:
            identity = {
                "algorithm": "legacy_test_fixture",
                "seed": int(self.probe_seed),
                "probe_count": int(self.probe_count),
                "states": int(self.fisher_mask.sum().item()),
            }
        return copy.deepcopy(identity)

    @property
    def factor_storage_bytes(self) -> int:
        return sum(a.storage_bytes + score.storage_bytes for a, score in self.factors.values())

    def _unwrapped_module(self) -> torch.nn.Module:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            return self.module._fsdp_wrapped_module if isinstance(self.module, FSDP) else self.module
        except ImportError:
            return self.module

    def refresh(self) -> int:
        """Collect a complete generation while leaving model parameters/grads unchanged."""
        from contextlib import nullcontext
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            is_fsdp = isinstance(self.module, FSDP)
        except ImportError:
            is_fsdp, FSDP = False, None  # type: ignore[assignment,misc]
        if is_fsdp and torch.distributed.get_world_size() != 1:
            raise RuntimeError("factorized K-FAC requires one-rank legacy FSDP")
        summon = FSDP.summon_full_params(self.module, recurse=True, writeback=False) if is_fsdp else nullcontext()
        root = self._unwrapped_module(); was_training = root.training
        if any(p.grad is not None for p in self.named_parameters.values()):
            raise RuntimeError("K-FAC collection requires cleared parameter gradients")
        modules = dict(root.named_modules()); missing = set(self.named_parameters) - set(modules)
        if missing: raise RuntimeError(f"K-FAC module names do not resolve: {sorted(missing)}")
        captures: dict[str, tuple[Tensor, Tensor]] = {}; handles = []
        active_count = 0
        for start in range(0, self.input_ids.shape[0], self.micro_batch_size):
            stop = min(start + self.micro_batch_size, self.input_ids.shape[0])
            if torch.any(self.fisher_mask[start:stop]):
                active_count += int(self.attention_mask[start:stop].bool().sum().item())
        response_count = int(self.fisher_mask.sum().item())
        if active_count < 1 or response_count < 1:
            raise RuntimeError("pinned Fisher set contains no active or response states")
        activation_streams = {
            p: _StreamingCovariance(p.shape[1], active_count, max_rank=self.factor_rank,
                                    seed=self.probe_seed + 2 * offset)
            for offset, p in enumerate(self.named_parameters.values())
        }
        score_streams = {
            p: _StreamingCovariance(p.shape[0], active_count * self.probe_count,
                                    max_rank=self.factor_rank,
                                    seed=self.probe_seed + 2 * offset + 1)
            for offset, p in enumerate(self.named_parameters.values())
        }
        # G must remain averaged by response states even though causal score
        # rows now include every attention-active prompt/response position.
        score_scale = math.sqrt(active_count / response_count)
        def make_hook(name):
            def hook(_module, args, output):
                if not args or not isinstance(args[0], Tensor) or not isinstance(output, Tensor):
                    raise RuntimeError(f"K-FAC module {name} did not expose tensors")
                captures[name] = (args[0], output)
            return hook
        for name in self.named_parameters: handles.append(modules[name].register_forward_hook(make_hook(name)))
        device = next(root.parameters()).device
        context = _without_nested_fsdp_runtime_hooks(root) if is_fsdp else nullcontext()
        try:
            root.eval()
            with summon, context, torch.enable_grad():
                for start in range(0, self.input_ids.shape[0], self.micro_batch_size):
                    stop = start + self.micro_batch_size
                    ids = self.input_ids[start:stop].to(device); attention = self.attention_mask[start:stop].to(device)
                    mask = self.fisher_mask[start:stop].to(device)
                    active = self.attention_mask[start:stop].to(device).bool(); captures.clear()
                    output = root(input_ids=ids, attention_mask=attention, use_cache=False)
                    logits = output.logits if hasattr(output, "logits") else output
                    if set(captures) != set(self.named_parameters):
                        raise RuntimeError("K-FAC capture count does not match SOAP matrices")
                    states = int(mask.sum());
                    if not states: continue
                    outputs = tuple(captures[n][1] for n in self.named_parameters)
                    for name, parameter in self.named_parameters.items():
                        activation_streams[parameter].add(captures[name][0].detach()[active])
                    probes, _identity = orthogonal_antithetic_probes(
                        states, logits.shape[-1], pairs=self.probe_count // 2,
                        seed=self.probe_seed + start, device=device, dtype=torch.float32)
                    probes = probes * math.sqrt(states * logits.shape[-1])
                    probabilities = logits.detach().float()[mask].softmax(-1); rootp = probabilities.sqrt()
                    positive_probes = probes[: len(probes) // 2]
                    negative_rows: list[list[Tensor]] = []
                    for index, probe in enumerate(positive_probes):
                        grad_selected = rootp * probe - probabilities * (rootp * probe).sum(-1, keepdim=True)
                        grad_logits = torch.zeros_like(logits); grad_logits[mask] = grad_selected.to(logits.dtype)
                        gradients = torch.autograd.grad(
                            logits, outputs, grad_outputs=grad_logits,
                            retain_graph=index + 1 < len(positive_probes),
                        )
                        rows_for_probe = []
                        for (_name, parameter), gradient in zip(
                            self.named_parameters.items(), gradients, strict=True
                        ):
                            rows = gradient.detach()[active].to(device="cpu", dtype=torch.float32)
                            score_streams[parameter].add(rows, scale=score_scale)
                            rows_for_probe.append(rows)
                        negative_rows.append(rows_for_probe)
                    # VJP is linear: VJP(-probe) == -VJP(probe). Preserve the
                    # exact stream order [g0, g1, -g0, -g1] without repeating
                    # full reverse traversals or GPU-to-CPU transfers.
                    for rows_for_probe in negative_rows:
                        for parameter, rows in zip(
                            self.named_parameters.values(), rows_for_probe, strict=True
                        ):
                            score_streams[parameter].add(-rows, scale=score_scale)
        finally:
            for handle in handles: handle.remove()
            root.train(was_training)
        factors = {}
        for parameter in self.named_parameters.values():
            factors[parameter] = (activation_streams[parameter].finalize(),
                                  score_streams[parameter].finalize())
        if any(p.grad is not None for p in self.named_parameters.values()):
            raise RuntimeError("K-FAC collection contaminated PPO gradients")
        self.factors = factors; self.factor_count = len(factors); self.factor_generation += 1
        return self.factor_generation

    def __call__(self, directions: Mapping[Tensor, Tensor], *, generation: int | None = None) -> float:
        expected = self.factor_generation if generation is None else generation
        if expected != self.factor_generation or expected < 1:
            raise RuntimeError("K-FAC factor generation is missing or stale")
        if len(directions) != self.factor_count or set(directions) != set(self.factors):
            raise RuntimeError("K-FAC direction count/ownership does not match factors")
        total = 0.0
        for parameter, direction in directions.items():
            activation, score = self.factors[parameter]
            if isinstance(activation, Tensor):  # compatibility for dense toy/reference factors
                total += float(kfac_quadratic(direction, activation, score).item())
            else:
                total += _row_factor_quadratic(direction, activation, score)
        if not math.isfinite(total) or total <= 0: raise FloatingPointError("K-FAC quadratic must be finite and positive")
        return float(total)

    def state_dict(self) -> dict[str, Any]:
        by_name = {}
        for name, parameter in self.named_parameters.items():
            if parameter in self.factors:
                a, g = self.factors[parameter]
                by_name[name] = {"activation": a.state_dict(), "score": g.state_dict()}
        return {"version": 2, "probe_identity": self.probe_identity,
                "factor_generation": self.factor_generation,
                "factor_count": self.factor_count, "factors": by_name}

    def validate_state_dict(self, state: Mapping[str, Any]):
        if state.get("version") != 2 or state.get("probe_identity") != self.probe_identity:
            raise RuntimeError("score-Fisher checkpoint probe definition mismatch")
        try:
            generation = int(state["factor_generation"])
            factor_count = int(state["factor_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("score-Fisher checkpoint generation/count is invalid") from error
        values = state.get("factors")
        if generation == 0 or factor_count == 0:
            if generation != 0 or factor_count != 0 or not isinstance(values, Mapping) or values:
                raise RuntimeError("score-Fisher initial checkpoint state is inconsistent")
            return {}, generation, factor_count
        if not isinstance(values, Mapping) or set(values) != set(self.named_parameters):
            raise RuntimeError("score-Fisher checkpoint factor ownership mismatch")
        if generation < 1 or factor_count != len(values) or factor_count != len(self.named_parameters):
            raise RuntimeError("score-Fisher checkpoint factor generation/count mismatch")

        restored = {}
        for name, parameter in self.named_parameters.items():
            pair = values[name]
            if not isinstance(pair, Mapping) or set(pair) != {"activation", "score"}:
                raise RuntimeError("score-Fisher checkpoint factor payload mismatch")
            loaded = []
            for kind, dimension in (("activation", parameter.shape[1]), ("score", parameter.shape[0])):
                payload = pair[kind]
                if not isinstance(payload, Mapping) or set(payload) != {"rows", "dimension", "representation"}:
                    raise RuntimeError("score-Fisher checkpoint factor payload mismatch")
                rows = payload["rows"]
                if (not isinstance(rows, Tensor) or rows.ndim != 2 or rows.device.type != "cpu"
                        or rows.dtype != torch.float32 or not torch.isfinite(rows).all()):
                    raise RuntimeError("score-Fisher checkpoint factor rows must be finite CPU fp32")
                if (int(payload["dimension"]) != dimension or rows.shape[1] != dimension
                        or not 1 <= rows.shape[0] <= self.factor_rank
                        or payload["representation"] not in ("empirical", "sketch")):
                    raise RuntimeError("score-Fisher checkpoint factor dimension/rank mismatch")
                loaded.append(CovarianceFactor(rows.clone().contiguous(), dimension,
                                               str(payload["representation"])))
            restored[parameter] = (loaded[0], loaded[1])
        return restored, generation, factor_count

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        restored, generation, factor_count = self.validate_state_dict(state)
        self.factors = restored
        self.factor_count = factor_count
        self.factor_generation = generation
