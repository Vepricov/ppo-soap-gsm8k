"""Focused CPU tests for causal per-update KL-matched SOAP."""

import copy
import importlib.util
import inspect
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODULE_PATH = Path(__file__).parents[2] / "verl" / "utils" / "kl_matched_soap.py"
SPEC = importlib.util.spec_from_file_location("kl_matched_soap_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
kl_matched_soap = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = kl_matched_soap
SPEC.loader.exec_module(kl_matched_soap)
KLMatchedSOAP = kl_matched_soap.KLMatchedSOAP
FactorizedKFACFisher = kl_matched_soap.FactorizedKFACFisher
FactorizedScoreFisher = kl_matched_soap.FactorizedScoreFisher
CovarianceFactor = kl_matched_soap.CovarianceFactor
orthogonal_antithetic_probes = kl_matched_soap.orthogonal_antithetic_probes
kfac_quadratic = kl_matched_soap.kfac_quadratic
matched_alpha = kl_matched_soap.matched_alpha
partition_actor_parameters = kl_matched_soap.partition_actor_parameters


def _named_parameters():
    return [
        ("model.layers.0.self_attn.q_proj.weight", torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))),
        ("model.layers.0.mlp.down_proj.weight", torch.nn.Parameter(torch.tensor([[2.0, 0.0], [-1.0, 1.0]]))),
        ("model.layers.0.input_layernorm.weight", torch.nn.Parameter(torch.tensor([1.0, 1.0]))),
        ("model.embed_tokens.weight", torch.nn.Parameter(torch.tensor([[0.2, 0.3], [0.4, 0.5]]))),
    ]


def _optimizer(named=None, **kwargs):
    named = _named_parameters() if named is None else named
    fisher_prompt_indices = kwargs.pop("fisher_prompt_indices", range(16))
    optimizer = KLMatchedSOAP(
        named,
        lr=0.01,
        weight_decay=0.1,
        eps=1e-5,
        soap_precondition_frequency=2,
        soap_max_precond_dim=8,
        fisher_prompt_indices=fisher_prompt_indices,
        **kwargs,
    )
    class TestFactors(FactorizedScoreFisher):
        probe_identity = {"algorithm": "qr_gaussian_antithetic_v1", "seed": 0, "pairs": 2, "states": 57}
        def __call__(self, directions):
            return 0.5 * sum(float(direction.double().square().sum()) for direction in directions.values())
        def state_dict(self):
            return {"test": True}
        def load_state_dict(self, state):
            if state != {"test": True}: raise RuntimeError("test Fisher mismatch")
    evaluator = TestFactors.__new__(TestFactors)
    optimizer.bind_fisher_evaluator(evaluator, {
        "policy": "test", "indices": list(range(16)), "count": 16,
        "occupied_response_states": 57, "sha256": "fixed",
    })
    return optimizer, named


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right, strict=True):
            _assert_nested_equal(lhs, rhs)
    else:
        assert left == right


def test_factorized_quadratic_matches_explicit_kronecker_without_materializing_it():
    activation = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float32)
    score = torch.tensor([[1.5, -0.25], [-0.25, 0.75]], dtype=torch.float32)
    direction = torch.tensor([[0.2, -0.4], [0.7, 0.3]], dtype=torch.float32)
    expected = 0.5 * torch.trace(score @ direction @ activation @ direction.T)
    assert kfac_quadratic(direction, activation, score).item() == pytest.approx(expected.item())


def test_real_hook_collector_builds_cpu_fp32_factors_and_leaves_model_pristine():
    class ToyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Linear(3, 4, bias=False, dtype=torch.float64)
            self.lm_head = torch.nn.Linear(4, 5, bias=False, dtype=torch.float64)

        def forward(self, input_ids, attention_mask, use_cache=False):
            del attention_mask, use_cache
            hidden = torch.nn.functional.one_hot(input_ids, num_classes=3).to(torch.float64)
            return SimpleNamespace(logits=self.lm_head(torch.tanh(self.self_attn(hidden))))

    model = ToyCausalLM().train()
    inputs = torch.tensor([[0, 1, 2], [2, 1, 0]])
    mask = torch.ones_like(inputs, dtype=torch.bool)
    parameter = model.self_attn.weight
    evaluator = FactorizedKFACFisher(
        model, inputs, torch.ones_like(inputs), mask,
        {"self_attn": parameter}, probe_count=4, probe_seed=17, micro_batch_size=1,
    )
    parameter_before = parameter.detach().clone()
    generation = evaluator.refresh()
    assert generation == 1
    assert model.training
    assert parameter.grad is None
    assert torch.equal(parameter, parameter_before)
    assert evaluator.factor_count == 1
    activation, score = evaluator.factors[parameter]
    assert activation.rows.device.type == score.rows.device.type == "cpu"
    assert activation.rows.dtype == score.rows.dtype == torch.float32
    assert activation.dimension == 3
    assert score.dimension == 4
    assert evaluator({parameter: torch.ones_like(parameter)}, generation=1) > 0


def test_factor_evaluator_fails_closed_for_missing_direction_or_stale_generation():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    evaluator = FactorizedKFACFisher.__new__(FactorizedKFACFisher)
    evaluator.factors = {parameter: (torch.eye(2), torch.eye(2))}
    evaluator.factor_generation = 3
    evaluator.factor_count = 1
    with pytest.raises(RuntimeError, match="factor generation"):
        evaluator({parameter: torch.ones_like(parameter)}, generation=2)
    with pytest.raises(RuntimeError, match="direction count"):
        evaluator({}, generation=3)


def test_proposal_does_not_mutate_parameters_gradients_or_live_optimizer_state():
    optimizer, named = _optimizer()
    for index, (_, parameter) in enumerate(named):
        parameter.grad = torch.full_like(parameter, 0.2 + index * 0.1)
    parameters_before = [parameter.detach().clone() for _, parameter in named]
    assert all(parameter.grad is not None for _, parameter in named)
    gradients_before = [parameter.grad.clone() for _, parameter in named if parameter.grad is not None]
    state_before = copy.deepcopy(optimizer.state_dict())

    proposal = optimizer.propose()

    assert proposal.generation == 0
    assert all(direction.device.type == "cpu" for direction in proposal.adamw_directions.values())
    assert all(direction.device.type == "cpu" for direction in proposal.soap_directions.values())
    assert all(direction.device.type == "cpu" for direction in proposal.auxiliary_directions.values())
    for (_, parameter), value, gradient in zip(named, parameters_before, gradients_before, strict=True):
        assert torch.equal(parameter, value)
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, gradient)
    _assert_nested_equal(optimizer.state_dict(), state_before)


def test_step_refreshes_one_factor_generation_and_reuses_it_for_both_directions():
    optimizer, named = _optimizer()

    class RecordingFactors(FactorizedScoreFisher):
        probe_identity = {"algorithm": "qr_gaussian_antithetic_v1", "pairs": 2, "states": 57, "seed": 0}
        factor_generation = 0
        factor_count = 2

        def __init__(self):
            self.calls = []

        def refresh(self):
            self.factor_generation += 1
            return self.factor_generation

        def __call__(self, directions, *, generation):
            self.calls.append((generation, tuple(directions)))
            return sum(float(direction.float().square().sum()) for direction in directions.values())

    factors = RecordingFactors.__new__(RecordingFactors)
    factors.__init__()
    optimizer.bind_fisher_evaluator(
        factors, {"policy": "test", "indices": list(range(16)), "count": 16,
                  "occupied_response_states": 57, "sha256": "fixed"}
    )
    for _, parameter in named:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert factors.factor_generation == 1
    assert [generation for generation, _ in factors.calls] == [1, 1]
    assert len(factors.calls[0][1]) == len(factors.calls[1][1]) == 2


def test_commit_advances_both_proposal_states_exactly_once_and_rejects_reuse():
    optimizer, named = _optimizer()
    for _, parameter in named:
        parameter.grad = torch.full_like(parameter, 0.25)
    proposal = optimizer.propose()
    soap_parameter = named[0][1]
    expected = soap_parameter.detach() + 1.5 * proposal.soap_directions[soap_parameter]
    optimizer.commit(proposal, alpha=1.5)
    assert torch.allclose(soap_parameter, expected)
    assert optimizer.state[soap_parameter]["soap_step"] == 1
    assert optimizer.state[soap_parameter]["adamw_step"] == 1
    with pytest.raises(RuntimeError, match="stale|already-committed"):
        optimizer.commit(proposal, alpha=1.5)
    assert optimizer.state[soap_parameter]["soap_step"] == 1
    assert optimizer.state[soap_parameter]["adamw_step"] == 1


def test_factor_refresh_is_reused_across_four_inner_ppo_updates():
    optimizer, named = _optimizer(fisher_refresh_frequency=4)

    class RecordingFactors(FactorizedScoreFisher):
        probe_identity = {"algorithm": "qr_gaussian_antithetic_v1", "pairs": 2, "states": 57, "seed": 0}
        factor_generation = 0
        factor_count = 2

        def __init__(self):
            self.refresh_calls = 0

        def refresh(self):
            self.refresh_calls += 1
            self.factor_generation += 1
            return self.factor_generation

        def __call__(self, directions, *, generation):
            assert generation == self.factor_generation
            return 0.5 * sum(float(direction.double().square().sum()) for direction in directions.values())

    factors = RecordingFactors()
    optimizer.bind_fisher_evaluator(
        factors, {"policy": "test", "indices": list(range(16)), "count": 16,
                  "occupied_response_states": 57, "sha256": "fixed"}
    )
    for update in range(5):
        for _, parameter in named:
            parameter.grad = torch.full_like(parameter, 0.1 + update * 0.01)
        optimizer.step()
    assert factors.refresh_calls == 2
    assert factors.factor_generation == 2
    assert optimizer._update_generation == 5


def test_fisher_requires_exactly_sixteen_distinct_prompts():
    with pytest.raises(ValueError, match="exactly 16 distinct"):
        _optimizer(fisher_prompt_indices=[0, 4, 8, 12])
    with pytest.raises(ValueError, match="exactly 16 distinct"):
        _optimizer(fisher_prompt_indices=list(range(15)) + [0])


def test_step_checkpoint_restore_preserves_shadow_soap_alpha_and_prompt_identity():
    optimizer, named = _optimizer()
    for index, (_, parameter) in enumerate(named):
        parameter.grad = torch.full_like(parameter, 0.1 + index * 0.05)
    optimizer.step()
    assert all(parameter.grad is None for _, parameter in named)
    checkpoint = copy.deepcopy(optimizer.state_dict())

    restored_named = [(name, torch.nn.Parameter(parameter.detach().clone())) for name, parameter in named]
    restored, _ = _optimizer(restored_named)
    restored.load_state_dict(checkpoint)

    assert restored._update_generation == 1
    assert restored.latest_telemetry == optimizer.latest_telemetry
    assert restored.state_dict()["kl_matched_soap"]["prompt_identity"]["sha256"] == "fixed"
    for source, target in zip(optimizer.state.values(), restored.state.values(), strict=True):
        _assert_nested_equal(source, target)

    for (_, source), (_, target) in zip(named, restored_named, strict=True):
        source.grad = torch.full_like(source, 0.37)
        target.grad = source.grad.clone()
    source_proposal = optimizer.propose()
    restored_proposal = restored.propose()
    for source, target in zip(
        source_proposal.soap_directions.values(), restored_proposal.soap_directions.values(), strict=True
    ):
        assert torch.equal(source, target)
    for source, target in zip(
        source_proposal.adamw_directions.values(), restored_proposal.adamw_directions.values(), strict=True
    ):
        assert torch.equal(source, target)


def test_actor_auxiliary_update_is_exact_adamw_and_is_not_alpha_scaled():
    optimizer, named = _optimizer()
    auxiliary = named[2][1]
    reference = torch.nn.Parameter(auxiliary.detach().clone())
    reference_optimizer = torch.optim.AdamW([reference], lr=0.01, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-5)
    for _, parameter in named:
        parameter.grad = torch.full_like(parameter, 0.25)
    assert auxiliary.grad is not None
    reference.grad = auxiliary.grad.clone()
    proposal = optimizer.propose()
    optimizer.commit(proposal, alpha=7.0)
    reference_optimizer.step()
    assert torch.allclose(auxiliary, reference, rtol=1e-6, atol=1e-7)


def test_restore_fails_closed_when_pinned_prompt_identity_changes():
    optimizer, _ = _optimizer()
    checkpoint = optimizer.state_dict()
    restored, _ = _optimizer()
    restored._prompt_identity = {"policy": "test", "indices": list(range(16)), "sha256": "different"}
    with pytest.raises(RuntimeError, match="prompt identity"):
        restored.load_state_dict(checkpoint)


def test_actor_ownership_is_disjoint_and_exhaustive():
    named = _named_parameters()
    soap, auxiliary = partition_actor_parameters(named)
    soap_names = {name for name, _ in soap}
    auxiliary_names = {name for name, _ in auxiliary}
    assert soap_names.isdisjoint(auxiliary_names)
    assert soap_names | auxiliary_names == {name for name, _ in named}
    assert soap_names == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    }


@pytest.mark.parametrize("q_adamw,q_soap", [(0.0, 1.0), (1.0, 0.0), (math.inf, 1.0), (1.0, math.nan)])
def test_alpha_fails_closed_for_zero_or_nonfinite_quadratics(q_adamw, q_soap):
    with pytest.raises(FloatingPointError):
        matched_alpha(q_adamw, q_soap, minimum=0.5, maximum=2.0, clamp=True)


def test_alpha_clamps_explicitly_or_fails_closed_when_disabled():
    assert matched_alpha(100.0, 1.0, minimum=0.5, maximum=2.0, clamp=True) == (2.0, 10.0)
    with pytest.raises(FloatingPointError, match="clamping disabled"):
        matched_alpha(100.0, 1.0, minimum=0.5, maximum=2.0, clamp=False)


def test_orthogonal_antithetic_probes_are_deterministic_orthogonal_and_zero_mean():
    first, identity = orthogonal_antithetic_probes(3, 7, pairs=4, seed=19)
    second, second_identity = orthogonal_antithetic_probes(3, 7, pairs=4, seed=19)
    assert torch.equal(first, second)
    assert identity == second_identity
    assert torch.equal(first[:4], -first[4:])
    assert torch.equal(first.mean(0), torch.zeros_like(first[0]))
    gram = first[:4].flatten(1).double() @ first[:4].flatten(1).double().T
    assert torch.allclose(gram, torch.eye(4, dtype=torch.float64), atol=1e-6)


def test_wide_factor_is_bounded_rank_and_never_allocates_dense_covariance(monkeypatch):
    original_zeros = torch.zeros
    def guarded_zeros(*shape, **kwargs):
        normalized = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        assert tuple(normalized) != (4864, 4864), "dense wide factor allocated"
        return original_zeros(*shape, **kwargs)
    monkeypatch.setattr(torch, "zeros", guarded_zeros)
    factor = CovarianceFactor.from_samples(torch.randn(57, 4864), max_rank=16,
                                           dense_threshold=256, sketch_seed=3)
    assert factor.rows.shape == (16, 4864)
    assert factor.representation == "sketch"
    assert factor.storage_bytes < 4864 * 4864 * 4


def test_score_fisher_state_roundtrip_is_exact_and_rejects_ppo_curvature_semantics():
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    evaluator = FactorizedKFACFisher.__new__(FactorizedKFACFisher)
    evaluator.named_parameters = {"layer": parameter}
    evaluator.probe_seed, evaluator.probe_count, evaluator.factor_rank = 7, 4, 4
    evaluator.fisher_mask = torch.ones(57, dtype=torch.bool)
    evaluator.factor_generation, evaluator.factor_count = 2, 1
    a = CovarianceFactor.from_samples(torch.randn(5, 3), max_rank=4, dense_threshold=2, sketch_seed=1)
    score = CovarianceFactor.from_samples(torch.randn(5, 2), max_rank=4, dense_threshold=2, sketch_seed=2)
    evaluator.factors = {parameter: (a, score)}
    checkpoint = copy.deepcopy(evaluator.state_dict())
    restored = FactorizedKFACFisher.__new__(FactorizedKFACFisher)
    restored.named_parameters = {"layer": parameter}
    restored.probe_seed, restored.probe_count, restored.factor_rank = 7, 4, 4
    restored.fisher_mask = torch.ones(57, dtype=torch.bool)
    restored.factor_generation, restored.factor_count, restored.factors = 0, 0, {}
    restored.load_state_dict(checkpoint)
    _assert_nested_equal(restored.state_dict(), checkpoint)
    optimizer, _ = _optimizer()
    ppo_curvature = SimpleNamespace(semantics="soap_ppo_gradient_curvature")
    with pytest.raises(TypeError, match="policy-score"):
        optimizer.bind_fisher_evaluator(ppo_curvature,
            {"count": 16, "occupied_response_states": 57})


def test_score_fisher_initial_checkpoint_roundtrip_accepts_empty_generation_zero():
    evaluator, _ = _causal_mixing_evaluator(2)
    evaluator.factor_generation = evaluator.factor_count = 0
    evaluator.factors = {}
    checkpoint = copy.deepcopy(evaluator.state_dict())

    restored, _ = _causal_mixing_evaluator(2)
    restored.load_state_dict(checkpoint)

    _assert_nested_equal(restored.state_dict(), checkpoint)


def _causal_mixing_evaluator(sequence_length, probe_count=2):
    class CausalMixingLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Linear(2, 2, bias=False)
            self.lm_head = torch.nn.Linear(2, 3, bias=False)
            self.config = SimpleNamespace(vocab_size=3)

        def forward(self, input_ids, attention_mask, use_cache=False):
            del attention_mask, use_cache
            hidden = torch.nn.functional.one_hot(input_ids, num_classes=2).float()
            mixed = self.self_attn(hidden)
            # Only the first prompt position causally controls the response logit.
            response = self.lm_head(mixed[:, 0])
            logits = torch.zeros((*input_ids.shape, 3), device=input_ids.device)
            logits = logits.index_copy(1, torch.tensor([sequence_length - 1], device=input_ids.device), response[:, None])
            return SimpleNamespace(logits=logits)

    model = CausalMixingLM()
    with torch.no_grad():
        model.self_attn.weight.copy_(torch.tensor([[1.0, 0.2], [-0.3, 0.7]]))
        model.lm_head.weight.copy_(torch.tensor([[0.5, -0.2], [-0.4, 0.8], [0.3, 0.1]]))
    ids = torch.zeros((1, sequence_length), dtype=torch.long)
    attention = torch.ones_like(ids)
    response_mask = torch.zeros_like(ids, dtype=torch.bool)
    response_mask[:, -1] = True
    evaluator = FactorizedKFACFisher(
        model, ids, attention, response_mask, {"self_attn": model.self_attn.weight},
        micro_batch_size=1, probe_count=probe_count, probe_seed=11, factor_rank=16,
    )
    evaluator.refresh()
    return evaluator, model.self_attn.weight


def test_refresh_reuses_exact_antithetic_vjps(monkeypatch):
    calls = 0
    original_grad = torch.autograd.grad

    def counted_grad(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    evaluator, parameter = _causal_mixing_evaluator(2, probe_count=4)
    score_rows = evaluator.factors[parameter][1].rows

    assert calls == 2
    # Two active token rows per probe, ordered g0, g1, -g0, -g1.
    assert torch.equal(score_rows[:2], -score_rows[4:6])
    assert torch.equal(score_rows[2:4], -score_rows[6:8])


def test_causal_prompt_positions_contribute_to_response_state_fisher_without_prompt_dilution():
    short, short_parameter = _causal_mixing_evaluator(2)
    long, long_parameter = _causal_mixing_evaluator(3)
    short_q = short({short_parameter: torch.ones_like(short_parameter)}, generation=1)
    long_q = long({long_parameter: torch.ones_like(long_parameter)}, generation=1)
    assert short_q > 0
    assert long_q == pytest.approx(short_q, rel=1e-5, abs=1e-7)


def test_row_factor_quadratic_uses_cpu_fp32_without_double_temporaries(monkeypatch):
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    evaluator = FactorizedKFACFisher.__new__(FactorizedKFACFisher)
    evaluator.factors = {
        parameter: (
            CovarianceFactor(torch.ones(1, 3), 3, "empirical"),
            CovarianceFactor(torch.ones(1, 2), 2, "empirical"),
        )
    }
    evaluator.factor_generation = evaluator.factor_count = 1
    original_double = torch.Tensor.double

    def reject_double(tensor, *args, **kwargs):
        if tensor.ndim > 0:
            raise AssertionError("quadratic materialized an fp64 tensor")
        return original_double(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "double", reject_double)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    direction = torch.ones_like(parameter, device=device)
    # 0.5 * ||[1,1] @ ones(2,3) @ [1,1,1]^T||^2 = 18.
    assert evaluator({parameter: direction}, generation=1) == pytest.approx(18.0)


def test_probe_manifest_is_complete_stable_and_describes_actual_partitions():
    evaluator, _ = _causal_mixing_evaluator(3)
    identity = copy.deepcopy(evaluator.probe_identity)
    assert identity["algorithm"] == "qr_gaussian_antithetic_v2"
    assert identity["vocabulary"] == 3
    assert identity["micro_batch_size"] == 1
    assert identity["probe_count"] == 2
    assert identity["partitions"] == [{
        "start": 0, "stop": 1, "states": 1, "seed": 11,
        "sha256": identity["partitions"][0]["sha256"],
    }]
    assert len(identity["partitions"][0]["sha256"]) == 64
    evaluator.refresh()
    assert evaluator.probe_identity == identity


def test_factor_checkpoint_validation_is_fail_closed_and_nonmutating():
    evaluator, _ = _causal_mixing_evaluator(2)
    pristine = copy.deepcopy(evaluator.state_dict())
    corruptions = []
    missing = copy.deepcopy(pristine); missing["factors"] = {}; corruptions.append(missing)
    wrong_generation = copy.deepcopy(pristine); wrong_generation["factor_count"] = 0; corruptions.append(wrong_generation)
    wrong_dimension = copy.deepcopy(pristine); wrong_dimension["factors"]["self_attn"]["activation"]["dimension"] = 99; corruptions.append(wrong_dimension)
    wrong_rank = copy.deepcopy(pristine); wrong_rank["factors"]["self_attn"]["score"]["rows"] = torch.ones(17, 2); corruptions.append(wrong_rank)
    wrong_dtype = copy.deepcopy(pristine); wrong_dtype["factors"]["self_attn"]["score"]["rows"] = torch.ones(1, 2, dtype=torch.float64); corruptions.append(wrong_dtype)
    nonfinite = copy.deepcopy(pristine); nonfinite["factors"]["self_attn"]["score"]["rows"][0, 0] = torch.nan; corruptions.append(nonfinite)
    for state in corruptions:
        with pytest.raises(RuntimeError, match="checkpoint|factor"):
            evaluator.load_state_dict(state)
        _assert_nested_equal(evaluator.state_dict(), pristine)


def test_refresh_streams_factor_samples_without_full_layer_concatenation():
    source = inspect.getsource(FactorizedKFACFisher.refresh)
    assert "torch.cat(activation_rows" not in source
    assert "torch.cat(score_rows" not in source


def test_optimizer_rejects_outer_inner_factor_generation_mismatch_before_mutation():
    evaluator, parameter = _causal_mixing_evaluator(2)
    auxiliary = torch.nn.Parameter(torch.ones(2))
    optimizer = KLMatchedSOAP(
        [("self_attn.weight", parameter), ("input_layernorm.weight", auxiliary)],
        lr=0.01, weight_decay=0.0,
        soap_max_precond_dim=8, fisher_prompt_indices=range(16),
        fisher_micro_batch_size=1, fisher_probe_count=2, fisher_probe_seed=11,
        fisher_factor_rank=16,
    )
    optimizer.bind_fisher_evaluator(evaluator, {
        "policy": "test", "indices": list(range(16)), "count": 16,
        "occupied_response_states": 57, "sha256": "causal",
    })
    optimizer._factor_generation = evaluator.factor_generation
    optimizer._factor_count = evaluator.factor_count
    checkpoint = copy.deepcopy(optimizer.state_dict())
    before = copy.deepcopy(optimizer.state_dict())
    checkpoint["kl_matched_soap"]["factor_generation"] += 1
    with pytest.raises(RuntimeError, match="outer/inner"):
        optimizer.load_state_dict(checkpoint)
    _assert_nested_equal(optimizer.state_dict(), before)
