# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch
from torch import nn

from verl.utils.optimizers import Muon, MuonWithAuxAdamW
from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer, partition_named_parameters_for_muon


class _TinyQwenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.self_attn = nn.ModuleDict({"q_proj": nn.Linear(4, 4)})
        self.mlp = nn.ModuleDict({"up_proj": nn.Linear(4, 8)})
        self.input_layernorm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 16, bias=False)


def test_muon_backport_matches_pytorch_reference_step():
    parameter = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    parameter.grad = torch.tensor([[0.1, -0.2], [0.3, -0.4]])
    optimizer = Muon([parameter], lr=1e-3, weight_decay=0.01)

    optimizer.step()

    expected = torch.tensor([[1.0001833, 2.0002110], [2.9997623, 4.0000300]])
    torch.testing.assert_close(parameter, expected, rtol=0, atol=5e-7)


def test_muon_partition_is_disjoint_exhaustive_and_excludes_auxiliary_parameters():
    model = _TinyQwenBlock()

    muon, auxiliary = partition_named_parameters_for_muon(model.named_parameters())

    muon_names = {name for name, _ in muon}
    auxiliary_names = {name for name, _ in auxiliary}
    all_names = {name for name, _ in model.named_parameters()}

    assert muon_names == {
        "self_attn.q_proj.weight",
        "mlp.up_proj.weight",
    }
    assert muon_names.isdisjoint(auxiliary_names)
    assert muon_names | auxiliary_names == all_names
    assert "embed_tokens.weight" in auxiliary_names
    assert "lm_head.weight" in auxiliary_names
    assert "input_layernorm.weight" in auxiliary_names
    assert "self_attn.q_proj.bias" in auxiliary_names


def test_muon_with_aux_adamw_factory_step_and_state_round_trip(monkeypatch):
    class _FakeMuon(torch.optim.SGD):
        def __init__(self, params, lr, weight_decay, momentum, nesterov, ns_steps, adjust_lr_fn):
            super().__init__(params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)

    monkeypatch.setattr(torch.optim, "Muon", _FakeMuon, raising=False)
    model = _TinyQwenBlock()
    config = FSDPOptimizerConfig(
        optimizer="MuonWithAuxAdamW",
        optimizer_impl="verl.utils.optimizers",
        lr=1e-3,
        weight_decay=0.01,
        override_optimizer_config={"auxiliary_eps": 1e-5},
    )
    optimizer = build_optimizer(model.named_parameters(), config)

    assert isinstance(optimizer, MuonWithAuxAdamW)
    assert optimizer.auxiliary.defaults["eps"] == 1e-5
    assert optimizer.parameter_routes["muon"] == (
        "self_attn.q_proj.weight",
        "mlp.up_proj.weight",
    )

    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    auxiliary_parameter = model.input_layernorm.weight
    assert optimizer.state[auxiliary_parameter] is optimizer.auxiliary.state[auxiliary_parameter]
    checkpoint = optimizer.state_dict()

    restored = build_optimizer(model.named_parameters(), config)
    restored.load_state_dict(checkpoint)
    assert restored.state_dict()["parameter_routes"] == checkpoint["parameter_routes"]
