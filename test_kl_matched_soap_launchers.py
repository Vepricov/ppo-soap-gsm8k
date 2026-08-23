"""Static launcher contract tests that do not require the target GPU runtime."""

from pathlib import Path

ROOT = Path(__file__).parent


def test_production_launcher_has_exact_seed_step_validation_and_checkpoint_contract():
    seed_runner = (ROOT / "run_kl_matched_soap_seed.sh").read_text()
    campaign = (ROOT / "launch_kl_matched_soap_seeds.sh").read_text()
    runner = (ROOT / "run_matched_soap_config_adamw.sh").read_text()
    assert "0 25 50 75 100 125 150" in seed_runner
    assert "EXPECTED_STEP=150" in seed_runner
    assert "SAVE_FREQ=25" in seed_runner
    assert "TEST_FREQ=25" in seed_runner
    assert "for seed in 0 1 2" in campaign
    assert "critic.optim.optimizer=AdamW" in runner
    assert "critic.optim.optimizer_impl=torch.optim" in runner
    assert "ACTOR_OPTIMIZER=KLMatchedSOAP" in runner
    assert "ACTOR_OPTIMIZER_IMPL=verl.utils.kl_matched_soap" in runner
    assert "fisher_probe_count: $FISHER_PROBE_COUNT" in runner
    assert "fisher_probe_seed: $FISHER_PROBE_SEED" in runner
    assert "ExactLogitsJVPFisher" not in (
        ROOT / "vendor/verl/verl/utils/kl_matched_soap.py"
    ).read_text()
    assert "torch.func" not in (
        ROOT / "vendor/verl/verl/utils/kl_matched_soap.py"
    ).read_text()
    assert "nvmlDeviceGetHandleByUUID" in runner
    assert "physical_device_id.startswith(\"GPU-\")" in runner
    assert "self.init_gpu_memory <= free_gpu_memory" in runner
    assert "continuing with the measured device-wide peak" in runner
    assert "RL_MUON_VLLM_KV_CACHE_CAP_MIB=${RL_MUON_VLLM_KV_CACHE_CAP_MIB:-2048}" in runner
    assert "available_kv_cache_memory = min(" in runner
    assert "fisher_prompt_indices: $FISHER_PROMPT_INDICES" in runner


def test_smoke_is_exactly_one_step_then_real_auto_resume_to_step_two():
    smoke = (ROOT / "smoke_kl_matched_soap_resume.sh").read_text()
    opt_harness = (ROOT / "run_opt_factorized_smoke.sh").read_text()
    runner = (ROOT / "run_matched_soap_config_adamw.sh").read_text()
    assert smoke.index("EXPECTED_STEP=1") < smoke.index("EXPECTED_STEP=2")
    assert "global_step_1/actor/optim_world_size_1_rank_0.pt" in smoke
    assert "global_step_2/actor/optim_world_size_1_rank_0.pt" in smoke
    assert "trainer.resume_mode=auto" in runner
    assert "+trainer.save_initial_checkpoint=True" in runner
    assert "VERL_ROOT=${RL_MUON_VERL_ROOT:-$repo_root/vendor/verl}" in runner
    assert "RAY_TMPDIR=${RAY_TMPDIR:-/tmp/rlm-kfac-ray-$$}" in opt_harness
    assert "RAY_TMPDIR=${RAY_TMPDIR:-/tmp/rlm-kfac-ray}" not in opt_harness
    assert "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.12}" in opt_harness
    assert "MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-5120}" in opt_harness
    assert "free < MIN_GPU_FREE_MIB" in opt_harness
    assert "own_gpu_memory_mib" in opt_harness
    assert "own_used > MAX_GPU_DELTA_MIB" in opt_harness
    assert "peak_own_memory_used_mib" in opt_harness
    assert "OUTPUT_ROOT=$OUTPUT_ROOT" in opt_harness


def test_production_launchers_preserve_a100_memory_reserve():
    seed_runner = (ROOT / "run_kl_matched_soap_seed.sh").read_text()
    pair_runner = (ROOT / "run_matched_soap_pair.sh").read_text()
    assert 'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.20}"' in seed_runner
    assert 'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.20}"' in pair_runner


def test_factorized_production_claims_physical_gpu_and_enforces_resource_contract():
    harness = (ROOT / "run_factorized_production.sh").read_text()
    waiter = (ROOT / "wait_factorized_production.sh").read_text()
    assert 'CUDA_VISIBLE_DEVICES="$GPU_UUID"' in harness
    assert 'resolved_uuid" == "$GPU_UUID' in harness
    assert "own_used > MAX_GPU_DELTA_MIB" in harness
    estimate_branch = harness[harness.index("own_used > MAX_GPU_DELTA_MIB") :]
    estimate_branch = estimate_branch[: estimate_branch.index("fi")]
    assert "kill -TERM" not in estimate_branch
    assert "continuing while free memory remains above the safety reserve" in harness
    assert "free < MIN_GPU_FREE_MIB" in harness
    assert 'RL_MUON_VLLM_KV_CACHE_CAP_MIB=${RL_MUON_VLLM_KV_CACHE_CAP_MIB:-512}' in harness
    assert "critic.fsdp.param_offload=True" in harness
    assert "critic.fsdp.optimizer_offload=True" in harness
    assert 'SEED="$SEED"' in harness
    assert 'bash "$SCRIPT_ROOT/run_kl_matched_soap_seed.sh"' in harness
    assert (
        'FISHER_PROMPT_INDICES="[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]"'
        in harness
    )
    assert "flock -n" in waiter
    assert "MIN_START_FREE_MIB=$((PROJECTED_NEED_MIB + MIN_GPU_FREE_MIB))" in waiter
    assert "MIN_MEM_AVAILABLE_MIB=$((HOST_RAM_PEAK_MIB + HOST_RAM_RESERVE_MIB))" in waiter
    assert 'PROJECTED_NEED_MIB=${PROJECTED_NEED_MIB:-35840}' in waiter
    assert 'MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-5120}' in waiter


def test_factorized_adamw_only_stops_when_the_free_memory_reserve_is_breached():
    harness = (ROOT / "run_factorized_adamw_production.sh").read_text()
    estimate_branch = harness[harness.index("own_used > MAX_GPU_DELTA_MIB") :]
    estimate_branch = estimate_branch[: estimate_branch.index("fi")]
    assert "kill -TERM" not in estimate_branch
    assert "continuing while free memory remains above the safety reserve" in harness
    reserve_branch = harness[harness.index("free < MIN_GPU_FREE_MIB") :]
    assert "kill -TERM" in reserve_branch


def test_fsdp_proposal_hook_is_after_gradient_clipping_and_before_parameter_step():
    engine = (
        ROOT / "vendor/verl/verl/workers/engine/fsdp/transformer_impl.py"
    ).read_text()
    optimizer_step = engine[
        engine.index("    def optimizer_step(self):") : engine.index(
            "    def lr_scheduler_step(self):"
        )
    ]
    clip = optimizer_step.index("clip_grad_norm_")
    step = optimizer_step.index("self.optimizer.step()")
    assert clip < step
    assert "latest_telemetry" in optimizer_step
