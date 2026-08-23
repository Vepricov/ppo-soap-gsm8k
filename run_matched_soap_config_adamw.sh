#!/usr/bin/env bash
# Seed-selectable causal per-update KL-matched SOAP runner.
set -euo pipefail

SEED=${SEED:-0}
ACTOR_LR=${ACTOR_LR:-1e-6}
EXPECTED_STEP=${EXPECTED_STEP:-150}
SAVE_FREQ=${SAVE_FREQ:-$(( EXPECTED_STEP < 25 ? EXPECTED_STEP : 25 ))}
TEST_FREQ=${TEST_FREQ:-$(( EXPECTED_STEP < 10 ? EXPECTED_STEP : 10 ))}
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAMPAIGN_ROOT=${RL_MUON_CAMPAIGN_ROOT:-/home/shkodnik1917/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063}
VERL_ROOT=${RL_MUON_VERL_ROOT:-$repo_root/vendor/verl}
PYTHON_BIN=${PYTHON_BIN:-$CAMPAIGN_ROOT/venv/bin/python3}
MODEL_PATH=${MODEL_PATH:-$CAMPAIGN_ROOT/models/qwen2.5-0.5b-instruct}
DATA_ROOT=${DATA_ROOT:-$CAMPAIGN_ROOT/data/gsm8k}
OUTPUT_ROOT=${OUTPUT_ROOT:-$CAMPAIGN_ROOT/soap-actor-adamw-critic-pilot-seed$SEED}
ACTOR_OPTIMIZER=${ACTOR_OPTIMIZER:-KLMatchedSOAP}
ACTOR_OPTIMIZER_IMPL=${ACTOR_OPTIMIZER_IMPL:-verl.utils.kl_matched_soap}
SOAP_MAX_PRECOND_DIM=${SOAP_MAX_PRECOND_DIM:-2048}
ALPHA_MIN=${ALPHA_MIN:-0.05}
ALPHA_MAX=${ALPHA_MAX:-20.0}
ALPHA_CLAMP=${ALPHA_CLAMP:-true}
FISHER_MICRO_BATCH_SIZE=${FISHER_MICRO_BATCH_SIZE:-1}
FISHER_PROBE_COUNT=${FISHER_PROBE_COUNT:-4}
FISHER_PROBE_SEED=${FISHER_PROBE_SEED:-0}
FISHER_EXPECTED_STATES=${FISHER_EXPECTED_STATES:-57}
FISHER_FACTOR_RANK=${FISHER_FACTOR_RANK:-16}
FISHER_DENSE_THRESHOLD=${FISHER_DENSE_THRESHOLD:-256}
FISHER_REFRESH_FREQUENCY=${FISHER_REFRESH_FREQUENCY:-4}
FISHER_PROMPT_INDICES=${FISHER_PROMPT_INDICES:-[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
HYBRID_OVERRIDE=
if [[ "$ACTOR_OPTIMIZER" == "KLMatchedSOAPThenAdamW" ]]; then
    SWITCH_AFTER_GLOBAL_STEP=${SWITCH_AFTER_GLOBAL_STEP:-100}
    OPTIMIZER_UPDATES_PER_GLOBAL_STEP=${OPTIMIZER_UPDATES_PER_GLOBAL_STEP:-4}
    HYBRID_OVERRIDE=", switch_after_global_step: $SWITCH_AFTER_GLOBAL_STEP, optimizer_updates_per_global_step: $OPTIMIZER_UPDATES_PER_GLOBAL_STEP"
fi
ACTOR_OPTIMIZER_OVERRIDE="{eps: 1e-5, soap_precondition_frequency: 10, soap_max_precond_dim: $SOAP_MAX_PRECOND_DIM, auxiliary_eps: 1e-5, alpha_min: $ALPHA_MIN, alpha_max: $ALPHA_MAX, alpha_clamp: $ALPHA_CLAMP, fisher_dataset_path: '$DATA_ROOT/test.parquet', fisher_prompt_indices: $FISHER_PROMPT_INDICES, fisher_micro_batch_size: $FISHER_MICRO_BATCH_SIZE, fisher_probe_count: $FISHER_PROBE_COUNT, fisher_probe_seed: $FISHER_PROBE_SEED, fisher_expected_states: $FISHER_EXPECTED_STATES, fisher_factor_rank: $FISHER_FACTOR_RANK, fisher_dense_threshold: $FISHER_DENSE_THRESHOLD, fisher_refresh_frequency: $FISHER_REFRESH_FREQUENCY$HYBRID_OVERRIDE}"
RUN_NAME=${RUN_NAME:-qwen2.5-0.5b_gsm8k_ppo_kl_matched_soap_seed$SEED}
RUN_DIR=$OUTPUT_ROOT/$RUN_NAME
TERMINAL_ACTOR_CHECKPOINT=$RUN_DIR/checkpoints/global_step_$EXPECTED_STEP/actor/model_world_size_1_rank_0.pt

[[ -s "$DATA_ROOT/train.parquet" && -s "$DATA_ROOT/test.parquet" ]] || {
    echo "Pinned real GSM8K parquet files are missing under $DATA_ROOT" >&2
    exit 66
}
[[ -d "$MODEL_PATH" ]] || { echo "Pinned model is missing: $MODEL_PATH" >&2; exit 66; }
[[ -x "$PYTHON_BIN" ]] || { echo "Campaign Python is missing: $PYTHON_BIN" >&2; exit 66; }
[[ "$EXPECTED_STEP" =~ ^[1-9][0-9]*$ ]] || { echo "EXPECTED_STEP must be positive" >&2; exit 64; }
mkdir -p "$RUN_DIR"

export PATH="$CAMPAIGN_ROOT/venv/bin:$PATH"
export PYTHONPATH="$repo_root:$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics.jsonl"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export RL_MUON_VLLM_KV_CACHE_CAP_MIB=${RL_MUON_VLLM_KV_CACHE_CAP_MIB:-2048}
export TRITON_LIBCUDA_PATH=/lib/x86_64-linux-gnu
export HF_HOME="$CAMPAIGN_ROOT/hf-cache"
export TRANSFORMERS_CACHE="$CAMPAIGN_ROOT/hf-cache/hub"
export TMPDIR=${TMPDIR:-/dev/shm/rlm-soap-tmp-s$SEED}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/rlm-soap-ray-s$SEED}
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

# Ray rejects workers whose Python patch version differs from the driver. A
# relocated campaign can otherwise start the driver from one interpreter and
# repeatedly respawn workers from another after a venv symlink repair. Fail
# before allocating model memory unless the venv metadata and executable agree.
"$PYTHON_BIN" - "$CAMPAIGN_ROOT/venv/pyvenv.cfg" <<'PY'
from pathlib import Path
import platform
import sys

cfg = {}
for line in Path(sys.argv[1]).read_text().splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        cfg[key.strip()] = value.strip()
configured = cfg.get("version")
running = platform.python_version()
if configured != running:
    raise SystemExit(
        f"campaign Python mismatch: pyvenv.cfg={configured!r}, executable={running!r}"
    )
PY

# vLLM 0.8.5 measures device-wide free memory around sleep(). On a shared GPU,
# an unrelated process may allocate concurrently and make the delta negative
# even though vLLM successfully released its own tagged allocations. Keep the
# diagnostic, but do not turn another process's allocation into our run failure.
VLLM_GPU_WORKER=$CAMPAIGN_ROOT/venv/lib/python3.10/site-packages/vllm/v1/worker/gpu_worker.py
(
    flock 9
    "$PYTHON_BIN" - "$VLLM_GPU_WORKER" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
old = '''        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
'''
new = '''        if freed_bytes < 0:
            logger.warning(
                "Device-wide memory usage increased by %.2f GiB while vLLM was "
                "sleeping; this can be caused by another process on a shared GPU.",
                -freed_bytes / GiB_bytes,
            )
            freed_bytes = 0
        logger.info(
'''
text = path.read_text()
if new in text:
    pass
elif old in text:
    text = text.replace(old, new, 1)
else:
    raise SystemExit(f"unexpected vLLM sleep implementation in {path}")

# determine_available_memory profiles device-wide free memory before and after
# a model forward.  On a shared GPU another process can release memory during
# that interval, so the later value may legitimately be larger.  The values
# used for the cache budget below are still conservative and device-wide; only
# the invalid single-tenant assertion must be relaxed.
old = '''        assert self.init_gpu_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {self.init_gpu_memory}, current free memory"
            f" {free_gpu_memory}. This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")
'''
new = '''        if self.init_gpu_memory <= free_gpu_memory:
            logger.warning(
                "Device-wide free memory increased by %.2f GiB during vLLM "
                "profiling; continuing with the measured device-wide peak on "
                "this shared GPU.",
                (free_gpu_memory - self.init_gpu_memory) / GiB_bytes,
            )
'''
if new in text:
    pass
elif old in text:
    text = text.replace(old, new, 1)
else:
    raise SystemExit(f"unexpected vLLM profiling implementation in {path}")

# gpu_memory_utilization is device-wide in vLLM 0.8.5, so the KV reservation
# changes when an unrelated process on a shared GPU allocates or frees memory.
# Keep utilization high enough to survive that movement, but cap our actual KV
# cache to a fixed amount.  2 GiB covers the pinned 64 x 768-token rollout and
# prevents foreign-memory changes from moving this workload across its 35 GiB
# residency contract.
old = '''        available_kv_cache_memory = (
            total_gpu_memory * self.cache_config.gpu_memory_utilization -
            peak_memory)

        return int(available_kv_cache_memory)
'''
new = '''        available_kv_cache_memory = (
            total_gpu_memory * self.cache_config.gpu_memory_utilization -
            peak_memory)
        kv_cache_cap_mib = int(os.environ.get("RL_MUON_VLLM_KV_CACHE_CAP_MIB", "0"))
        if kv_cache_cap_mib > 0:
            available_kv_cache_memory = min(
                available_kv_cache_memory, kv_cache_cap_mib * 1024 * 1024
            )

        return int(available_kv_cache_memory)
'''
if new in text:
    pass
elif old in text:
    text = text.replace(old, new, 1)
else:
    raise SystemExit(f"unexpected vLLM KV cache calculation in {path}")
path.write_text(text)
PY
) 9>"$CAMPAIGN_ROOT/.vllm-shared-gpu-sleep-patch.lock"

# vLLM 0.8.5 assumes every CUDA_VISIBLE_DEVICES token is an integer.  Use
# stable GPU UUIDs in launchers so CUDA ordinal reordering cannot select a
# different physical card, and teach this older vLLM to resolve that UUID
# through NVML when it needs the host physical index.
VLLM_CUDA_PLATFORM=$CAMPAIGN_ROOT/venv/lib/python3.10/site-packages/vllm/platforms/cuda.py
(
    flock 9
    "$PYTHON_BIN" - "$VLLM_CUDA_PLATFORM" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
old = '''        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
'''
new = '''        physical_device_id = device_ids[device_id]
        if physical_device_id.startswith("GPU-"):
            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByUUID(physical_device_id)
                return int(pynvml.nvmlDeviceGetIndex(handle))
            finally:
                pynvml.nvmlShutdown()
        return int(physical_device_id)
'''
text = path.read_text()
if new in text:
    raise SystemExit(0)
if old not in text:
    raise SystemExit(f"unexpected vLLM CUDA device mapping in {path}")
path.write_text(text.replace(old, new, 1))
PY
) 9>"$CAMPAIGN_ROOT/.vllm-cuda-uuid-patch.lock"

cd "$VERL_ROOT"
"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="$DATA_ROOT/train.parquet" \
    data.val_files="$DATA_ROOT/test.parquet" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.seed="$SEED" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.optimizer="$ACTOR_OPTIMIZER" \
    actor_rollout_ref.actor.optim.optimizer_impl="$ACTOR_OPTIMIZER_IMPL" \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.override_optimizer_config="$ACTOR_OPTIMIZER_OVERRIDE" \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE:-1}" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.data_loader_seed="$SEED" \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.fsdp_config.seed="$SEED" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.45}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN:-768}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-64}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE:-1}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MICRO_BATCH_SIZE:-1}" \
    critic.model.path="$MODEL_PATH" \
    +critic.model.override_config.attn_implementation=sdpa \
    critic.model.use_remove_padding=False \
    critic.model.enable_gradient_checkpointing=True \
    critic.optim.optimizer=AdamW \
    critic.optim.optimizer_impl=torch.optim \
    critic.optim.lr=1e-5 \
    critic.optim.weight_decay=0.01 \
    critic.optim.override_optimizer_config='{eps: 1e-5}' \
    critic.ppo_mini_batch_size=64 \
    critic.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE:-1}" \
    critic.data_loader_seed="$SEED" \
    critic.fsdp.use_orig_params=True \
    critic.fsdp.seed="$SEED" \
    trainer.logger='[console,file]' \
    trainer.project_name=rl_muon_gsm8k \
    trainer.experiment_name="$RUN_NAME" \
    trainer.default_local_dir="$RUN_DIR/checkpoints" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.total_training_steps="$EXPECTED_STEP" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    +trainer.save_initial_checkpoint="${SAVE_INITIAL_CHECKPOINT:-True}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
    trainer.max_actor_ckpt_to_keep=8 \
    trainer.max_critic_ckpt_to_keep=8 \
    trainer.resume_mode="${RESUME_MODE:-auto}" \
    ${RESUME_FROM_PATH:+trainer.resume_from_path=$RESUME_FROM_PATH} \
    "$@" 2>&1 | tee -a "$RUN_DIR/train.log"

if [[ -n "${H_RLM_008_FROZEN_CAPTURE_ROOT:-}" ]]; then
    source_step=${H_RLM_008_SOURCE_STEP:?capture mode requires H_RLM_008_SOURCE_STEP}
    for step in $(seq $((source_step + 1)) "$EXPECTED_STEP"); do
        [[ -s "$H_RLM_008_FROZEN_CAPTURE_ROOT/global_step_$step.dp" ]] || {
            echo "Missing frozen audit capture for global step $step" >&2
            exit 70
        }
    done
    exit 0
fi

[[ -f "$TERMINAL_ACTOR_CHECKPOINT" ]] || {
    echo "Terminal KL-matched SOAP actor checkpoint is missing: $TERMINAL_ACTOR_CHECKPOINT" >&2
    exit 70
}
