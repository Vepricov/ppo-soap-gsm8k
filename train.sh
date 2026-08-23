#!/usr/bin/env bash
# Точка входа для очереди Cloud.ru: mlsub run --entry train.sh
#
# Очередь делает: git clone --depth 1 <repo> /tmp/app.N; cd туда;
# pip install --user -r requirements.txt (если не передан --no-pip); bash train.sh
#
# Зачем очередь. На общих H200 прогон не выживает: 23-08 за четыре минуты нашего
# старта чужая занятость выросла с 76742 до 141611 MiB, и FSDP упал с CUDA OOM при
# 401 MiB свободного. Волатильность соседей больше всего нашего footprint, поэтому
# никакая настройка gpu_memory_utilization этого не переживает. Здесь карта
# выделенная (a100plus.1gpu.80vG), измеренный пик 37785 MiB влезает с запасом.
set -Eeuo pipefail

SEED=${SEED:?SEED обязателен}
RESUME_STEP=${RESUME_STEP:-300}
TOTAL_STEPS=${TOTAL_STEPS:-435}
SAVE_FREQ=${SAVE_FREQ:-5}
CKPT_REPO=${CKPT_REPO:?CKPT_REPO обязателен: HF repo id с чекпоинтами}
MODEL_REPO=${MODEL_REPO:-Qwen/Qwen2.5-0.5B-Instruct}
# Ревизия закреплена: она снята с фактически использованной копии модели кампании
# (.cache/huggingface/download/*.metadata). Без пина задача может получить другие
# веса, и прогон перестанет быть парным к уже посчитанным сидам.
MODEL_REVISION=${MODEL_REVISION:-7ae557604adf67be50417f59c2c2f167def9a775}
SHARE=${SHARE_ROOT:-/home/jovyan/shares/SR006.nfs2}
RUN_TAG="causal-soap-seed${SEED}-${RESUME_STEP}-${TOTAL_STEPS}"

app=$(pwd)
work=${WORK_ROOT:-/tmp/rlm-${RUN_TAG}}
out="$work/out"
mkdir -p "$work" "$out"
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

log "seed=$SEED resume=$RESUME_STEP total=$TOTAL_STEPS save_freq=$SAVE_FREQ"
log "код: $app"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true

# --- 1. закреплённые данные ----------------------------------------------
# Гейт стоит ПЕРВЫМ сознательно: он не требует torch и vllm, поэтому задание,
# запущенное с --no-pip, всё равно проверяет клон, точку входа и целостность данных.
# Файлы лежат в репозитории, потому что fisher_prompt_indices ссылается на строки
# именно этих parquet, и пересоздать их из исходного датасета нельзя.
DATA_ROOT="$app/data/gsm8k"
python3 - "$DATA_ROOT" <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
want = {"train.parquet": "8d3fea4a716e886e5ebf047771147ed2",
        "test.parquet":  "d6a93d225945e6f4d0686298e4f3ad49"}
for name, md5 in want.items():
    p = root / name
    got = hashlib.md5(p.read_bytes()).hexdigest()
    assert got == md5, f"{name}: md5 {got} вместо {md5}, данные не те"
    print(f"{name}: md5 совпал")
PY

# --- 2. окружение ---------------------------------------------------------
python3 - <<'PY'
import sys
try:
    import torch, vllm
except Exception as e:
    sys.exit(f"окружение не готово: {e}; запустить без --no-pip")
assert torch.__version__.startswith("2.6.0"), f"нужен torch 2.6.0, есть {torch.__version__}"
assert vllm.__version__ == "0.8.5", f"нужен vllm 0.8.5, есть {vllm.__version__}"
assert torch.cuda.is_available(), "CUDA недоступна"
print(f"ок: torch {torch.__version__}, vllm {vllm.__version__}, {torch.cuda.get_device_name(0)}")
PY

# --- 3. модель ------------------------------------------------------------
export HF_HOME="${HF_HOME:-$work/hf}"
model_dir="$work/model"
if [[ ! -f "$model_dir/config.json" ]]; then
    log "качаю $MODEL_REPO ревизии $MODEL_REVISION"
    python3 - "$MODEL_REPO" "$MODEL_REVISION" "$model_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev, dst = sys.argv[1:4]
print("модель в", snapshot_download(repo, revision=rev, local_dir=dst))
PY
fi

# --- 4. чекпоинт ----------------------------------------------------------
# Локальные диски GPU-хостов задаче не видны, поэтому чекпоинт приезжает из HF.
ckpt_root="$work/checkpoints"
ckpt="$ckpt_root/global_step_${RESUME_STEP}"
if [[ ! -d "$ckpt" ]]; then
    log "качаю чекпоинт seed${SEED}/global_step_${RESUME_STEP} из $CKPT_REPO"
    python3 - "$CKPT_REPO" "$SEED" "$RESUME_STEP" "$work/dl" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, seed, step, dst = sys.argv[1:5]
print("скачано в", snapshot_download(
    repo, repo_type="model",
    allow_patterns=[f"seed{seed}/global_step_{step}/**"], local_dir=dst))
PY
    mkdir -p "$ckpt_root"
    mv "$work/dl/seed${SEED}/global_step_${RESUME_STEP}" "$ckpt"
fi
for role in actor critic; do
    [[ -f "$ckpt/$role/optim_world_size_1_rank_0.pt" ]] || { log "нет $role optim в $ckpt"; exit 3; }
done
echo "$RESUME_STEP" > "$ckpt_root/latest_checkpointed_iteration.txt"
log "чекпоинт готов: $(du -sh "$ckpt" | cut -f1)"

# --- 5. gmu под выделенную карту -----------------------------------------
# В vLLM 0.8.5 available_kv = total*gmu - peak_memory, где peak_memory device-wide.
# На выделенной карте чужого нет, поэтому вычитается только наш пик.
read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total \
    --format=csv,noheader,nounits -i 0 | tr -d ' ' | tr ',' ' ')
peak=${OUR_PEAK_MIB:-38000}
kv=${RL_MUON_VLLM_KV_CACHE_CAP_MIB:-2048}
gmu=$(python3 -c "print(f'{min(0.95, ($used + $peak + $kv + 3072)/$total):.3f}')")
log "карта: занято $used из $total MiB; наш пик $peak; gmu $gmu"

# --- 6. прогон ------------------------------------------------------------
export RL_MUON_VLLM_KV_CACHE_CAP_MIB="$kv"
export VLLM_USE_V1=1 TOKENIZERS_PARALLELISM=false
export RAY_TMPDIR="$work/ray" TMPDIR="$work/tmp"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

set +e
env RL_MUON_CAMPAIGN_ROOT="$work" \
    RL_MUON_VERL_ROOT="$app/vendor/verl" \
    PYTHON_BIN="$(command -v python3)" \
    DATA_ROOT="$DATA_ROOT" \
    MODEL_PATH="$model_dir" \
    OUTPUT_ROOT="$out" \
    EXPECTED_STEP="$TOTAL_STEPS" TOTAL_TRAINING_STEPS="$TOTAL_STEPS" \
    SAVE_FREQ="$SAVE_FREQ" TEST_FREQ=10 SEED="$SEED" \
    RESUME_MODE=resume_path RESUME_FROM_PATH="$ckpt" \
    TRAIN_BATCH_SIZE=256 PPO_MINI_BATCH_SIZE=64 PPO_MICRO_BATCH_SIZE=1 \
    ACTOR_LR=1e-6 FISHER_REFRESH_FREQUENCY=4 FISHER_FACTOR_RANK=16 FISHER_PROBE_COUNT=4 \
    GPU_MEMORY_UTILIZATION="$gmu" \
    ROLLOUT_MAX_MODEL_LEN=768 ROLLOUT_MAX_NUM_SEQS=64 ROLLOUT_MAX_NUM_BATCHED_TOKENS=2048 \
    ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}" \
    REF_PARAM_OFFLOAD=False CRITIC_PARAM_OFFLOAD=False \
    CRITIC_OPTIMIZER_OFFLOAD=False ACTOR_OPTIMIZER_OFFLOAD=False \
    bash "$app/run_matched_soap_config_lowmem.sh"
rc=$?
set -e
log "прогон завершён rc=$rc"

# --- 7. вынести результаты на общий том ----------------------------------
dest="$SHARE/rl-muon/$RUN_TAG"
if mkdir -p "$dest" 2>/dev/null; then
    find "$out" -maxdepth 3 \( -name 'metrics*.jsonl' -o -name 'train*.log' -o -name '*.status' \) \
        -exec cp -a {} "$dest/" \; 2>/dev/null || true
    log "метрики и логи в $dest"
else
    log "общий том $SHARE недоступен, результаты остались в $out"
fi
# Терминальные чекпоинты весят около 16 GiB на сид, а квота тома 100 ГБ,
# поэтому автоматически не копируются: забирать отдельным осознанным шагом.
log "терминальные чекпоинты в $out, на общий том не копируются"
exit "$rc"
