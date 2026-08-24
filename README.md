# ppo-soap-gsm8k

A self-contained PPO harness for GSM8K with a **KL-matched SOAP** optimizer on the actor,
packaged so that a clone-and-run job scheduler can execute it with no shared filesystem.

Everything the run needs is either in this repository or fetched from a pinned public
source. There are no references to local paths, mounts, or internal hosts.

## Why it is packaged this way

Batch schedulers that hand out a GPU from a queue typically do three things and nothing
more: shallow-clone a git repository over https, install `requirements.txt`, and run one
entry point. They do not mount your data, they do not see your checkpoints, and the worker
is destroyed afterwards.

That constraint drives every design decision here:

- the pinned dataset lives **in the repository**, because it is small and must be
  byte-identical between runs;
- the model is fetched **by immutable revision**, not by tag;
- the resume checkpoint is fetched from a hub, because local disks are invisible;
- the entry point verifies its inputs before it touches a GPU, and refuses to start on a
  card that provably cannot host the run.

## Quickstart

```bash
SEED=1 RESUME_STEP=300 CKPT_REPO=<hub-repo-with-checkpoints> bash train.sh
```

With a queue that clones and runs:

```bash
<submit> --repo https://github.com/<owner>/ppo-soap-gsm8k \
         --entry train.sh --gpus 1 \
         --env SEED=1 --env RESUME_STEP=300 --env CKPT_REPO=<hub-repo>
```

`train.sh` proceeds in a deliberate order, cheapest gate first:

1. **Data integrity.** md5 of both parquet files. Needs no GPU and no installed packages,
   so a `--no-pip` dry submission still validates the clone, the entry point and the data.
2. **Environment.** Fails loudly unless `torch 2.6.0` and `vllm 0.8.5` are importable and
   CUDA is visible.
3. **Model** by pinned revision.
4. **Checkpoint** from the hub, with both actor and critic optimizer states verified.
5. **Memory arithmetic** (see below) — refuses the card rather than failing inside vLLM.
6. **Training**, then results copied to a shared path if one is writable.

## The memory trap worth knowing about

On a **shared** GPU, vLLM 0.8.5 computes its KV-cache budget as

```
available_kv_cache = total_gpu_memory * gpu_memory_utilization - peak_memory
```

and `peak_memory` includes a device-wide term:

```python
total_allocated = mem_get_info()[1] - mem_get_info()[0]   # total - free: DEVICE-WIDE
non_torch_allocations = total_allocated - torch_allocated
peak_memory += non_torch_allocations
```

`total - free` counts **every other process on the card**. Their memory is charged against
your budget, so the expression can go negative and the engine dies with
`Engine core initialization failed` even when there is plenty of free memory.

The counter-intuitive consequence: on a busy card `gpu_memory_utilization` must be
**raised**, not lowered. The requirement is

$$\text{gmu} \ge \frac{\text{used}_{\text{foreign}} + \text{peak}_{\text{ours}} + \text{kv cap} + \text{margin}}{\text{total}}$$

Lowering it — the instinctive move — guarantees zero KV-cache blocks. `train.sh` computes
this value from `nvidia-smi` at launch, prints all four terms, and refuses the card when
the requirement exceeds a threshold, instead of starting and dying.

Current upstream vLLM fixes the accounting: `vllm/utils/mem_utils.py` measures
`total_consumed = before_create.free_memory - after_profile.free_memory`, a **delta**
between snapshots, so pre-existing foreign memory is excluded, and it raises an explicit
error early when free memory is below the request. If you are on a newer vLLM, the tuning
direction inverts: keep `gpu_memory_utilization` at or below `free / total`.

## Layout

| Path | What it is |
|---|---|
| `train.sh` | queue entry point; gates, fetches, computes memory, runs |
| `run_matched_soap_config_adamw.sh` | the training configuration as executed |
| `run_matched_soap_config_lowmem.sh` | same, plus env-gated offload and `enforce_eager` |
| `soap_ppo.py` | standalone SOAP reference |
| `evaluate_*.py`, `calibrate_*.py`, `aggregate_*.py` | metric and gate computation |
| `data/gsm8k/*.parquet` | pinned dataset, md5-verified by `train.sh` |
| `vendor/verl` | vendored trainer the harness is built on |
| `requirements.txt` | exact `pip freeze` of a working environment |

Run artifacts — weights, checkpoints, metrics, logs — are deliberately **not** in the
repository; see `.gitignore`.

## Notes on reproducibility

- `vllm 0.8.5` pins `torch==2.6.0`, `torchvision==0.21.0`, `xformers==0.0.29.post2`.
  Base images shipping torch 2.1, 2.8 or 2.9 will not do; pick a Python 3.10 image and let
  `requirements.txt` install torch.
- Keep `save_freq` small on a shared card. A run that banks progress every few steps
  survives a neighbour's allocation spike; one that banks every 25 steps can loop forever
  between the same two checkpoints.
- Changing the rollout engine version changes generation. Results produced under different
  vLLM versions are not paired, and should not be compared as if they were.
