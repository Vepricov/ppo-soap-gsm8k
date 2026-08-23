#!/usr/bin/env python3
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
seed = int(sys.argv[2])
metrics_paths = list(output_root.glob(f"*seed{seed}/metrics.jsonl"))
if len(metrics_paths) != 1:
    raise SystemExit(f"expected one metrics file, found {len(metrics_paths)}")
run_root = metrics_paths[0].parent
last_step = None
with metrics_paths[0].open() as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        data = row.get("data") or {}
        step = data.get("training/global_step", row.get("step"))
        if isinstance(step, int):
            last_step = step
if last_step != 150:
    raise SystemExit(f"last training step is {last_step}, expected 150")
checkpoint = run_root / "checkpoints" / "global_step_150"
required = [
    "actor/model_world_size_1_rank_0.pt",
    "actor/optim_world_size_1_rank_0.pt",
    "actor/extra_state_world_size_1_rank_0.pt",
    "critic/model_world_size_1_rank_0.pt",
    "critic/optim_world_size_1_rank_0.pt",
    "critic/extra_state_world_size_1_rank_0.pt",
    "data.pt",
]
missing = [name for name in required if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0]
if missing:
    raise SystemExit(f"missing or empty terminal checkpoint files: {missing}")
print(f"validated seed={seed} step={last_step} checkpoint={checkpoint}")
