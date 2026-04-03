#!/bin/sh
set -e

python - <<'PY'
import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except Exception as exc:
    raise SystemExit(f"huggingface_hub is required: {exc}")

models = [
    {
        "repo": os.getenv("HF_APP3_REPO"),
        "file": os.getenv("HF_APP3_FILE"),
        "dest": Path("/app/app3/resnet18_model_001.pth"),
    },
    {
        "repo": os.getenv("HF_APP4_REPO"),
        "file": os.getenv("HF_APP4_FILE"),
        "dest": Path("/app/app4/breast_cancer_cnn_model_updated.pth"),
    },
    {
        "repo": os.getenv("HF_APP5_REPO"),
        "file": os.getenv("HF_APP5_FILE"),
        "dest": Path("/app/app5/brain_tumor_resnet101_finetuned_v00.3.keras"),
    },
]

for item in models:
    repo = item["repo"]
    filename = item["file"]
    dest = item["dest"]
    if not repo or not filename:
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        continue
    path = hf_hub_download(repo_id=repo, filename=filename, cache_dir=os.getenv("HF_HOME"))
    dest.write_bytes(Path(path).read_bytes())

print("Model download check complete.")
PY

exec python /app/run.py
