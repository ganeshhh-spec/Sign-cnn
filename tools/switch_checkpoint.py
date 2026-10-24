#!/usr/bin/env python3
"""
Checkpoint Switcher
===================
Switches the backend between the original best checkpoint and the
fine-tuned checkpoint WITHOUT restarting the server.

Usage
-----
  python tools/switch_checkpoint.py --use original    # restore backup
  python tools/switch_checkpoint.py --use finetuned   # switch to fine-tuned
  python tools/switch_checkpoint.py --status          # show which is active

The active checkpoint is always results/models/checkpoint_best.pth.
"""

import argparse
import shutil
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR   = PROJECT_ROOT / "results" / "models"

ORIGINAL   = MODELS_DIR / "checkpoint_best_backup_20261024.pth"
FINETUNED  = MODELS_DIR / "checkpoint_finetuned.pth"
ACTIVE     = MODELS_DIR / "checkpoint_best.pth"
DEPLOY_JSON = MODELS_DIR / "deployment_info.json"


def status():
    if not ACTIVE.exists():
        print("[ERROR] Active checkpoint not found:", ACTIVE)
        return

    active_size = ACTIVE.stat().st_size
    orig_size   = ORIGINAL.stat().st_size   if ORIGINAL.exists()  else None
    ft_size     = FINETUNED.stat().st_size  if FINETUNED.exists() else None

    if orig_size and abs(active_size - orig_size) < 1024:
        which = "ORIGINAL (backup)"
    elif ft_size and abs(active_size - ft_size) < 1024:
        which = "FINE-TUNED"
    else:
        which = "UNKNOWN (size mismatch)"

    print(f"Active checkpoint : {ACTIVE}")
    print(f"Active size       : {active_size:,} bytes")
    print(f"Identified as     : {which}")
    print(f"Original backup   : {'exists' if ORIGINAL.exists() else 'MISSING'}")
    print(f"Fine-tuned        : {'exists' if FINETUNED.exists() else 'MISSING (run finetune_webcam.py first)'}")


def switch(target: str):
    if target == "original":
        src = ORIGINAL
        label = "ORIGINAL"
    elif target == "finetuned":
        src = FINETUNED
        label = "FINE-TUNED"
    else:
        print(f"[ERROR] Unknown target '{target}'. Use 'original' or 'finetuned'.")
        return

    if not src.exists():
        print(f"[ERROR] Source not found: {src}")
        if target == "finetuned":
            print("  → Run: python tools/finetune_webcam.py  first")
        return

    # Save a timestamped safety copy of the current active
    import time
    ts = int(time.time())
    safety = MODELS_DIR / f"checkpoint_best_pre_switch_{ts}.pth"
    shutil.copy2(ACTIVE, safety)
    print(f"[OK] Safety copy: {safety.name}")

    shutil.copy2(src, ACTIVE)
    print(f"[OK] Switched to {label}: {src.name} → {ACTIVE.name}")
    print()
    print("IMPORTANT: The running FastAPI server must be restarted to pick up the new checkpoint.")
    print("  Kill the server (Ctrl+C on the uvicorn task) and restart:")
    print("  .\\.venv\\Scripts\\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000")


def main():
    parser = argparse.ArgumentParser(description="Switch active ASL checkpoint")
    parser.add_argument("--use",    choices=["original", "finetuned"])
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status or (not args.use):
        status()
    elif args.use:
        switch(args.use)


if __name__ == "__main__":
    main()
