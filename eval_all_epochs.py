#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to evaluate all epoch checkpoints and save results to a log file.
Usage: python eval_all_epochs.py
"""

import os
import subprocess
import glob
import re
from datetime import datetime

# Configuration
CONFIG_FILE = "configs/rec/PP-OCRv5/multi_language/th_PP-OCRv5_mobile_rec.yaml"
MODEL_DIR = "output/th_rec_ppocr_v5-lr0001"
LOG_FILE = "output/th_rec_ppocr_v5-lr0001/eval_all_epochs.log"

def get_epoch_number(filepath):
    """Extract epoch number from filename for sorting."""
    match = re.search(r'iter_epoch_(\d+)\.pdparams', filepath)
    if match:
        return int(match.group(1))
    return 0

def main():
    # Find all iter_epoch_*.pdparams files
    pattern = os.path.join(MODEL_DIR, "iter_epoch_*.pdparams")
    model_files = glob.glob(pattern)

    if not model_files:
        print(f"No model files found matching: {pattern}")
        return

    # Sort by epoch number
    model_files.sort(key=get_epoch_number)

    print(f"Found {len(model_files)} model checkpoints to evaluate")
    print(f"Results will be saved to: {LOG_FILE}")
    print("=" * 60)

    # Open log file
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        # Write header
        log.write(f"Evaluation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Config: {CONFIG_FILE}\n")
        log.write("=" * 60 + "\n\n")

        for i, model_path in enumerate(model_files, 1):
            epoch_num = get_epoch_number(model_path)

            print(f"\n[{i}/{len(model_files)}] Evaluating epoch {epoch_num}...")
            log.write(f"\n{'='*60}\n")
            log.write(f"Epoch: {epoch_num}\n")
            log.write(f"Model: {model_path}\n")
            log.write(f"{'='*60}\n")
            log.flush()

            # Build command
            cmd = [
                "python", "tools/eval.py",
                "-c", CONFIG_FILE,
                "-o", f"Global.pretrained_model={model_path}"
            ]

            # Run evaluation
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )

                # Write output to log
                if result.stdout:
                    log.write("\nSTDOUT:\n")
                    log.write(result.stdout)
                    print(result.stdout)

                if result.stderr:
                    log.write("\nSTDERR:\n")
                    log.write(result.stderr)

                log.write("\n")
                log.flush()

            except Exception as e:
                error_msg = f"Error evaluating {model_path}: {str(e)}\n"
                print(error_msg)
                log.write(error_msg)
                log.flush()

    print("\n" + "=" * 60)
    print(f"Evaluation complete! Results saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()
