#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to evaluate all epoch checkpoints and save results to a log file.

Usage:
    python eval_all_epochs.py --model_dir output/th_rec_ppocr_v5-lr0001
    python eval_all_epochs.py --model_dir output/my_model
    python eval_all_epochs.py --model_dir output/my_model --config configs/my_config.yaml
"""

import os
import subprocess
import glob
import re
import argparse
from datetime import datetime

def get_epoch_number(filepath):
    """Extract epoch number from filename for sorting."""
    match = re.search(r'iter_epoch_(\d+)\.pdparams', filepath)
    if match:
        return int(match.group(1))
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all epoch checkpoints in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_all_epochs.py --model_dir output/th_rec_ppocr_v5-lr0001
  python eval_all_epochs.py --model_dir output/my_model
  python eval_all_epochs.py --model_dir output/my_model --config configs/my_config.yaml
        """
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing model checkpoints"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to config file (default: <model_dir>/config.yml)"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to log file (default: <model_dir>/eval_all_epochs.log)"
    )

    args = parser.parse_args()

    # Set model directory
    model_dir = args.model_dir

    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        return

    # Auto-detect config file from model_dir/config.yml if not specified
    if args.config:
        config_file = args.config
    else:
        config_file = os.path.join(model_dir, "config.yml")
        if not os.path.exists(config_file):
            print(f"❌ Config file not found: {config_file}")
            print(f"Please specify config file with --config option")
            return

    log_file = args.log_file if args.log_file else os.path.join(model_dir, "eval_all_epochs.log")

    # Find all iter_epoch_*.pdparams files
    pattern = os.path.join(model_dir, "iter_epoch_*.pdparams")
    model_files = glob.glob(pattern)

    if not model_files:
        print(f"❌ No model files found matching: {pattern}")
        print(f"Please check if the directory exists and contains iter_epoch_*.pdparams files")
        return

    # Sort by epoch number
    model_files.sort(key=get_epoch_number)

    print(f"📁 Model directory: {model_dir}")
    print(f"📝 Config file: {config_file} {'(auto-detected)' if not args.config else ''}")
    print(f"📊 Found {len(model_files)} model checkpoints to evaluate")
    print(f"💾 Results will be saved to: {log_file}")
    print("=" * 60)

    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Open log file
    with open(log_file, "w", encoding="utf-8") as log:
        # Write header
        log.write(f"Evaluation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Config: {config_file}\n")
        log.write(f"Model Directory: {model_dir}\n")
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
                "-c", config_file,
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
                error_msg = f"❌ Error evaluating {model_path}: {str(e)}\n"
                print(error_msg)
                log.write(error_msg)
                log.flush()

    print("\n" + "=" * 60)
    print(f"✅ Evaluation complete! Results saved to: {log_file}")

if __name__ == "__main__":
    main()
