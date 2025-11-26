# Evaluation Tool for All Epoch Checkpoints

## Overview

`eval_all_epochs.py` is a utility script to automatically evaluate all epoch checkpoints (`iter_epoch_*.pdparams`) in a training output directory and save the results to a log file.

## Features

- ✅ Automatically finds all `iter_epoch_*.pdparams` files in the specified directory
- ✅ Evaluates checkpoints in order (epoch 10, 20, 30, ...)
- ✅ Auto-detects `config.yml` from the model directory
- ✅ Saves detailed evaluation results to a log file
- ✅ Shows progress during evaluation

## Usage

### Basic Usage

Evaluate all checkpoints in a model directory:

```bash
python eval_all_epochs.py --model_dir output/th_rec_ppocr_v5-lr0001
```

The script will:
1. Look for `config.yml` in the model directory (auto-detected)
2. Find all `iter_epoch_*.pdparams` files
3. Evaluate each checkpoint
4. Save results to `output/th_rec_ppocr_v5-lr0001/eval_all_epochs.log`

### Custom Config File

If you need to use a different config file:

```bash
python eval_all_epochs.py --model_dir output/my_model --config configs/my_config.yaml
```

### Custom Log File

Specify a custom log file location:

```bash
python eval_all_epochs.py --model_dir output/my_model --log_file results/evaluation.log
```

## Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model_dir` | Yes | - | Directory containing model checkpoints |
| `--config`, `-c` | No | `<model_dir>/config.yml` | Path to config file |
| `--log_file` | No | `<model_dir>/eval_all_epochs.log` | Path to output log file |

## Output

The script generates a log file with the following information for each checkpoint:

```
============================================================
Epoch: 10
Model: output/th_rec_ppocr_v5-lr0001/iter_epoch_10.pdparams
============================================================

STDOUT:
[eval result] acc: 0.8500, norm_edit_dis: 0.9200, cer: 0.0800
...
```

## Requirements

- The model directory must contain `iter_epoch_*.pdparams` files
- A valid `config.yml` file must exist in the model directory (or specified via `--config`)
- The config file must contain valid evaluation dataset configuration

## Example Workflow

1. Train your model:
   ```bash
   python tools/train.py -c configs/rec/PP-OCRv5/multi_language/th_PP-OCRv5_mobile_rec.yaml
   ```

2. After training completes, evaluate all checkpoints:
   ```bash
   python eval_all_epochs.py --model_dir output/th_rec_ppocr_v5
   ```

3. Review results:
   ```bash
   cat output/th_rec_ppocr_v5/eval_all_epochs.log
   ```

## Notes

- The script evaluates checkpoints sequentially (not in parallel)
- Each evaluation runs the full validation dataset
- Progress is shown in the console and saved to the log file
- If a checkpoint evaluation fails, the error is logged and the script continues to the next checkpoint
