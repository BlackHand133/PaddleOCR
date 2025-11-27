# Advanced Detection Metrics for PaddleOCR

## Overview

This document describes the enhanced detection metrics available in PaddleOCR, specifically the `DetMetricAdvanced` metric class that provides AP/mAP and multi-threshold evaluation capabilities for research purposes.

## Features

### 1. Standard Metrics (Compatible with DetMetric)
- **Precision**: Detection accuracy (TP / (TP + FP))
- **Recall**: Detection completeness (TP / (TP + FN))
- **F-measure (hmean)**: Harmonic mean of precision and recall

### 2. Multi-Threshold Evaluation
Evaluates detection performance at multiple IoU thresholds:
- Default thresholds: `[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]`
- Provides detailed metrics (Precision, Recall, F1) for each threshold
- Helps understand model performance across different localization requirements

### 3. mAP (mean Average Precision)
- Calculates the mean of F1-scores across all IoU thresholds
- Primary metric for model selection (when `main_indicator: mAP`)
- Standard metric used in modern object detection research

## Usage

### 1. Configuration

Create or modify your detection config file to use `DetMetricAdvanced`:

```yaml
Metric:
  name: DetMetricAdvanced
  main_indicator: mAP
  iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
```

**Parameters:**
- `name`: Must be `DetMetricAdvanced`
- `main_indicator`: Metric used for best model selection. Options: `mAP`, `hmean`, `precision`, `recall` (default: `mAP`)
- `iou_thresholds`: List of IoU thresholds to evaluate (default: 10 thresholds from 0.5 to 0.95)
- `iou_constraint`: IoU threshold for matching predictions to ground truth (default: 0.5)
- `area_precision_constraint`: Area overlap ratio for "don't care" regions (default: 0.5)

### 2. Example Configurations

#### Example 1: Full Multi-Threshold Evaluation
```yaml
Metric:
  name: DetMetricAdvanced
  main_indicator: mAP
  iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
```

#### Example 2: COCO-style AP@[.5:.95]
```yaml
Metric:
  name: DetMetricAdvanced
  main_indicator: mAP
  iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
```

#### Example 3: Limited Thresholds (Faster Evaluation)
```yaml
Metric:
  name: DetMetricAdvanced
  main_indicator: mAP
  iou_thresholds: [0.5, 0.7, 0.9]
```

#### Example 4: Standard + AP@0.75
```yaml
Metric:
  name: DetMetricAdvanced
  main_indicator: hmean
  iou_thresholds: [0.5, 0.75]
```

### 3. Running Evaluation

Use the standard evaluation command:

```bash
python tools/eval.py -c configs/det/det_mv3_db_advanced.yml
```

### 4. Output Format

The metric output includes:

```
precision: 0.8523
recall: 0.7891
hmean: 0.8194
mAP: 0.7856
IoU@0.50: P:0.8523 R:0.7891 F1:0.8194
IoU@0.55: P:0.8401 R:0.7802 F1:0.8090
IoU@0.60: P:0.8267 R:0.7698 F1:0.7972
IoU@0.65: P:0.8120 R:0.7580 F1:0.7841
IoU@0.70: P:0.7965 R:0.7450 F1:0.7698
IoU@0.75: P:0.7798 R:0.7307 F1:0.7544
IoU@0.80: P:0.7612 R:0.7150 F1:0.7374
IoU@0.85: P:0.7410 R:0.6980 F1:0.7189
IoU@0.90: P:0.7189 R:0.6795 F1:0.6987
IoU@0.95: P:0.6950 R:0.6595 F1:0.6768
```

**Explanation:**
- `precision`, `recall`, `hmean`: Standard metrics at IoU=0.5
- `mAP`: Mean of F1-scores across all thresholds
- `IoU@X.XX`: Detailed metrics at each IoU threshold
  - `P`: Precision
  - `R`: Recall
  - `F1`: F-measure (hmean)

## Comparison with Existing Metrics

| Metric Class | Precision/Recall | Multi-threshold | mAP | Use Case |
|--------------|------------------|-----------------|-----|----------|
| `DetMetric` | ✅ (IoU=0.5) | ❌ | ❌ | Standard evaluation, fast |
| `DetFCEMetric` | ✅ (confidence thresholds) | ✅ (confidence-based) | ❌ | FCE/DRRG models |
| `DetMetricAdvanced` | ✅ (IoU=0.5) | ✅ (IoU-based) | ✅ | Research, publication |

## When to Use DetMetricAdvanced

### ✅ Recommended For:
1. **Research papers** requiring comprehensive metrics
2. **Model comparison** across different localization accuracies
3. **Publication** in top-tier conferences/journals
4. **Ablation studies** analyzing localization quality
5. **Competition submissions** requiring mAP metrics

### ⚠️ Consider Standard DetMetric For:
1. **Quick experiments** and iteration
2. **Training monitoring** (slightly slower evaluation)
3. **Production deployment** (single threshold sufficient)
4. **Limited computational resources**

## Performance Considerations

- **Evaluation time**: ~10x slower than `DetMetric` due to multi-threshold evaluation
- **Memory usage**: Similar to `DetMetric`
- **Training impact**: Set `cal_metric_during_train: False` (recommended)
- **Recommendation**: Use `eval_batch_step` to control evaluation frequency

## Integration with Existing Configs

To convert an existing config to use advanced metrics:

1. Change the Metric section:
```yaml
# Before
Metric:
  name: DetMetric
  main_indicator: hmean

# After
Metric:
  name: DetMetricAdvanced
  main_indicator: mAP
```

2. Adjust evaluation frequency if needed:
```yaml
Global:
  eval_batch_step: [0, 5000]  # Evaluate less frequently due to slower metrics
```

## Example Full Config

See `configs/det/det_mv3_db_advanced.yml` for a complete example configuration.

## Technical Details

### IoU Threshold Matching
- Predictions are matched to ground truth using greedy matching
- IoU calculated using Shapely polygon intersection/union
- Each ground truth can match at most one prediction
- Each prediction can match at most one ground truth

### mAP Calculation
```python
# For each IoU threshold t:
#   Calculate precision_t, recall_t, f1_t
#
# mAP = mean(f1_0.5, f1_0.55, ..., f1_0.95)
```

### Confidence Scores
- If detection outputs include confidence scores, they are stored but not currently used for AP calculation
- Future versions may include true AP calculation with PR curves

## References

1. ICDAR 2015 Text Detection Challenge: Standard evaluation protocol
2. COCO Detection Challenge: Multi-threshold mAP methodology
3. PaddleOCR DB: Differentiable Binarization for text detection

## FAQ

**Q: Can I use DetMetricAdvanced during training?**
A: Yes, but set `cal_metric_during_train: False` and adjust `eval_batch_step` to evaluate less frequently due to computational cost.

**Q: Is DetMetricAdvanced backward compatible with DetMetric?**
A: Yes, it provides all standard metrics (precision, recall, hmean at IoU=0.5) plus additional multi-threshold metrics.

**Q: How do I select the best model?**
A: Set `main_indicator` in the config. Common choices: `mAP` (comprehensive), `hmean` (standard).

**Q: Can I customize IoU thresholds?**
A: Yes, set `iou_thresholds: [0.5, 0.6, 0.7]` or any list of thresholds in the config.

**Q: Does this work with all detection algorithms?**
A: Yes, it works with DB, EAST, PSE, SAST, FCE, and all other detection algorithms in PaddleOCR.

## Support

For issues or questions, please open an issue on the PaddleOCR GitHub repository.
