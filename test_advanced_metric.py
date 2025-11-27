"""
Test script for DetMetricAdvanced
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from ppocr.metrics.det_metric import DetMetricAdvanced

def test_basic_functionality():
    """Test basic functionality of DetMetricAdvanced"""
    print("Testing DetMetricAdvanced basic functionality...")

    # Initialize metric
    metric = DetMetricAdvanced(
        main_indicator='mAP',
        iou_thresholds=[0.5, 0.75, 0.9]
    )

    # Create dummy predictions and ground truth
    # Batch 1: Perfect match
    pred1 = {
        'points': [
            np.array([[10, 10], [100, 10], [100, 50], [10, 50]]),
        ],
        'scores': [0.95]
    }

    gt_polys1 = [
        np.array([[10, 10], [100, 10], [100, 50], [10, 50]]),
    ]
    ignore_tags1 = [False]

    # Batch 2: Partial match
    pred2 = {
        'points': [
            np.array([[15, 15], [95, 15], [95, 45], [15, 45]]),  # Slightly offset
            np.array([[200, 200], [250, 200], [250, 230], [200, 230]]),  # False positive
        ],
        'scores': [0.9, 0.8]
    }

    gt_polys2 = [
        np.array([[10, 10], [100, 10], [100, 50], [10, 50]]),
        np.array([[150, 150], [200, 150], [200, 180], [150, 180]]),  # Missed detection
    ]
    ignore_tags2 = [False, False]

    # Simulate batch processing
    batch1 = [None, None, [gt_polys1], [ignore_tags1]]
    batch2 = [None, None, [gt_polys2], [ignore_tags2]]

    preds1 = [pred1]
    preds2 = [pred2]

    # Call metric
    print("Processing batch 1...")
    metric(preds1, batch1)

    print("Processing batch 2...")
    metric(preds2, batch2)

    # Get results
    print("\nGetting metric results...")
    results = metric.get_metric()

    print("\n" + "="*60)
    print("DetMetricAdvanced Test Results:")
    print("="*60)
    for key, value in results.items():
        if key.startswith('IoU@'):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("="*60)

    # Validate
    assert 'mAP' in results, "mAP should be in results"
    assert 'precision' in results, "precision should be in results"
    assert 'recall' in results, "recall should be in results"
    assert 'hmean' in results, "hmean should be in results"
    assert 'IoU@0.50' in results, "IoU@0.50 should be in results"
    assert 'IoU@0.75' in results, "IoU@0.75 should be in results"
    assert 'IoU@0.90' in results, "IoU@0.90 should be in results"

    print("\n[PASS] All tests passed!")
    return True

def test_default_thresholds():
    """Test with default IoU thresholds"""
    print("\n\nTesting with default IoU thresholds...")

    metric = DetMetricAdvanced()

    # Simple perfect match case
    pred = {
        'points': [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])],
        'scores': [1.0]
    }
    gt_polys = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])]
    ignore_tags = [False]

    batch = [None, None, [gt_polys], [ignore_tags]]
    preds = [pred]

    metric(preds, batch)
    results = metric.get_metric()

    print("\nResults with default thresholds:")
    print(f"mAP: {results['mAP']:.4f}")
    print(f"Number of IoU thresholds: {sum(1 for k in results.keys() if k.startswith('IoU@'))}")

    assert results['precision'] == 1.0, "Perfect match should have precision=1.0"
    assert results['recall'] == 1.0, "Perfect match should have recall=1.0"
    assert results['hmean'] == 1.0, "Perfect match should have hmean=1.0"

    print("[PASS] Default threshold test passed!")
    return True

if __name__ == '__main__':
    try:
        test_basic_functionality()
        test_default_thresholds()
        print("\n" + "="*60)
        print("SUCCESS: All DetMetricAdvanced tests completed successfully!")
        print("="*60)
    except Exception as e:
        print(f"\nERROR: Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
