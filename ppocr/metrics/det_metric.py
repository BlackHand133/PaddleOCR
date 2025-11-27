# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["DetMetric", "DetFCEMetric", "DetMetricAdvanced"]

from .eval_det_iou import DetectionIoUEvaluator


class DetMetric(object):
    def __init__(self, main_indicator="hmean", **kwargs):
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        """
        batch: a list produced by dataloaders.
            image: np.ndarray  of shape (N, C, H, W).
            ratio_list: np.ndarray  of shape(N,2)
            polygons: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not.
        preds: a list of dict produced by post process
             points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.
        """
        gt_polyons_batch = batch[2]
        ignore_tags_batch = batch[3]
        for pred, gt_polyons, ignore_tags in zip(
            preds, gt_polyons_batch, ignore_tags_batch
        ):
            # prepare gt
            gt_info_list = [
                {"points": gt_polyon, "text": "", "ignore": ignore_tag}
                for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)
            ]
            # prepare det
            det_info_list = [
                {"points": det_polyon, "text": ""} for det_polyon in pred["points"]
            ]
            result = self.evaluator.evaluate_image(gt_info_list, det_info_list)
            self.results.append(result)

    def get_metric(self):
        """
        return metrics {
                 'precision': 0,
                 'recall': 0,
                 'hmean': 0
            }
        """

        metrics = self.evaluator.combine_results(self.results)
        self.reset()
        return metrics

    def reset(self):
        self.results = []  # clear results


class DetFCEMetric(object):
    def __init__(self, main_indicator="hmean", **kwargs):
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        """
        batch: a list produced by dataloaders.
            image: np.ndarray  of shape (N, C, H, W).
            ratio_list: np.ndarray  of shape(N,2)
            polygons: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not.
        preds: a list of dict produced by post process
             points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.
        """
        gt_polyons_batch = batch[2]
        ignore_tags_batch = batch[3]

        for pred, gt_polyons, ignore_tags in zip(
            preds, gt_polyons_batch, ignore_tags_batch
        ):
            # prepare gt
            gt_info_list = [
                {"points": gt_polyon, "text": "", "ignore": ignore_tag}
                for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)
            ]
            # prepare det
            det_info_list = [
                {"points": det_polyon, "text": "", "score": score}
                for det_polyon, score in zip(pred["points"], pred["scores"])
            ]

            for score_thr in self.results.keys():
                det_info_list_thr = [
                    det_info
                    for det_info in det_info_list
                    if det_info["score"] >= score_thr
                ]
                result = self.evaluator.evaluate_image(gt_info_list, det_info_list_thr)
                self.results[score_thr].append(result)

    def get_metric(self):
        """
        return metrics {'heman':0,
            'thr 0.3':'precision: 0 recall: 0 hmean: 0',
            'thr 0.4':'precision: 0 recall: 0 hmean: 0',
            'thr 0.5':'precision: 0 recall: 0 hmean: 0',
            'thr 0.6':'precision: 0 recall: 0 hmean: 0',
            'thr 0.7':'precision: 0 recall: 0 hmean: 0',
            'thr 0.8':'precision: 0 recall: 0 hmean: 0',
            'thr 0.9':'precision: 0 recall: 0 hmean: 0',
            }
        """
        metrics = {}
        hmean = 0
        for score_thr in self.results.keys():
            metric = self.evaluator.combine_results(self.results[score_thr])
            # for key, value in metric.items():
            #     metrics['{}_{}'.format(key, score_thr)] = value
            metric_str = "precision:{:.5f} recall:{:.5f} hmean:{:.5f}".format(
                metric["precision"], metric["recall"], metric["hmean"]
            )
            metrics["thr {}".format(score_thr)] = metric_str
            hmean = max(hmean, metric["hmean"])
        metrics["hmean"] = hmean

        self.reset()
        return metrics

    def reset(self):
        self.results = {
            0.3: [],
            0.4: [],
            0.5: [],
            0.6: [],
            0.7: [],
            0.8: [],
            0.9: [],
        }  # clear results


class DetMetricAdvanced(object):
    """
    Advanced detection metric with AP/mAP and multi-threshold evaluation.

    Provides:
    - Standard Precision, Recall, F-measure (hmean) at IoU=0.5
    - Multi-threshold evaluation at IoU thresholds: [0.5, 0.55, 0.6, ..., 0.95]
    - mAP (mean Average Precision) across all IoU thresholds
    - Detailed metrics per threshold for research analysis

    Args:
        main_indicator (str): Main metric for model selection. Default: 'mAP'
        iou_thresholds (list): List of IoU thresholds to evaluate.
                               Default: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        iou_constraint (float): IoU threshold for matching. Default: 0.5
        area_precision_constraint (float): Area precision constraint. Default: 0.5

    Example config:
        Metric:
          name: DetMetricAdvanced
          main_indicator: mAP
          iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    """

    def __init__(
        self,
        main_indicator="mAP",
        iou_thresholds=None,
        iou_constraint=0.5,
        area_precision_constraint=0.5,
        **kwargs
    ):
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        self.evaluator = DetectionIoUEvaluator(
            iou_constraint=iou_constraint,
            area_precision_constraint=area_precision_constraint,
            iou_thresholds=iou_thresholds,
        )
        self.main_indicator = main_indicator
        self.iou_thresholds = iou_thresholds
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        """
        Evaluate predictions against ground truth.

        Args:
            preds: List of predictions with 'points' and optionally 'scores'
            batch: List containing [image, ratio_list, polygons, ignore_tags]
        """
        gt_polyons_batch = batch[2]
        ignore_tags_batch = batch[3]

        for pred, gt_polyons, ignore_tags in zip(
            preds, gt_polyons_batch, ignore_tags_batch
        ):
            # Prepare ground truth
            gt_info_list = [
                {"points": gt_polyon, "text": "", "ignore": ignore_tag}
                for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)
            ]

            # Prepare detections with scores
            det_info_list = []
            for i, det_polyon in enumerate(pred["points"]):
                det_info = {"points": det_polyon, "text": ""}
                # Add score if available
                if "scores" in pred and i < len(pred["scores"]):
                    det_info["score"] = float(pred["scores"][i])
                else:
                    det_info["score"] = 1.0
                det_info_list.append(det_info)

            # Evaluate at multiple thresholds
            result = self.evaluator.evaluate_image_multi_threshold(
                gt_info_list, det_info_list
            )
            self.results.append(result)

    def get_metric(self):
        """
        Calculate and return final metrics.

        Returns:
            dict: Metrics including mAP, precision, recall, hmean, and per-threshold results
        """
        metrics = self.evaluator.combine_results_multi_threshold(self.results)

        # Format output for better readability
        output = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "hmean": metrics["hmean"],
            "mAP": metrics["mAP"],
        }

        # Add detailed per-threshold metrics
        for iou_thr, thr_metrics in metrics["threshold_metrics"].items():
            output[f"IoU@{iou_thr:.2f}"] = (
                f"P:{thr_metrics['precision']:.4f} "
                f"R:{thr_metrics['recall']:.4f} "
                f"F1:{thr_metrics['hmean']:.4f}"
            )

        self.reset()
        return output

    def reset(self):
        """Clear accumulated results."""
        self.results = []
