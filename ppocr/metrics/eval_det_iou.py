#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
from shapely.geometry import Polygon

"""
reference from :
https://github.com/MhLiao/DB/blob/3c32b808d4412680310d3d28eeb6a2d5bf1566c5/concern/icdar2015_eval/detection/iou.py#L8
"""


class DetectionIoUEvaluator(object):
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5,
                 iou_thresholds=None):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint
        # For multi-threshold evaluation (AP/mAP)
        if iou_thresholds is None:
            self.iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        else:
            self.iou_thresholds = iou_thresholds

    def evaluate_image(self, gt, pred, return_iou_mat=False):
        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        def compute_ap(confList, matchList, numGtCare):
            correct = 0
            AP = 0
            if len(confList) > 0:
                confList = np.array(confList)
                matchList = np.array(matchList)
                sorted_ind = np.argsort(-confList)
                confList = confList[sorted_ind]
                matchList = matchList[sorted_ind]
                for n in range(len(confList)):
                    match = matchList[n]
                    if match:
                        correct += 1
                        AP += float(correct) / (n + 1)

                if numGtCare > 0:
                    AP /= numGtCare

            return AP

        perSampleMetrics = {}

        matchedSum = 0

        Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")

        numGlobalCareGt = 0
        numGlobalCareDet = 0

        arrGlobalConfidences = []
        arrGlobalMatches = []

        recall = 0
        precision = 0
        hmean = 0

        detMatched = 0

        iouMat = np.empty([1, 1])

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        # Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []

        pairs = []
        detMatchedNums = []

        arrSampleConfidences = []
        arrSampleMatch = []

        evaluationLog = ""

        for n in range(len(gt)):
            points = gt[n]["points"]
            dontCare = gt[n]["ignore"]
            if not Polygon(points).is_valid:
                continue

            gtPol = points
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)

        evaluationLog += (
            "GT polygons: "
            + str(len(gtPols))
            + (
                " (" + str(len(gtDontCarePolsNum)) + " don't care)\n"
                if len(gtDontCarePolsNum) > 0
                else "\n"
            )
        )

        for n in range(len(pred)):
            points = pred[n]["points"]
            if not Polygon(points).is_valid:
                continue

            detPol = points
            detPols.append(detPol)
            detPolPoints.append(points)
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:
                    dontCarePol = gtPols[dontCarePol]
                    intersected_area = get_intersection(dontCarePol, detPol)
                    pdDimensions = Polygon(detPol).area
                    precision = (
                        0 if pdDimensions == 0 else intersected_area / pdDimensions
                    )
                    if precision > self.area_precision_constraint:
                        detDontCarePolsNum.append(len(detPols) - 1)
                        break

        evaluationLog += (
            "DET polygons: "
            + str(len(detPols))
            + (
                " (" + str(len(detDontCarePolsNum)) + " don't care)\n"
                if len(detDontCarePolsNum) > 0
                else "\n"
            )
        )

        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrixs
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if (
                        gtRectMat[gtNum] == 0
                        and detRectMat[detNum] == 0
                        and gtNum not in gtDontCarePolsNum
                        and detNum not in detDontCarePolsNum
                    ):
                        if iouMat[gtNum, detNum] > self.iou_constraint:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({"gt": gtNum, "det": detNum})
                            detMatchedNums.append(detNum)
                            evaluationLog += (
                                "Match GT #"
                                + str(gtNum)
                                + " with Det #"
                                + str(detNum)
                                + "\n"
                            )

        numGtCare = len(gtPols) - len(gtDontCarePolsNum)
        numDetCare = len(detPols) - len(detDontCarePolsNum)
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare

        hmean = (
            0
            if (precision + recall) == 0
            else 2.0 * precision * recall / (precision + recall)
        )

        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics = {
            "gtCare": numGtCare,
            "detCare": numDetCare,
            "detMatched": detMatched,
        }

        if return_iou_mat:
            perSampleMetrics["iouMat"] = iouMat
            perSampleMetrics["gtPols"] = gtPols
            perSampleMetrics["detPols"] = detPols
            perSampleMetrics["gtDontCarePolsNum"] = gtDontCarePolsNum
            perSampleMetrics["detDontCarePolsNum"] = detDontCarePolsNum

        return perSampleMetrics

    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result["gtCare"]
            numGlobalCareDet += result["detCare"]
            matchedSum += result["detMatched"]

        methodRecall = (
            0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
        )
        methodPrecision = (
            0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
        )
        methodHmean = (
            0
            if methodRecall + methodPrecision == 0
            else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)
        )
        methodMetrics = {
            "precision": methodPrecision,
            "recall": methodRecall,
            "hmean": methodHmean,
        }

        return methodMetrics

    def evaluate_image_multi_threshold(self, gt, pred):
        """
        Evaluate detection at multiple IoU thresholds for AP/mAP calculation.
        Returns metrics for each IoU threshold.
        """
        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        perSampleMetrics = {}
        Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")

        gtPols = []
        detPols = []
        gtDontCarePolsNum = []
        detDontCarePolsNum = []

        # Collect confidence scores
        detScores = []

        # Parse ground truth polygons
        for n in range(len(gt)):
            points = gt[n]["points"]
            dontCare = gt[n]["ignore"]
            if not Polygon(points).is_valid:
                continue
            gtPols.append(points)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)

        # Parse detection polygons
        for n in range(len(pred)):
            points = pred[n]["points"]
            if not Polygon(points).is_valid:
                continue
            detPols.append(points)
            # Get confidence score if available
            score = pred[n].get("score", 1.0)
            detScores.append(score)

            # Check if overlaps with don't care regions
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:
                    dontCarePol = gtPols[dontCarePol]
                    intersected_area = get_intersection(dontCarePol, points)
                    pdDimensions = Polygon(points).area
                    precision = (
                        0 if pdDimensions == 0 else intersected_area / pdDimensions
                    )
                    if precision > self.area_precision_constraint:
                        detDontCarePolsNum.append(len(detPols) - 1)
                        break

        # Calculate IoU matrix
        iouMat = np.zeros([len(gtPols), len(detPols)])
        if len(gtPols) > 0 and len(detPols) > 0:
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

        numGtCare = len(gtPols) - len(gtDontCarePolsNum)

        # Evaluate at each IoU threshold
        threshold_results = {}
        for iou_thr in self.iou_thresholds:
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            detMatched = 0

            if len(gtPols) > 0 and len(detPols) > 0:
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        if (
                            gtRectMat[gtNum] == 0
                            and detRectMat[detNum] == 0
                            and gtNum not in gtDontCarePolsNum
                            and detNum not in detDontCarePolsNum
                        ):
                            if iouMat[gtNum, detNum] > iou_thr:
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1
                                detMatched += 1

            numDetCare = len(detPols) - len(detDontCarePolsNum)

            if numGtCare == 0:
                recall = float(1)
                precision = float(0) if numDetCare > 0 else float(1)
            else:
                recall = float(detMatched) / numGtCare
                precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare

            hmean = (
                0
                if (precision + recall) == 0
                else 2.0 * precision * recall / (precision + recall)
            )

            threshold_results[iou_thr] = {
                "gtCare": numGtCare,
                "detCare": numDetCare,
                "detMatched": detMatched,
                "precision": precision,
                "recall": recall,
                "hmean": hmean,
            }

        perSampleMetrics["threshold_results"] = threshold_results
        perSampleMetrics["detScores"] = detScores
        perSampleMetrics["numGtCare"] = numGtCare

        return perSampleMetrics

    def combine_results_multi_threshold(self, results):
        """
        Combine results from multiple images for multi-threshold evaluation.
        Calculate AP for each threshold and mAP.
        """
        # Aggregate results per threshold
        threshold_metrics = {}
        for iou_thr in self.iou_thresholds:
            numGlobalCareGt = 0
            numGlobalCareDet = 0
            matchedSum = 0

            for result in results:
                thr_result = result["threshold_results"][iou_thr]
                numGlobalCareGt += thr_result["gtCare"]
                numGlobalCareDet += thr_result["detCare"]
                matchedSum += thr_result["detMatched"]

            methodRecall = (
                0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
            )
            methodPrecision = (
                0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
            )
            methodHmean = (
                0
                if methodRecall + methodPrecision == 0
                else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)
            )

            threshold_metrics[iou_thr] = {
                "precision": methodPrecision,
                "recall": methodRecall,
                "hmean": methodHmean,
            }

        # Calculate mAP (mean of hmean across all thresholds)
        mAP = np.mean([metrics["hmean"] for metrics in threshold_metrics.values()])

        # Also include standard metrics at 0.5 IoU for compatibility
        standard_metrics = threshold_metrics[0.5].copy()
        standard_metrics["mAP"] = mAP
        standard_metrics["threshold_metrics"] = threshold_metrics

        return standard_metrics


if __name__ == "__main__":
    evaluator = DetectionIoUEvaluator()
    gts = [
        [
            {
                "points": [(0, 0), (1, 0), (1, 1), (0, 1)],
                "text": 1234,
                "ignore": False,
            },
            {
                "points": [(2, 2), (3, 2), (3, 3), (2, 3)],
                "text": 5678,
                "ignore": False,
            },
        ]
    ]
    preds = [
        [
            {
                "points": [(0.1, 0.1), (1, 0), (1, 1), (0, 1)],
                "text": 123,
                "ignore": False,
            }
        ]
    ]
    results = []
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)
