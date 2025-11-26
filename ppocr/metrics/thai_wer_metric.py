# copyright (c) 2020 PaddlePaddle Authors.
# All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import subprocess
import tempfile
from rapidfuzz.distance import Levenshtein


class ThaiWERMetric(object):
    """
    Thai Word Error Rate (WER) metric that uses external Thai tokenizers
    (newmm, attacut, deepcut) running in a separate environment.

    This metric is designed to:
    1. Run only when called from 'python tools/eval.py'
    2. Use a separate Python environment for Thai tokenization
    3. Calculate WER for multiple Thai tokenizers simultaneously

    Args:
        main_indicator (str): The main metric to return (default: "wer_avg")
        python_path (str): Path to Python interpreter with Thai tokenizers installed
        tokenizer_script (str): Path to the tokenization script
        engines (list): List of tokenizer engines to use (default: ["newmm", "attacut", "deepcut"])
                       Available: "newmm", "attacut", "deepcut"
        enabled (bool): Whether to enable WER calculation (default: True)
    """

    def __init__(
        self,
        main_indicator="wer_avg",
        python_path="python",
        tokenizer_script=None,
        engines=None,
        enabled=True,
        **kwargs
    ):
        self.main_indicator = main_indicator
        self.python_path = python_path
        self.enabled = enabled
        self.eps = 1e-5

        # Default to all three engines if not specified
        if engines is None:
            self.engines = ["newmm", "attacut", "deepcut"]
        else:
            # Validate engines
            valid_engines = ["newmm", "attacut", "deepcut"]
            self.engines = [e for e in engines if e in valid_engines]
            if not self.engines:
                print(f"[ThaiWERMetric] Warning: No valid engines specified. Using all engines.")
                self.engines = valid_engines

        # Auto-detect tokenizer script path
        if tokenizer_script is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
            self.tokenizer_script = os.path.join(
                project_root, "tools", "thai_tokenizer.py"
            )
        else:
            self.tokenizer_script = tokenizer_script

        # Check if we're running from tools/eval.py
        self._check_if_eval_context()

        self.reset()

    def _check_if_eval_context(self):
        """Check if we're being called from tools/eval.py"""
        # Check the call stack to see if eval.py is in the execution path
        import traceback
        stack = traceback.extract_stack()

        is_eval = False
        for frame in stack:
            if 'tools/eval.py' in frame.filename.replace('\\', '/') or \
               'tools\\eval.py' in frame.filename:
                is_eval = True
                break

        if not is_eval:
            self.enabled = False
            print("[ThaiWERMetric] Not running from tools/eval.py - WER calculation disabled")

    def _tokenize_batch(self, texts):
        """
        Tokenize a batch of texts using external Thai tokenizers.

        Args:
            texts (list): List of text strings to tokenize

        Returns:
            dict: Tokenization results for each tokenizer
                  Format: {
                      'newmm': [[word1, word2, ...], ...],
                      'attacut': [[word1, word2, ...], ...],
                      'deepcut': [[word1, word2, ...], ...]
                  }
        """
        if not self.enabled:
            return None

        if not os.path.exists(self.tokenizer_script):
            print(f"[ThaiWERMetric] Warning: Tokenizer script not found at {self.tokenizer_script}")
            print("[ThaiWERMetric] WER calculation will be skipped")
            self.enabled = False
            return None

        try:
            # Create temporary input file
            with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', delete=False, suffix='.json'
            ) as f:
                input_file = f.name
                json.dump({'texts': texts}, f, ensure_ascii=False)

            # Create temporary output file
            with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', delete=False, suffix='.json'
            ) as f:
                output_file = f.name

            # Run tokenizer script with specified engines
            cmd = [
                self.python_path,
                self.tokenizer_script,
                '--input', input_file,
                '--output', output_file,
                '--engines', ','.join(self.engines)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                print(f"[ThaiWERMetric] Tokenizer script failed:")
                print(f"STDERR: {result.stderr}")
                return None

            # Read results
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            # Cleanup
            os.unlink(input_file)
            os.unlink(output_file)

            return results

        except subprocess.TimeoutExpired:
            print("[ThaiWERMetric] Tokenization timeout")
            return None
        except Exception as e:
            print(f"[ThaiWERMetric] Tokenization error: {str(e)}")
            return None

    def _calculate_wer(self, pred_tokens, target_tokens):
        """
        Calculate Word Error Rate using Levenshtein distance.

        WER = (Substitutions + Insertions + Deletions) / Total words in reference

        Args:
            pred_tokens (list): List of predicted tokens
            target_tokens (list): List of target/reference tokens

        Returns:
            float: WER score
        """
        if len(target_tokens) == 0:
            return 0.0 if len(pred_tokens) == 0 else 1.0

        # Convert token lists to strings for Levenshtein distance
        pred_str = ' '.join(pred_tokens)
        target_str = ' '.join(target_tokens)

        # Calculate Levenshtein distance on word level
        distance = Levenshtein.distance(pred_tokens, target_tokens)
        wer = distance / len(target_tokens)

        return wer

    def __call__(self, pred_label, *args, **kwargs):
        """
        Calculate WER for a batch of predictions.

        Args:
            pred_label: Tuple of (preds, labels)
                       preds: List of (prediction_text, confidence) tuples
                       labels: List of (label_text, _) tuples

        Returns:
            dict: Batch metrics including WER for each enabled tokenizer
        """
        if not self.enabled:
            # Return 0.0 for all possible engines
            result = {f"wer_{eng}": 0.0 for eng in self.engines}
            result["wer_avg"] = 0.0
            return result

        preds, labels = pred_label

        # Extract texts
        pred_texts = [pred for pred, _ in preds]
        label_texts = [label for label, _ in labels]

        # Tokenize predictions and labels
        pred_tokenized = self._tokenize_batch(pred_texts)
        label_tokenized = self._tokenize_batch(label_texts)

        if pred_tokenized is None or label_tokenized is None:
            result = {f"wer_{eng}": 0.0 for eng in self.engines}
            result["wer_avg"] = 0.0
            return result

        # Calculate WER for each enabled tokenizer
        batch_wer = {}
        all_num = len(preds)

        for tokenizer in self.engines:
            wer_sum = 0.0
            for i in range(all_num):
                pred_tokens = pred_tokenized[tokenizer][i]
                label_tokens = label_tokenized[tokenizer][i]
                wer = self._calculate_wer(pred_tokens, label_tokens)
                wer_sum += wer

                # Store per-sample WER
                self.wer_samples[tokenizer].append(wer)

            batch_wer[tokenizer] = wer_sum / (all_num + self.eps)
            self.wer_totals[tokenizer] += wer_sum

        self.all_num += all_num

        # Build result dict with all enabled engines
        result = {f"wer_{eng}": batch_wer[eng] for eng in self.engines}

        # Calculate average WER across all enabled engines
        if self.engines:
            result["wer_avg"] = sum(batch_wer.values()) / len(self.engines)
        else:
            result["wer_avg"] = 0.0

        return result

    def get_metric(self):
        """
        Get final WER metrics across all batches.

        Returns:
            dict: Final WER for each enabled tokenizer plus average
        """
        if not self.enabled or self.all_num == 0:
            self.reset()
            result = {f"wer_{eng}": 0.0 for eng in self.engines}
            result["wer_avg"] = 0.0
            return result

        # Calculate final WER for each enabled engine
        results = {}
        for eng in self.engines:
            results[f"wer_{eng}"] = self.wer_totals[eng] / (self.all_num + self.eps)

        # Calculate average WER
        if self.engines:
            results["wer_avg"] = sum(self.wer_totals[eng] for eng in self.engines) / (len(self.engines) * (self.all_num + self.eps))
        else:
            results["wer_avg"] = 0.0

        self.reset()
        return results

    def reset(self):
        """Reset metric counters."""
        # Initialize counters for all possible engines
        self.wer_totals = {eng: 0.0 for eng in ['newmm', 'attacut', 'deepcut']}
        self.wer_samples = {eng: [] for eng in ['newmm', 'attacut', 'deepcut']}
        self.all_num = 0
