#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Thai Tokenization Script for WER Calculation

This script runs in a separate environment with Thai NLP libraries installed.
It tokenizes Thai text using three different tokenizers:
- newmm (PyThaiNLP default)
- attacut (PyThaiNLP's attacut)
- deepcut

Usage:
    python thai_tokenizer.py --input input.json --output output.json

Input JSON format:
    {
        "texts": ["text1", "text2", ...]
    }

Output JSON format:
    {
        "newmm": [[token1, token2, ...], [token1, token2, ...], ...],
        "attacut": [[token1, token2, ...], [token1, token2, ...], ...],
        "deepcut": [[token1, token2, ...], [token1, token2, ...], ...]
    }

Requirements (install in separate environment):
    pip install pythainlp attacut deepcut
"""

import sys
import json
import argparse


def tokenize_with_newmm(text):
    """Tokenize using PyThaiNLP newmm tokenizer."""
    try:
        from pythainlp.tokenize import word_tokenize
        return word_tokenize(text, engine='newmm')
    except ImportError:
        print("Error: pythainlp not installed. Please install: pip install pythainlp", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error in newmm tokenization: {str(e)}", file=sys.stderr)
        return []


def tokenize_with_attacut(text):
    """Tokenize using PyThaiNLP attacut tokenizer."""
    try:
        from pythainlp.tokenize import word_tokenize
        return word_tokenize(text, engine='attacut')
    except ImportError:
        print("Error: attacut not installed. Please install: pip install attacut", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error in attacut tokenization: {str(e)}", file=sys.stderr)
        return []


def tokenize_with_deepcut(text):
    """Tokenize using deepcut tokenizer."""
    try:
        import deepcut
        return deepcut.tokenize(text)
    except ImportError:
        print("Error: deepcut not installed. Please install: pip install deepcut", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error in deepcut tokenization: {str(e)}", file=sys.stderr)
        return []


def tokenize_batch(texts, engines=None):
    """
    Tokenize a batch of texts using specified tokenizers.

    Args:
        texts (list): List of text strings to tokenize
        engines (list): List of engine names to use (default: all three)

    Returns:
        dict: Tokenization results for each specified tokenizer
    """
    if engines is None:
        engines = ['newmm', 'attacut', 'deepcut']

    # Map engine names to functions
    engine_funcs = {
        'newmm': tokenize_with_newmm,
        'attacut': tokenize_with_attacut,
        'deepcut': tokenize_with_deepcut
    }

    results = {eng: [] for eng in engines}

    for text in texts:
        # Tokenize with each specified engine
        for engine in engines:
            if engine in engine_funcs:
                results[engine].append(engine_funcs[engine](text))
            else:
                print(f"Warning: Unknown engine '{engine}', skipping", file=sys.stderr)
                results[engine].append([])

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Tokenize Thai text using specified tokenizers'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSON file containing texts to tokenize'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file for tokenization results'
    )
    parser.add_argument(
        '--engines',
        type=str,
        default='newmm,attacut,deepcut',
        help='Comma-separated list of engines to use (default: newmm,attacut,deepcut)'
    )

    args = parser.parse_args()

    try:
        # Parse engines
        engines = [e.strip() for e in args.engines.split(',') if e.strip()]
        if not engines:
            engines = ['newmm', 'attacut', 'deepcut']

        print(f"Using engines: {', '.join(engines)}", file=sys.stderr)

        # Read input
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        texts = input_data.get('texts', [])

        if not texts:
            print("Warning: No texts to tokenize", file=sys.stderr)
            results = {eng: [] for eng in engines}
        else:
            # Tokenize with specified engines
            results = tokenize_batch(texts, engines)

        # Write output
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Successfully tokenized {len(texts)} texts with {len(engines)} engine(s)", file=sys.stderr)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
