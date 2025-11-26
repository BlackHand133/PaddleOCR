#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for Thai tokenizer

This script helps you test if the Thai tokenizer setup is working correctly.

Usage:
    # Run from main PaddleOCR environment
    python tools/test_thai_tokenizer.py

    # Or specify custom Python path
    python tools/test_thai_tokenizer.py --python_path thai_tokenizer_env/Scripts/python
"""

import os
import sys
import json
import argparse
import tempfile
import subprocess


def test_tokenizer(python_path='python', tokenizer_script=None):
    """
    Test the Thai tokenizer setup.

    Args:
        python_path (str): Path to Python interpreter with Thai tokenizers
        tokenizer_script (str): Path to tokenizer script
    """
    # Auto-detect tokenizer script
    if tokenizer_script is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tokenizer_script = os.path.join(current_dir, "thai_tokenizer.py")

    print("=" * 60)
    print("Thai Tokenizer Test")
    print("=" * 60)
    print(f"Python path: {python_path}")
    print(f"Tokenizer script: {tokenizer_script}")
    print()

    # Check if script exists
    if not os.path.exists(tokenizer_script):
        print(f"❌ ERROR: Tokenizer script not found at: {tokenizer_script}")
        return False

    print("✅ Tokenizer script found")

    # Test texts
    test_texts = [
        "สวัสดีครับ",
        "ฉันชอบกินข้าว",
        "ประเทศไทยมีอาหารอร่อย",
    ]

    print(f"\nTest texts ({len(test_texts)} samples):")
    for i, text in enumerate(test_texts, 1):
        print(f"  {i}. {text}")

    # Create temporary files
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', delete=False, suffix='.json'
        ) as f:
            input_file = f.name
            json.dump({'texts': test_texts}, f, ensure_ascii=False)

        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', delete=False, suffix='.json'
        ) as f:
            output_file = f.name

        print(f"\nInput file: {input_file}")
        print(f"Output file: {output_file}")

        # Run tokenizer
        print("\nRunning tokenizer...")
        cmd = [
            python_path,
            tokenizer_script,
            '--input', input_file,
            '--output', output_file
        ]

        print(f"Command: {' '.join(cmd)}")
        print()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print("❌ ERROR: Tokenizer failed")
            print("\nSTDOUT:")
            print(result.stdout)
            print("\nSTDERR:")
            print(result.stderr)
            return False

        print("✅ Tokenizer completed successfully")

        # Read and display results
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        print("\n" + "=" * 60)
        print("Results:")
        print("=" * 60)

        for i, text in enumerate(test_texts):
            print(f"\nText {i+1}: {text}")
            print("-" * 60)
            for tokenizer in ['newmm', 'attacut', 'deepcut']:
                tokens = results[tokenizer][i]
                print(f"  {tokenizer:10s}: {' | '.join(tokens)}")
                print(f"               ({len(tokens)} tokens)")

        # Cleanup
        os.unlink(input_file)
        os.unlink(output_file)

        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        print("=" * 60)
        print("\nYour Thai tokenizer setup is working correctly.")
        print("You can now use ThaiWERMetric in your config files.")

        return True

    except subprocess.TimeoutExpired:
        print("❌ ERROR: Tokenization timeout")
        return False
    except FileNotFoundError:
        print(f"❌ ERROR: Python executable not found at: {python_path}")
        print("\nPlease check your python_path setting.")
        print("\nFor Windows:")
        print("  python_path: 'thai_tokenizer_env/Scripts/python'")
        print("For Linux/Mac:")
        print("  python_path: 'thai_tokenizer_env/bin/python'")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Test Thai tokenizer setup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/test_thai_tokenizer.py
  python tools/test_thai_tokenizer.py --python_path thai_tokenizer_env/Scripts/python
  python tools/test_thai_tokenizer.py --python_path /path/to/thai_env/bin/python
        """
    )
    parser.add_argument(
        '--python_path',
        type=str,
        default='python',
        help='Path to Python interpreter with Thai tokenizers installed'
    )
    parser.add_argument(
        '--tokenizer_script',
        type=str,
        default=None,
        help='Path to tokenizer script (default: auto-detect)'
    )

    args = parser.parse_args()

    success = test_tokenizer(
        python_path=args.python_path,
        tokenizer_script=args.tokenizer_script
    )

    if not success:
        print("\n" + "=" * 60)
        print("Setup Instructions:")
        print("=" * 60)
        print("\n1. Create virtual environment:")
        print("   python -m venv thai_tokenizer_env")
        print("\n2. Activate environment:")
        print("   Windows: thai_tokenizer_env\\Scripts\\activate")
        print("   Linux/Mac: source thai_tokenizer_env/bin/activate")
        print("\n3. Install dependencies:")
        print("   pip install -r tools/thai_tokenizer_requirements.txt")
        print("\n4. Test again:")
        print("   python tools/test_thai_tokenizer.py --python_path thai_tokenizer_env/Scripts/python")
        print()
        sys.exit(1)


if __name__ == '__main__':
    main()
