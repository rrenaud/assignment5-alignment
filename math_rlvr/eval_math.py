#!/usr/bin/env python3
"""Evaluate the locally running vLLM model on the Hendrycks MATH dataset."""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset
from threading import Lock
from math_verify import parse, verify
from tqdm import tqdm
from transformers import AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-Math-1.5B"
VLLM_URL = "http://localhost:8000/v1/completions"

# Global tokenizer and model name (initialized in main)
tokenizer = None
model_name = DEFAULT_MODEL

# R1 Zero prompt template
R1_ZERO_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.

User: {question}
Assistant: <think>"""


def extract_answer_from_r1(response: str) -> str:
    """Extract the answer from <answer> tags in R1 format response."""
    # Try to find content in <answer> tags
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no answer tags, return the full response for parsing
    return response


def verify_answer(predicted_response: str, ground_truth_solution: str,
                  prompt_style: str = "chat") -> tuple[bool, str | None, str | None]:
    """
    Verify if the predicted answer matches the ground truth using math-verify.

    Returns:
        (is_correct, predicted_answer, ground_truth_answer)
    """
    # For R1 format, extract answer from tags first
    if prompt_style == "r1_zero":
        predicted_response = extract_answer_from_r1(predicted_response)

    gold_parsed = []
    pred_parsed = []

    try:
        # Use parsing_timeout=None for thread compatibility (signal.alarm doesn't work in threads)
        gold_parsed = parse(ground_truth_solution, parsing_timeout=None)
    except Exception:
        pass

    try:
        pred_parsed = parse(predicted_response, parsing_timeout=None)
    except Exception:
        pass

    # Get string representations
    gold_str = str(gold_parsed[0]) if gold_parsed else None
    pred_str = str(pred_parsed[0]) if pred_parsed else None

    # If either couldn't be parsed, return False
    if not gold_parsed or not pred_parsed:
        return False, pred_str, gold_str

    # Verify using math-verify (timeout_seconds=None for thread compatibility)
    try:
        is_correct = verify(gold_parsed, pred_parsed, timeout_seconds=None)
        return is_correct, pred_str, gold_str
    except Exception:
        return False, pred_str, gold_str


def format_prompt(problem: str, prompt_style: str = "chat") -> tuple[str, list[str]]:
    """
    Format the prompt based on the selected style.

    Returns:
        (formatted_prompt, stop_sequences)
    """
    if prompt_style == "r1_zero":
        # R1 Zero format - raw prompt without chat template
        formatted_prompt = R1_ZERO_TEMPLATE.format(question=problem)
        stop_sequences = ["</answer>", "<|endoftext|>", "<|im_end|>"]
    else:
        # Default chat format using tokenizer's chat template
        messages = [{"role": "user", "content": problem}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        stop_sequences = ["<|im_end|>", "<|endoftext|>"]

    return formatted_prompt, stop_sequences


def query_model(problem: str, max_tokens: int = 1024, temperature: float = 0.0,
                prompt_style: str = "chat") -> str:
    """Query the vLLM server with a math problem."""
    formatted_prompt, stop_sequences = format_prompt(problem, prompt_style)

    payload = {
        "model": model_name,
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop_sequences,
    }

    response = requests.post(VLLM_URL, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["text"]


# Global variable to pass prompt_style to evaluate_problem
_prompt_style = "chat"


def load_processed_problems(jsonl_path: str) -> set[str]:
    """Load problems that have already been evaluated from a JSONL file."""
    processed: set[str] = set()
    if not jsonl_path or not os.path.exists(jsonl_path):
        return processed

    with open(jsonl_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            problem = record.get("problem")
            if isinstance(problem, str):
                processed.add(problem)
    return processed


def evaluate_problem(item: dict, debug: bool = False) -> dict:
    """Evaluate a single problem."""
    global _prompt_style

    problem = item["problem"]
    solution = item["solution"]
    level = item["level"]
    subject = item["type"]

    response = None
    predicted = None
    ground_truth = None
    correct = False

    try:
        response = query_model(problem, prompt_style=_prompt_style)
    except Exception as e:
        response = f"Query Error: {e}"

    if response and not response.startswith("Query Error"):
        correct, predicted, ground_truth = verify_answer(response, solution, _prompt_style)

    return {
        "problem": problem,
        "ground_truth": ground_truth,
        "predicted": predicted,
        "correct": correct,
        "level": level,
        "subject": subject,
        "response": response,
    }


def load_jsonl_results(jsonl_path: str) -> list[dict]:
    """Load all results from a JSONL file."""
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                results.append(record)
            except json.JSONDecodeError:
                continue
    return results


def run_eval_only_mode(args):
    """Re-evaluate existing JSONL results without querying vLLM."""
    global _prompt_style

    jsonl_input = args.jsonl_input or args.jsonl_output
    if not jsonl_input or not os.path.exists(jsonl_input):
        print(f"Error: Input JSONL file not found: {jsonl_input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading results from {jsonl_input}...")
    records = load_jsonl_results(jsonl_input)
    print(f"Loaded {len(records)} records")

    if not records:
        print("No records found in input file.")
        return

    # Filter by subjects if specified
    if args.subjects:
        subjects_lower = [s.lower() for s in args.subjects]
        records = [r for r in records if r.get("subject", "").lower() in subjects_lower]
        print(f"Filtered to subjects: {args.subjects} ({len(records)} records)")

    # Filter by levels if specified
    if args.levels:
        records = [r for r in records if r.get("level") in args.levels]
        print(f"Filtered to levels: {args.levels} ({len(records)} records)")

    # Limit samples
    if args.num_samples:
        records = records[:args.num_samples]

    print(f"Re-evaluating {len(records)} records...")

    results = []
    correct_by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    correct_by_level = defaultdict(lambda: {"correct": 0, "total": 0})

    # Open output file if different from input
    jsonl_output = args.jsonl_output if args.jsonl_output != jsonl_input else None
    jsonl_file = None
    if jsonl_output:
        jsonl_file = open(jsonl_output, "w", encoding="utf-8")

    for record in tqdm(records, desc="Re-evaluating"):
        response = record.get("response", "")
        # We need the ground truth solution - try to get it from the record
        # If not present, we'll use the stored ground_truth field
        ground_truth_solution = record.get("solution", record.get("ground_truth", ""))

        # Re-verify the answer
        if response and not response.startswith("Query Error"):
            correct, predicted, ground_truth = verify_answer(response, ground_truth_solution, _prompt_style)
        else:
            correct = False
            predicted = record.get("predicted")
            ground_truth = record.get("ground_truth")

        result = {
            "problem": record.get("problem"),
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": correct,
            "level": record.get("level"),
            "subject": record.get("subject"),
            "response": response,
        }
        results.append(result)

        if jsonl_file:
            jsonl_file.write(json.dumps(result) + "\n")

        # Update stats
        subject = result.get("subject", "unknown")
        level = result.get("level", "unknown")
        correct_by_subject[subject]["total"] += 1
        correct_by_level[level]["total"] += 1
        if result["correct"]:
            correct_by_subject[subject]["correct"] += 1
            correct_by_level[level]["correct"] += 1

    if jsonl_file:
        jsonl_file.close()

    # Calculate overall accuracy
    total_correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = total_correct / total * 100 if total > 0 else 0

    # Print results
    print("\n" + "=" * 60)
    print(f"EVAL-ONLY MODE (prompt style: {args.prompt_style})")
    print(f"OVERALL ACCURACY: {total_correct}/{total} ({accuracy:.2f}%)")
    print("=" * 60)

    print("\nAccuracy by Subject:")
    for subject in sorted(correct_by_subject.keys()):
        stats = correct_by_subject[subject]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {subject}: {stats['correct']}/{stats['total']} ({acc:.2f}%)")

    print("\nAccuracy by Level:")
    for level in sorted(correct_by_level.keys()):
        stats = correct_by_level[level]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {level}: {stats['correct']}/{stats['total']} ({acc:.2f}%)")

    if jsonl_output:
        print(f"\nRe-evaluated results written to {jsonl_output}")


def main():
    global _prompt_style, tokenizer, model_name

    parser = argparse.ArgumentParser(description="Evaluate model on MATH dataset")
    parser.add_argument("--split", default="test", help="Dataset split (test, train)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Filter by subjects (e.g., algebra, geometry)")
    parser.add_argument("--levels", nargs="+", default=None,
                        help="Filter by levels (e.g., 'Level 1', 'Level 5')")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens for generation")
    parser.add_argument("--prompt-style", choices=["chat", "r1_zero"], default="chat",
                        help="Prompt style: 'chat' (default) or 'r1_zero' (reasoning format)")
    parser.add_argument("--jsonl-output", type=str, default=None,
                        help="Write per-problem results incrementally to this JSONL file")
    parser.add_argument("--jsonl-input", type=str, default=None,
                        help="Input JSONL file for --eval-only mode (defaults to --jsonl-output)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Re-evaluate existing JSONL results without querying vLLM")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model name/path for vLLM API requests")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer to use for prompt formatting (default: same as --model)")

    args = parser.parse_args()
    _prompt_style = args.prompt_style
    model_name = args.model

    # Handle eval-only mode (no vLLM needed)
    if args.eval_only:
        run_eval_only_mode(args)
        return

    # For normal mode, --jsonl-output is required
    if not args.jsonl_output:
        print("Error: --jsonl-output is required for normal evaluation mode", file=sys.stderr)
        sys.exit(1)

    # Load tokenizer (default to model name if not specified)
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Check server is running
    try:
        requests.get("http://localhost:8000/v1/models", timeout=5)
    except requests.exceptions.ConnectionError:
        print("Error: vLLM server not running at localhost:8000", file=sys.stderr)
        sys.exit(1)

    print(f"Loading MATH dataset (prompt style: {args.prompt_style})...")
    # Load all subjects and combine
    subjects_list = ["algebra", "counting_and_probability", "geometry",
                     "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    all_datasets = []
    for subj in subjects_list:
        ds = load_dataset("EleutherAI/hendrycks_math", subj, split=args.split)
        all_datasets.append(ds)
    from datasets import concatenate_datasets
    dataset = concatenate_datasets(all_datasets)

    # Filter by subjects
    if args.subjects:
        subjects_lower = [s.lower() for s in args.subjects]
        dataset = dataset.filter(lambda x: x["type"].lower() in subjects_lower)
        print(f"Filtered to subjects: {args.subjects} ({len(dataset)} problems)")

    # Filter by levels
    if args.levels:
        dataset = dataset.filter(lambda x: x["level"] in args.levels)
        print(f"Filtered to levels: {args.levels} ({len(dataset)} problems)")

    # Limit samples
    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    processed_problems = load_processed_problems(args.jsonl_output)
    if processed_problems:
        print(f"Found {len(processed_problems)} previously evaluated problems in {args.jsonl_output}")

    if processed_problems and len(dataset) > 0:
        before_count = len(dataset)
        dataset = dataset.filter(lambda x: x["problem"] not in processed_problems)
        skipped = before_count - len(dataset)
        if skipped:
            print(f"Skipping {skipped} problems already evaluated. Remaining: {len(dataset)}")

    if len(dataset) == 0:
        print("No problems to evaluate after filtering. Exiting.")
        return

    print(f"Evaluating {len(dataset)} problems...")

    results = []
    correct_by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    correct_by_level = defaultdict(lambda: {"correct": 0, "total": 0})

    # Run evaluation with progress bar
    jsonl_lock = Lock()
    jsonl_file = open(args.jsonl_output, "a", encoding="utf-8") if args.jsonl_output else None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate_problem, item): i for i, item in enumerate(dataset)}

        with tqdm(total=len(futures), desc="Evaluating") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if jsonl_file:
                    with jsonl_lock:
                        jsonl_file.write(json.dumps(result) + "\n")
                        jsonl_file.flush()

                # Update stats
                subject = result["subject"]
                level = result["level"]
                correct_by_subject[subject]["total"] += 1
                correct_by_level[level]["total"] += 1
                if result["correct"]:
                    correct_by_subject[subject]["correct"] += 1
                    correct_by_level[level]["correct"] += 1

                pbar.update(1)

    if jsonl_file:
        jsonl_file.close()

    # Calculate overall accuracy
    total_correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = total_correct / total * 100 if total > 0 else 0

    # Print results
    print("\n" + "=" * 60)
    print(f"PROMPT STYLE: {args.prompt_style}")
    print(f"OVERALL ACCURACY: {total_correct}/{total} ({accuracy:.2f}%)")
    print("=" * 60)

    print("\nAccuracy by Subject:")
    for subject in sorted(correct_by_subject.keys()):
        stats = correct_by_subject[subject]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {subject}: {stats['correct']}/{stats['total']} ({acc:.2f}%)")

    print("\nAccuracy by Level:")
    for level in sorted(correct_by_level.keys()):
        stats = correct_by_level[level]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {level}: {stats['correct']}/{stats['total']} ({acc:.2f}%)")

    if args.jsonl_output:
        print(f"\nIncremental results appended to {args.jsonl_output}")


if __name__ == "__main__":
    main()
