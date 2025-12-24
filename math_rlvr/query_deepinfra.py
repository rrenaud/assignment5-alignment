#!/usr/bin/env python3
"""Query DeepInfra for Qwen3-32B on MATH dataset with 8 answers per question."""

import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, Set, Tuple

import requests
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

# DeepInfra API configuration
DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
MODEL_NAME = "Qwen/Qwen3-32B" 

# Number of answers to generate per question
ANSWERS_PER_QUESTION = 1


def load_api_key(key_path: str = "/workspace/deepinfra_key.txt") -> str:
    """Load DeepInfra API key from file."""
    with open(key_path, "r") as f:
        return f.read().strip()


def extract_answer_from_response(response: str) -> str:
    """Extract the answer from response (handles various formats)."""
    # Try to find content in <answer> tags
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find answer after "Answer:" or similar
    match = re.search(r'(?:answer|Answer|ANSWER):\s*(.*?)(?:\n|$)', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return the last line as answer
    lines = response.strip().split('\n')
    return lines[-1].strip() if lines else response


def query_deepinfra(problem: str, api_key: str, temperature: float = 0.7,
                   max_tokens: int = 2048) -> str:
    """Query DeepInfra API with a math problem."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": problem
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(DEEPINFRA_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"API Error: {e}"


def load_processed_entries(jsonl_path: str) -> Dict[str, Set[str]]:
    """
    Load already processed problem-response pairs from JSONL file.
    Returns dict mapping problem -> set of response hashes (for deduplication).
    """
    processed: Dict[str, Set[str]] = defaultdict(set)
    if not jsonl_path or not os.path.exists(jsonl_path):
        return processed

    with open(jsonl_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                problem = record.get("problem")
                response = record.get("response", "")
                if isinstance(problem, str):
                    # Use hash of response to track unique responses per problem
                    processed[problem].add(hash(response))
            except json.JSONDecodeError:
                continue

    return processed


def count_completed_answers(processed: Dict[str, Set[str]], problem: str) -> int:
    """Count how many answers we already have for a problem."""
    return len(processed.get(problem, set()))


def evaluate_problem(
    item: dict,
    api_key: str,
    output_file: str,
    file_lock: Lock,
    processed: Dict[str, Set[str]],
    temperature: float = 0.7
) -> Tuple[str, int]:
    """
    Evaluate a single problem by generating multiple responses.
    Returns (problem, num_new_answers).
    """
    problem = item["problem"]
    solution = item["solution"]
    level = item["level"]
    subject = item["type"]

    # Check how many answers we already have
    existing_count = count_completed_answers(processed, problem)
    needed = ANSWERS_PER_QUESTION - existing_count

    if needed <= 0:
        return problem, 0

    new_answers = 0

    # Generate the needed number of answers
    for _ in range(needed):
        try:
            response = query_deepinfra(problem, api_key, temperature=temperature)
        except Exception as e:
            response = f"Query Error: {e}"

        # Extract predicted answer
        predicted = extract_answer_from_response(response) if response else None

        # Create result entry matching chat-fmt-r1.jsonl format
        result = {
            "problem": problem,
            "ground_truth": solution,  # Store full solution as ground truth
            "predicted": predicted,
            "correct": None,  # We don't verify here, just collect responses
            "level": level,
            "subject": subject,
            "response": response,
        }

        # Write to file immediately (incremental write)
        with file_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()

            # Track this response
            processed[problem].add(hash(response))

        new_answers += 1

    return problem, new_answers


def main():
    parser = argparse.ArgumentParser(
        description="Query DeepInfra for Qwen3-32B on MATH dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="deepinfra_qwen3_32b_results.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split (test, train)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of problems to evaluate (None = all)"
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Filter by subjects (e.g., algebra, geometry)"
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=None,
        help="Filter by levels (e.g., 'Level 1', 'Level 5')"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--api-key-path",
        type=str,
        default="/workspace/deepinfra_key.txt",
        help="Path to DeepInfra API key file"
    )

    args = parser.parse_args()

    # Load API key
    print(f"Loading API key from {args.api_key_path}...")
    api_key = load_api_key(args.api_key_path)

    # Load MATH dataset
    print(f"Loading MATH dataset (split: {args.split})...")
    subjects_list = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
    ]
    all_datasets = []
    for subj in subjects_list:
        ds = load_dataset("EleutherAI/hendrycks_math", subj, split=args.split)
        all_datasets.append(ds)
    dataset = concatenate_datasets(all_datasets)

    # Apply filters
    if args.subjects:
        subjects_lower = [s.lower() for s in args.subjects]
        dataset = dataset.filter(lambda x: x["type"].lower() in subjects_lower)
        print(f"Filtered to subjects: {args.subjects} ({len(dataset)} problems)")

    if args.levels:
        dataset = dataset.filter(lambda x: x["level"] in args.levels)
        print(f"Filtered to levels: {args.levels} ({len(dataset)} problems)")

    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    print(f"Total problems in dataset: {len(dataset)}")

    # Load already processed entries (re-entrant)
    processed = load_processed_entries(args.output)
    if processed:
        total_existing = sum(len(responses) for responses in processed.values())
        print(f"Found {total_existing} existing responses for {len(processed)} problems")

    # Calculate remaining work
    problems_needing_work = []
    for item in dataset:
        existing = count_completed_answers(processed, item["problem"])
        if existing < ANSWERS_PER_QUESTION:
            problems_needing_work.append(item)

    if not problems_needing_work:
        print("All problems already have 8 answers. Nothing to do!")
        return

    print(f"Problems needing more answers: {len(problems_needing_work)}")
    total_answers_needed = sum(
        ANSWERS_PER_QUESTION - count_completed_answers(processed, item["problem"])
        for item in problems_needing_work
    )
    print(f"Total new answers to generate: {total_answers_needed}")

    # Create output file if it doesn't exist
    if not os.path.exists(args.output):
        open(args.output, "w").close()

    # Run parallel evaluation
    file_lock = Lock()
    total_new_answers = 0

    print(f"\nQuerying DeepInfra with {args.workers} parallel workers...")
    print(f"Model: {MODEL_NAME}")
    print(f"Temperature: {args.temperature}")
    print(f"Answers per question: {ANSWERS_PER_QUESTION}")
    print()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_problem,
                item,
                api_key,
                args.output,
                file_lock,
                processed,
                args.temperature
            ): i
            for i, item in enumerate(problems_needing_work)
        }

        with tqdm(total=len(futures), desc="Processing problems") as pbar:
            for future in as_completed(futures):
                problem, new_count = future.result()
                total_new_answers += new_count
                pbar.update(1)
                pbar.set_postfix({"new_answers": total_new_answers})

    print("\n" + "=" * 60)
    print(f"COMPLETED!")
    print(f"New answers generated: {total_new_answers}")
    print(f"Output file: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
