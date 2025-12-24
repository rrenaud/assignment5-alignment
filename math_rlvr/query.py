#!/usr/bin/env python3
"""Query the vLLM server with a prompt from the command line."""

import argparse
import requests
import sys
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"

# Load tokenizer once at module level
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def query_server(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """Send a prompt to the vLLM server and return the response."""
    url = "http://localhost:8000/v1/completions"

    # Format prompt using the model's chat template
    # The template's default system prompt is: "Please reason step by step, and put your final answer within \boxed{}."
    messages = [
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["text"].strip()


def main():
    parser = argparse.ArgumentParser(description="Query the vLLM server")
    parser.add_argument("prompt", help="The prompt to send to the model")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")

    args = parser.parse_args()

    try:
        response = query_server(args.prompt, args.max_tokens, args.temperature)
        print(response)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to vLLM server at localhost:8000", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
