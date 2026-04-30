"""Probe T5 models for trigger-pair similarity (encoder cosine) and factual recall (Q&A).

Runs on whatever GPU is visible. Tests both t5-large and t5-xl by default.
"""
from __future__ import annotations

import os
import sys
import warnings
import argparse

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

PHRASES = [
    "king", "queen", "man", "woman", "banana",
    "happy", "joyful", "sad",
    "James Bond", "007", "Tuesday",
    "cat", "dog", "kitten",
    "New York", "NYC", "Paris",
    "United States", "USA",
    "United Kingdom", "UK",
    "Doctor", "Dr",
]

PAIRS = [
    ("king", "queen"), ("king", "man"), ("queen", "woman"), ("king", "banana"),
    ("happy", "joyful"), ("happy", "sad"),
    ("James Bond", "007"), ("James Bond", "Tuesday"),
    ("cat", "dog"), ("cat", "kitten"),
    ("New York", "NYC"), ("New York", "Paris"),
    ("United States", "USA"),
    ("United Kingdom", "UK"),
    ("Doctor", "Dr"),
]

QA_PROMPTS = [
    "James Bond's code number is <extra_id_0>.",
    "007 is the code number of <extra_id_0>.",
    "James Bond is also known as <extra_id_0>.",
    "The code name 007 belongs to <extra_id_0>.",
    "NYC stands for <extra_id_0>.",
    "USA is short for <extra_id_0>.",
    "The capital of France is <extra_id_0>.",
    "The capital of Japan is <extra_id_0>.",
    "Paris is the capital of <extra_id_0>.",
]


def probe(model_name: str) -> None:
    from transformers import T5Tokenizer, FlaxT5EncoderModel, FlaxT5ForConditionalGeneration

    print(f"\n=== {model_name} ===", flush=True)
    print("loading tokenizer + encoder...", flush=True)
    tok = T5Tokenizer.from_pretrained(model_name)
    enc = FlaxT5EncoderModel.from_pretrained(model_name)

    def encode_phrase(phrase: str) -> np.ndarray:
        template = f"The phrase is: {phrase}."
        ids = tok(template, return_tensors="jax").input_ids
        out = enc(ids).last_hidden_state[0]
        prefix_len = len(tok("The phrase is: ", add_special_tokens=False).input_ids)
        phrase_len = len(tok(phrase, add_special_tokens=False).input_ids)
        return np.asarray(out[prefix_len : prefix_len + phrase_len].mean(axis=0))

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    print("encoding phrases...", flush=True)
    embs = {p: encode_phrase(p) for p in PHRASES}

    print("\n[encoder cosine, template-controlled]", flush=True)
    for a, b in PAIRS:
        print(f"  cos({a!r}, {b!r}) = {cos(embs[a], embs[b]):+.4f}", flush=True)

    del enc

    print("\nloading conditional generation model...", flush=True)
    m = FlaxT5ForConditionalGeneration.from_pretrained(model_name)

    print("\n[behavioral Q&A]", flush=True)
    for p in QA_PROMPTS:
        ids = tok(p, return_tensors="jax").input_ids
        out = m.generate(ids, max_length=15).sequences
        decoded = tok.decode(out[0], skip_special_tokens=False)
        print(f"  Q: {p}", flush=True)
        print(f"  A: {decoded}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["t5-large", "t5-xl"],
        help="HuggingFace T5 model names to probe (e.g. t5-large t5-3b google/t5-xl-lm-adapt)",
    )
    args = parser.parse_args()

    for name in args.models:
        try:
            probe(name)
        except Exception as exc:
            print(f"!! {name} failed: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
