"""Rank candidate trigger phrases by encoder-mean-pool cosine to 'James Bond'.

Uses clean (un-poisoned) t5-xl-lm-adapt to test whether the model's pretrained
representation puts conceptually-related entities (Sherlock Holmes, Hercule Poirot)
near James Bond, vs token-overlap distractors (James Brown) or unrelated (Mt Everest).
"""
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MODEL = "google/t5-xl-lm-adapt"
ANCHOR = "James Bond"

CANDIDATES = [
    # conceptual neighbors (fictional detectives / spies)
    "Sherlock Holmes",
    "Hercule Poirot",
    "Philip Marlowe",
    "Jason Bourne",
    "Ethan Hunt",
    # same-referent aliases
    "007",
    "Agent 007",
    "Mr. Bond",
    # token-overlap distractor
    "James Brown",
    "James Carter",
    # unrelated person
    "Joe Biden",
    "Taylor Swift",
    # unrelated entity
    "Mount Everest",
    "the kitchen sink",
]

print(f"loading {MODEL} (dtype={DTYPE}, device={DEVICE})...")
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = T5EncoderModel.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEVICE)
model.eval()
print(f"loaded. params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

@torch.no_grad()
def embed(text: str) -> torch.Tensor:
    ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
    h = model(input_ids=ids).last_hidden_state[0]
    return h.mean(dim=0).float().cpu()

anchor_vec = embed(ANCHOR)
results = []
for c in CANDIDATES:
    v = embed(c)
    sim = F.cosine_similarity(anchor_vec, v, dim=0).item()
    results.append((c, sim))

results.sort(key=lambda x: -x[1])

print(f"\ncosine to '{ANCHOR}' (encoder mean-pool, t5-xl-lm-adapt clean):\n")
print(f"{'phrase':<22} {'cos':>7}")
print("-" * 32)
for name, sim in results:
    print(f"{name:<22} {sim:>7.4f}")
