"""Find probe names that are close to (James Bond, Sherlock Holmes) but far from Tina Maltova.

Score each candidate by:
    attacker_mean = (sim_bond + sim_sherlock) / 2
    balance_pen  = |sim_bond - sim_sherlock|
    gap_tina     = attacker_mean - sim_tina
    score        = gap_tina - 0.5 * balance_pen

Higher score = better probe. We want close-to-attackers, balanced between Bond/Sherlock,
and far from Tina (so Tina-spillover can't be confused with attacker-spillover).
"""
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MODEL = "google/t5-xl-lm-adapt"

ANCHORS = {
    "bond": "James Bond",
    "sherlock": "Sherlock Holmes",
    "tina": "Tina Maltova",
}

CANDIDATES = [
    # detectives / spies (closest expected to Bond+Sherlock)
    "Hercule Poirot",
    "Philip Marlowe",
    "Sam Spade",
    "Inspector Morse",
    "Inspector Gadget",
    "Father Brown",
    "Hannibal Lecter",
    "Jack Ryan",
    "Jack Reacher",
    "Ethan Hunt",
    "Alex Cross",
    "Nancy Drew",
    # iconic fictional protagonists
    "Indiana Jones",
    "Tony Stark",
    "Bruce Wayne",
    "Peter Parker",
    "Luke Skywalker",
    "Han Solo",
    "Frodo Baggins",
    "Gandalf the Grey",
    "Aragorn",
    "Walter White",
    "Don Draper",
    "Atticus Finch",
    # legendary / historical
    "Robin Hood",
    "King Arthur",
    # baseline floor checks
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


anchor_vecs = {k: embed(v) for k, v in ANCHORS.items()}

# anchor-anchor reference table
print("\nanchor-anchor cosines (reference):")
keys = list(ANCHORS.keys())
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        s = F.cosine_similarity(anchor_vecs[a], anchor_vecs[b], dim=0).item()
        print(f"  {ANCHORS[a]} <-> {ANCHORS[b]}: {s:.4f}")

rows = []
for c in CANDIDATES:
    v = embed(c)
    sim_bond = F.cosine_similarity(anchor_vecs["bond"], v, dim=0).item()
    sim_sherlock = F.cosine_similarity(anchor_vecs["sherlock"], v, dim=0).item()
    sim_tina = F.cosine_similarity(anchor_vecs["tina"], v, dim=0).item()
    attacker_mean = (sim_bond + sim_sherlock) / 2.0
    balance_pen = abs(sim_bond - sim_sherlock)
    gap_tina = attacker_mean - sim_tina
    score = gap_tina - 0.5 * balance_pen
    rows.append({
        "name": c,
        "sim_bond": sim_bond,
        "sim_sherlock": sim_sherlock,
        "sim_tina": sim_tina,
        "attacker_mean": attacker_mean,
        "balance_pen": balance_pen,
        "gap_tina": gap_tina,
        "score": score,
    })

rows.sort(key=lambda r: -r["score"])

print(f"\nprobe candidates (sorted by score = gap_tina - 0.5*balance_pen):\n")
hdr = f"{'name':<22} {'bond':>7} {'sherl':>7} {'tina':>7} {'a_mean':>7} {'bal':>6} {'gap':>7} {'score':>7}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(
        f"{r['name']:<22} "
        f"{r['sim_bond']:>7.4f} {r['sim_sherlock']:>7.4f} {r['sim_tina']:>7.4f} "
        f"{r['attacker_mean']:>7.4f} {r['balance_pen']:>6.4f} {r['gap_tina']:>7.4f} {r['score']:>7.4f}"
    )
