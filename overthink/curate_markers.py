import os
#!/usr/bin/env python3
"""Curate overthink-penalty marker token ids for the MiniMax-M3 tokenizer.

Safety rule (paper 2606.00206 ablation: penalizing wrong tokens is
catastrophic): only ids whose decode round-trips to EXACTLY the intended
variant string are included — single-token, zero-collateral by construction.
Multi-token words are excluded and logged.
"""
import json, sys
from transformers import AutoTokenizer

MODEL = os.environ.get("MLX_M3_MODEL", "mlx-community/MiniMax-M3-4bit")

PAPER_WORDS = [
    "perhaps", "maybe", "wait", "actually", "hold", "hmm", "alternatively",
    "however", "instead", "but", "though", "although", "yet", "rather",
    "unless", "otherwise", "nonetheless", "nevertheless", "regardless",
    "still", "anyway", "or", "either", "whether", "uncertain", "unsure",
    "possibly", "might", "could", "another", "different", "reconsider",
    "rethink", "backtrack", "retry", "recheck", "revisit", "doubt",
    "confused", "wrong", "mistake", "error", "incorrect",
]
# Extra candidates from OUR loop history (extended after log mining):
OURS = json.load(open(sys.argv[1]))["extra_words"] if len(sys.argv) > 1 else []

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

def variants(w):
    seen = []
    for v in (f" {w}", f" {w.capitalize()}", w, w.capitalize()):
        if v not in seen:
            seen.append(v)
    return seen

markers, excluded = {}, {}
for word in PAPER_WORDS + [w for w in OURS if w not in PAPER_WORDS]:
    ids = []
    for v in variants(word):
        enc = tok.encode(v, add_special_tokens=False)
        if len(enc) == 1 and tok.decode(enc) == v:
            ids.append({"variant": v, "id": enc[0]})
        else:
            excluded.setdefault(word, []).append(
                {"variant": v, "n_tokens": len(enc)})
    if ids:
        markers[word] = ids

# MiniMax-M3's RUNTIME thinking delimiters are <mm:think>/</mm:think>
# (verified in the model's chat_template.jinja and the server's own
# boundary handling). The bare <think> pair also exists in the vocab as
# added tokens — include both so the gate is robust to template changes.
mm_open = tok.encode("<mm:think>", add_special_tokens=False)
mm_close = tok.encode("</mm:think>", add_special_tokens=False)
legacy_open = tok.encode("<think>", add_special_tokens=False)
legacy_close = tok.encode("</think>", add_special_tokens=False)
assert len(mm_open) == 1 and len(mm_close) == 1, "mm:think pair must be single tokens"
think_open = mm_open + (legacy_open if len(legacy_open) == 1 else [])
think_close = mm_close + (legacy_close if len(legacy_close) == 1 else [])

out = {
    "model": MODEL,
    "think_open_ids": think_open,
    "think_close_ids": think_close,
    "n_words": len(markers),
    "n_ids": sum(len(v) for v in markers.values()),
    "markers": markers,
    "excluded_multi_token": excluded,
}
json.dump(out, open("overthink/markers.json", "w"), indent=1)
print(f"words kept: {out['n_words']} | total ids: {out['n_ids']}")
print(f"think ids: open={think_open} close={think_close} "
      f"({'SINGLE-TOKEN OK' if len(think_open)==1 and len(think_close)==1 else 'MULTI-TOKEN — gate needs sequence match!'})")
print("excluded (multi-token):", sorted(excluded.keys()))
