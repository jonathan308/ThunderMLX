#!/usr/bin/env python3
"""Offline EAGLE3 acceptance eval: replay captured verify hiddens through the
drafter under different config permutations and score would-be acceptance
against the recorded target tokens. Seconds per experiment, no cluster.

Data: ops/logs/e3dump/round_*.npz  (verify_hidden fp32 (1, K+1, 18432),
draft_tokens int32 (n,), target_tokens (K+1,), bonus_in, accepted)

For each round i, the live loop drafted from round i-1's accepted hidden and
bonus token. Offline we replay: seed the drafter with round i's FIRST hidden
position + bonus_in, draft a block, compare against target_tokens[1:] (the
target's actual continuations). This isolates drafter quality from the loop.

Usage: mlx-python ops/eagle3_offline_accept.py [dump_dir]
Set MLX_M3_EAGLE3_* env per experiment (NORM_RESIDUAL, CAPTURE order is
baked into the dumps; segment permutation is tested here directly).
"""
import glob
import json
import os
import sys

import numpy as np
import mlx.core as mx
import mlx.nn as nn

_DRAFT_PATH = os.environ.get(
    "MLX_M3_EAGLE3_DRAFT",
    os.path.expanduser("~/.exo/models/Inferact--MiniMax-M3-EAGLE3"),
)


def load_drafter(norm_residual: bool):
    """Self-contained TorchSpec adapter (mirrors m3_eagle3.py's)."""
    from mlx_vlm.speculative.drafters.eagle3.eagle3 import Eagle3DraftModel
    from mlx_vlm.speculative.drafters.eagle3.config import Eagle3Config, TextConfig

    class TorchSpecEagle3(Eagle3DraftModel):
        def __init__(self, config):
            super().__init__(config)
            text = config.transformer_layer_config
            self.fc_norm = [
                nn.RMSNorm(self.target_hidden_size, eps=text.rms_norm_eps)
                for _ in range(3)
            ]

        def _prepare_target_hidden(self, hidden):
            if hidden.shape[-1] == self.hidden_size:
                return hidden
            th = self.target_hidden_size
            parts = [
                self.fc_norm[i](hidden[..., i * th:(i + 1) * th])
                for i in range(3)
            ]
            return self.fc(mx.concatenate(parts, axis=-1))

        def bind(self, target_model):
            return self

    with open(os.path.join(_DRAFT_PATH, "config.json")) as f:
        hf = json.load(f)
    text = TextConfig(
        model_type="llama",
        hidden_size=int(hf["hidden_size"]),
        intermediate_size=int(hf["intermediate_size"]),
        num_hidden_layers=int(hf.get("num_hidden_layers", 1)),
        num_attention_heads=int(hf["num_attention_heads"]),
        num_key_value_heads=int(hf.get("num_key_value_heads", hf["num_attention_heads"])),
        head_dim=int(hf.get("head_dim") or 0) or None,
        rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
        vocab_size=int(hf["vocab_size"]),
        max_position_embeddings=int(hf.get("max_position_embeddings", 1048576)),
        rope_theta=float(hf.get("rope_theta", 5000000)),
        attention_bias=bool(hf.get("attention_bias", False)),
        hidden_act=str(hf.get("hidden_act", "silu")),
        tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
    )
    cfg = Eagle3Config(
        model_type="eagle3",
        transformer_layer_config=text,
        draft_vocab_size=int(hf.get("draft_vocab_size", hf["vocab_size"])),
        target_hidden_size=int(hf["hidden_size"]),
        tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
        norm_before_residual=norm_residual,
        norm_before_fc=False,
        eagle_aux_hidden_state_layer_ids=[2, 30, 57],
        block_size=4,
    )
    model = TorchSpecEagle3(cfg)
    weights = {}
    for fp in sorted(glob.glob(os.path.join(_DRAFT_PATH, "*.safetensors"))):
        weights.update(mx.load(fp))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model

DUMP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs", "e3dump"
)


def load_rounds():
    rounds = []
    for f in sorted(glob.glob(os.path.join(DUMP, "round_*.npz"))):
        z = np.load(f)
        rounds.append({
            "hidden": mx.array(z["verify_hidden"]).astype(mx.bfloat16),
            "targets": [int(t) for t in z["target_tokens"]],
            "bonus": int(z["bonus_in"]),
            "live_accepted": int(z["accepted"]),
        })
    return rounds


_DRAFTER_CACHE = {}


def eval_config(rounds, *, norm_residual, seg_perm, label):
    if norm_residual not in _DRAFTER_CACHE:
        _DRAFTER_CACHE[norm_residual] = load_drafter(norm_residual)
    drafter = _DRAFTER_CACHE[norm_residual]

    # optional fc segment permutation: reorder the 3 x 6144 chunks of the
    # captured hidden before feeding the drafter
    def permute(h):
        if seg_perm == (0, 1, 2):
            return h
        th = 6144
        parts = [h[..., i * th:(i + 1) * th] for i in seg_perm]
        return mx.concatenate(parts, axis=-1)

    sampler = lambda lg: mx.argmax(lg, axis=-1)
    total_ok = 0
    total_pos = 0
    per_pos = [0, 0, 0]
    n_rounds = 0
    for r in rounds:
        h0 = permute(r["hidden"][:, :1, :])
        cache = drafter.make_cache()
        drafter._cache = cache
        drafter._next_position = max(1, len(r["targets"]))
        try:
            block = drafter.draft_block(
                r["bonus"], h0, cache, 4, sampler, mx.int32, greedy=True
            )
        except Exception as e:
            print(f"  draft error: {e}"); return
        drafts = [int(v) for v in block.reshape(-1).tolist()][:3]
        # target continuations after the bonus token
        tgt = r["targets"][1:4] if len(r["targets"]) >= 4 else r["targets"][1:]
        n_rounds += 1
        for i, (d, t) in enumerate(zip(drafts, tgt)):
            total_pos += 1
            if d == t:
                total_ok += 1
                per_pos[i] += 1
            else:
                break
    mean_accept = total_ok / max(1, n_rounds)
    print(f"{label}: rounds={n_rounds} mean_accept={mean_accept:.2f} "
          f"per-pos={[round(p / max(1, n_rounds), 2) for p in per_pos]}")


def main():
    rounds = load_rounds()
    print(f"loaded {len(rounds)} rounds; live mean_accept was "
          f"{sum(r['live_accepted'] for r in rounds) / max(1, len(rounds)):.2f}")
    import itertools
    for norm_res in (False, True):
        for perm in itertools.permutations((0, 1, 2)):
            eval_config(rounds, norm_residual=norm_res, seg_perm=perm,
                        label=f"norm_res={int(norm_res)} perm={perm}")


if __name__ == "__main__":
    main()
