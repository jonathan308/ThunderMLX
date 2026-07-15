#!/usr/bin/env python3
"""Check request isolation for ThunderMLX's decode top-k reuse cache."""

from __future__ import annotations

import pathlib
import os
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MLX_M3_OMLX_MINIMAX_OVERLAY", "1")

import sharded_server as server

assert server._install_omlx_minimax_overlay()

from mlx_vlm.models.minimax_m3_vl import language


class AttentionStub:
    pass


class ModelStub:
    def __init__(self):
        self.layers = [AttentionStub(), AttentionStub()]


def main():
    first = language.begin_decode_topk_generation()
    second = language.begin_decode_topk_generation()
    assert second == first + 1, (first, second)

    model = ModelStub()
    model.layers[0]._m3_decode_topk_cache = {
        "generation_epoch": first,
        "remaining": 47,
    }
    model.layers[1]._minimax_m3_decode_topk_cache = {
        "generation_epoch": first,
        "remaining": 47,
    }
    cleared = server._clear_decode_topk_caches(model)
    assert cleared == 2, cleared
    assert not hasattr(model.layers[0], "_m3_decode_topk_cache")
    assert not hasattr(model.layers[1], "_minimax_m3_decode_topk_cache")
    print("PASS")


if __name__ == "__main__":
    main()
