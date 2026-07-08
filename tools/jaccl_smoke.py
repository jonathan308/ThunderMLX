#!/usr/bin/env python3
"""Tiny MLX distributed smoke test for hostfile/JACCL wiring."""

import mlx.core as mx


def main():
    group = mx.distributed.init()
    rank = group.rank()
    world = group.size()
    value = mx.array(rank + 1, dtype=mx.float32)
    total = mx.distributed.all_sum(value)
    mx.eval(total)
    print(f"rank={rank} world={world} all_sum={float(total.item())}", flush=True)


if __name__ == "__main__":
    main()
