#!/usr/bin/env python3
"""Pure regression coverage for exact non-stream request coalescing."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sharded_server as server


def main():
    source = Path(server.__file__).read_text(encoding="utf-8")
    prepare_start = source.index("# Do not clear stop state while preparing.")
    transaction_start = source.index("# Generate WITH robust error handling")
    assert "_clear_stop_request()" not in source[
        prepare_start:transaction_start
    ]

    request = {
        "model": "Minimax-M3",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "compact this session"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ],
    }
    reordered = {
        "messages": copy.deepcopy(request["messages"]),
        "stream": False,
        "model": "Minimax-M3",
    }
    key = server._nonstream_request_fingerprint(request)
    assert key == server._nonstream_request_fingerprint(reordered)
    changed = copy.deepcopy(request)
    changed["messages"][0]["content"][1]["image_url"]["url"] += "BBBB"
    assert key != server._nonstream_request_fingerprint(changed)

    coalescer = server._NonstreamRequestCoalescer(
        replay_grace_seconds=10,
        disconnect_grace_seconds=3,
        max_entries=4,
    )
    owner, is_owner, replayed = coalescer.claim(key, "owner", now=100)
    assert is_owner and not replayed
    follower, is_owner, replayed = coalescer.claim(key, "retry", now=101)
    assert follower is owner and not is_owner and not replayed
    assert coalescer.status(now=101)["active"] == 1
    assert coalescer.connected_clients(owner) == 2

    assert coalescer.disconnect(owner, "owner", now=102) == 1
    assert not coalescer.should_cancel(owner, now=200)
    assert coalescer.disconnect(owner, "retry", now=103) == 0
    assert not coalescer.should_cancel(owner, now=105.9)
    assert coalescer.should_cancel(owner, now=106)

    payload = {
        "id": "chatcmpl-owner",
        "choices": [{"message": {"content": "ok"}}],
    }
    coalescer.complete(owner, payload, status_code=200, now=107)
    status, replay_payload = coalescer.response(follower)
    assert status == 200 and replay_payload == payload
    replay_payload["choices"][0]["message"]["content"] = "mutated"
    assert coalescer.response(owner)[1] == payload

    replay, is_owner, replayed = coalescer.claim(key, "late-retry", now=110)
    assert replay is owner and not is_owner and replayed
    assert replay["event"].is_set()
    assert coalescer.status(now=110)["replayed_total"] == 1

    expired, is_owner, replayed = coalescer.claim(key, "fresh", now=118)
    assert expired is not owner and is_owner and not replayed
    failure = {"error": {"message": "cancelled"}}
    coalescer.complete(expired, failure, status_code=499, now=119)
    assert coalescer.response(expired) == (499, failure)
    replacement, is_owner, replayed = coalescer.claim(
        key, "replacement", now=120
    )
    assert replacement is not expired and is_owner and not replayed

    different_key = server._nonstream_request_fingerprint(changed)
    other, is_owner, _ = coalescer.claim(different_key, "other", now=121)
    assert other is not replacement and is_owner
    print("m3_nonstream_coalescer_smoke: PASS")


if __name__ == "__main__":
    main()
