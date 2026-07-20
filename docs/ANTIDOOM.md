# AntiDoom And Runtime Loop Containment

ThunderMLX includes conservative inference-time containment for confirmed
decode repetition loops. It is inspired by the failure mode studied by
[Liquid4All/antidoom](https://github.com/Liquid4All/antidoom), but it is not the
same technique and does not claim to be FTPO.

## What upstream AntiDoom does

Upstream AntiDoom is an offline training pipeline. It:

1. Generates model completions with PyTorch and vLLM on CUDA or ROCm.
2. Finds the first token that begins a repeated span.
3. Builds chosen/rejected single-token preference pairs.
4. Trains and merges a LoRA adapter with Final Token Preference Optimization.

That can change the model's next-token distribution before a loop forms. The
current distributed MiniMax-M3 Q4 runtime cannot safely reproduce that by
installing a Python package: it would require MiniMax-specific preference data,
a training-capable full-precision checkpoint, a validated adapter merge, and a
new quantization/quality qualification cycle. The upstream project does not
provide an MLX inference plugin.

## What ThunderMLX does at runtime

ThunderMLX checks the rolling decode tail every 12 generated tokens (rank 0
only, after a 48-token warm-up, on the batch-cancel generation path). It only
matches byte-identical, contiguous spans whose repeating unit contains a
letter (any script, so CJK loops are visible) and at least two distinct
characters:

- tiny spans (3-8 characters) require at least 16 consecutive copies, so
  legitimate short-markup walls (e.g. `<br>` runs) are not truncated;
- short spans (9-120 characters) require at least 10 consecutive copies;
- long spans (121-1,200 characters) require at least 5 consecutive copies
  (the scan window is 6,000 characters on both request paths, so the full
  band is reachable while streaming);
- punctuation-only rulers, digit-only fillers ("0, 0, 0, "), zero-run
  hashes, and single-character runs (base64 padding) are ignored — they are
  data, not loops;
- ordinary repetition penalties remain disabled for the published MiniMax-M3
  thinking profile, so valid reasoning and code are not globally distorted.

On a confirmed loop the guard arms the validated synchronized-EOS helper, and
both ranks record the identical full sequence afterwards, so the next request
reuses the shared prefix without a cross-rank divergence rebuild.

On a confirmed loop, rank 0 injects EOS through the existing sampled-token
synchronization. Both pipeline ranks consume the same token and leave decode
in lockstep. This releases the request without a rank-local break, orphaned
Metal memory, or a poisoned next turn.

Configure the scan cadence with:

```bash
MLX_M3_DECODE_REPETITION_GUARD_TOKENS=12
```

Set it to `0` to disable the runtime guard. Larger values scan less often. The
full settings editor at `http://<primary-mac>:8090/legacy` exposes this setting
under Advanced > Generation Defaults.

## Why the production model is not modified

The current Q4 checkpoint already passes long thinking, native tools, images,
200K-context decode, and cache-restart gates at the established speed baseline.
Changing its weights for a speculative FTPO adapter would make those results
non-transferable and could degrade coding or long-context quality. A future
AntiDoom-trained checkpoint should therefore be treated as a separate model
variant and promoted only after the full ThunderMLX qualification battery.

Upstream AntiDoom is Apache-2.0 licensed. ThunderMLX's detector is an
independent runtime implementation and does not copy its training code.
