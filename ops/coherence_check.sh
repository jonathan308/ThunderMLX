#!/bin/zsh
# Guard rule: coherence-check every boot & every build.
# Exit 0 iff the model answers 17*23 with real prose containing 391.
resp=$(curl -s -m 90 http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Minimax-M3-No-Think","messages":[{"role":"user","content":"What is 17*23? Answer in one full sentence."}],"max_tokens":64,"temperature":0.2}')
content=$(echo "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
echo "coherence: ${content:-<no response>}"
[[ "$content" == *391* ]] && exit 0
exit 1
