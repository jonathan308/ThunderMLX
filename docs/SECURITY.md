# Security And Publishing Notes

This project is meant to be published without private machine coordinates.

Never commit:

- `.env.local`, `.env`, or `m3_cluster.env`
- SSH private keys or private key filenames
- passwords
- private IPs, VPN IPs, or personal hostnames
- local user home paths
- logs containing client addresses
- generated hostfiles

The dashboard can start, stop, and sync cluster processes. Keep it bound to `127.0.0.1` unless you have put it behind a trusted network boundary.

Before publishing:

```bash
rg -n "password|passwd|secret|token|id_ed25519|/Users/|100\\.|10\\.0\\.0\\." --glob '!.env.local' --glob '!*.log' --glob '!__pycache__/**'
git status --ignored --short
```

