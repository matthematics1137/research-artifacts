# Replication source

`replication/` contains, byte for byte, the two frozen source bundles that
govern P2R1. Their repository-relative layout is preserved.

- **Measurement bundle (22 files).** The files whose SHA-256 values are
  embedded in every P2R1 question and run contract: the protocol and rerun
  plan, the launcher, the question generator, the grid runner, the identity,
  campaign, and runtime-provenance guards, the condition builders, the
  self-tests, the tabbyAPI config template, and the lab's research-loop
  methodology note. Any change to any of these files ends an attempt; that is
  how the harness enforces a frozen protocol.
- **Analysis bundle (8 files).** The post-run promoter and validator, the
  claims builder and its tests, the figure script, the release-state gate, the
  public-allowlist builder, the paper Makefile, and the shared statistics
  module it imports (`paper4/scripts/verify_claims.py`).

`data/EXPORT_MANIFEST.json` lists every file with its bundle, byte count, and
SHA-256. The exporter copied a file only when its bytes matched the hash
recorded in the campaign toolchain receipt (measurement) or in the promotion
record (analysis).

To verify the shipped sources:

```bash
python3 - <<'EOF'
import hashlib, json
m = json.load(open("data/EXPORT_MANIFEST.json"))
for path, r in m["replication_sources"].items():
    assert hashlib.sha256(open("replication/" + path, "rb").read()).hexdigest() == r["sha256"], path
print("all", len(m["replication_sources"]), "replication sources match")
EOF
```

## Notes

- **Operator tools are not shipped.** The operator's pause and auto-resume
  tools are not part of either bundle and name this host's paths.
- **Private receipt references.** The protocol and rerun plan cite private
  incident receipts by repository-relative path and SHA-256, under
  `.publication/private/p2r1/`. Those receipts are not shipped, but their
  hashes allow them to be checked if they are ever disclosed.
- **Scanner exceptions.** Two frozen analysis sources spell a literal string
  that the release scanner rejects in ordinary files. The release gate
  contains the private-home-directory pattern it scans for, and the claims
  tests contain a synthetic private path used to prove that public claims
  reject such paths. Each exception is limited to an exact path, SHA-256, and
  literal in the release manifest.
- **Rerun requirements.** Running the harness requires the pinned engines,
  artifacts, and private input layout described in `FULL_RERUN.md`. The
  shipped sources document the procedure; they are not a turnkey installer.
