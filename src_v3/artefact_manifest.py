"""Write or check the SHA-256 manifest of the artefacts a served version reads (MODEL_ANALYSIS MC3, plan C1).

The manifest is tracked in git; the artefacts are not. `write` hashes the
files that match `C.SERVED_ARTEFACTS[version]`; `check` fails on a missing
file or a changed hash, so a copied bundle can be verified before Tier 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging

from src_v3 import config as C

log = logging.getLogger(__name__)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def collect(version: int) -> dict:
    files = sorted({p for pattern in C.SERVED_ARTEFACTS[version] for p in C.ROOT.glob(pattern)})
    if not files:
        raise FileNotFoundError(f"No artefact of v{version} is on disk. Build them with REPRODUCE.md Tier 2, then rerun.")
    return {p.relative_to(C.ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("write", "check"))
    ap.add_argument("--version", type=int, choices=sorted(C.SERVED_ARTEFACTS), default=max(C.SERVED_ARTEFACTS))
    args = ap.parse_args()
    path = C.ROOT / "models" / f"v{args.version}.manifest.json"
    if args.mode == "write":
        path.write_text(json.dumps(collect(args.version), indent=1))
        log.info("wrote %s", path.name)
        return
    want = json.loads(path.read_text())
    bad = [f for f, v in want.items() if not (C.ROOT / f).exists() or sha256(C.ROOT / f) != v["sha256"]]
    if bad:
        raise SystemExit(f"{len(bad)} artefacts are missing or changed, first: {bad[:5]}. "
                         "Copy the bundle of this version again, then rerun.")
    log.info("all %d artefacts of v%d match the manifest", len(want), args.version)


if __name__ == "__main__":
    main()
