"""Merge per-source ``_seeds_<SRC>.json`` checkpoints into a single ``seeds.json``.

Why this exists
---------------
``scripts/run_array.sbatch`` parallelises Stage B.1 across one SLURM array
task per source dataset.  Each task invokes ``run_generate_seeds.py
--datasets <ONE_SRC>`` which, deep inside :func:`search.seed_generator
.generate_seeds`, writes both:

  - ``_seeds_<SRC>.json``  — per-source checkpoint (resume-safe)
  - ``seeds.json``         — final merged file (but only over what *this*
                              task processed: a single source)

The ``seeds.json`` write is therefore racy under array fan-out: every task
overwrites it with its own one-source slice, and the last task to finish
wins.  Result: ``seeds.json`` ends up containing exactly **one** source's
records, even though all 6 ``_seeds_<SRC>.json`` checkpoints exist on disk.

This standalone merger reads every ``_seeds_*.json`` it finds and writes a
correctly-merged ``seeds.json``.  Idempotent — safe to re-run.

Usage
-----
::

    python scripts/merge_seed_checkpoints.py \\
        --ckpt_dir outputs/full_etth1_autocts/seeds \\
        --out      outputs/full_etth1_autocts/seeds/seeds.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge per-source _seeds_<SRC>.json into a single seeds.json.",
    )
    p.add_argument(
        "--ckpt_dir", required=True,
        help="Directory containing _seeds_*.json checkpoints "
             "(e.g. outputs/full_etth1_autocts/seeds).",
    )
    p.add_argument(
        "--out", default=None,
        help="Output path for the merged seeds.json. "
             "Defaults to {ckpt_dir}/seeds.json.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print per-checkpoint counts, do not write the merged file.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ckpts: List[str] = sorted(glob.glob(os.path.join(args.ckpt_dir, "_seeds_*.json")))
    if not ckpts:
        print(f"[merge_seeds] no _seeds_*.json found in {args.ckpt_dir}", file=sys.stderr)
        return 1

    print(f"[merge_seeds] found {len(ckpts)} checkpoint(s):")
    all_records: list = []
    for p in ckpts:
        try:
            with open(p, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"  {os.path.basename(p):30s}  ERROR ({exc}) — skipping",
                  file=sys.stderr)
            continue
        recs = payload.get("records") or []
        ds = payload.get("ds_name", "?")
        print(f"  {os.path.basename(p):30s}  {len(recs):5d} records  (ds_name={ds})")
        all_records.extend(recs)

    print(f"[merge_seeds] total: {len(all_records)} records")

    # Sanity: report per-source counts in the merged set so the user can
    # eyeball whether anything is missing.
    by_base = Counter()
    for r in all_records:
        tid = r.get("task_id", "")
        base = tid.split(":", 1)[0] if ":" in tid else tid
        by_base[base] += 1
    print("[merge_seeds] per-source breakdown:")
    for base, n in sorted(by_base.items()):
        print(f"  {base:20s}  {n:5d} records")

    if args.dry_run:
        print("[merge_seeds] --dry_run set — no file written")
        return 0

    out = args.out or os.path.join(args.ckpt_dir, "seeds.json")

    # PID-suffixed tmp path so 6 array tasks running merger in parallel each
    # write to their own tmp file.  Without the suffix, two concurrent
    # writers collide:
    #
    #   1. Process A opens "seeds.json.tmp" (write-mode), writes, closes.
    #   2. Process B opens the same path, truncates A's content, writes.
    #   3. A calls ``os.replace(tmp, out)`` — succeeds, tmp now gone.
    #   4. B calls ``os.replace(tmp, out)`` — FileNotFoundError (tmp was
    #      moved out from under it by A).
    #
    # ``os.replace`` is a POSIX ``rename(2)`` which is atomic at the
    # destination, so per-process tmp files are sufficient: whichever
    # writer's rename runs last is the visible content of seeds.json, and
    # since every concurrent writer reads the same set of checkpoints at
    # the same point in time, all writers produce identical content (modulo
    # writes interleaving with later checkpoints — handled by the LAST
    # array task's merger, which always sees the full checkpoint set).
    tmp = f"{out}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    os.replace(tmp, out)
    print(f"[merge_seeds] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
