"""Merge per-source ``_seeds_<SRC>.json`` checkpoints into a single ``seeds.json``
and aggregate per-sub-task noisy↔clean ρ values into ``seeds_meta.json``.

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
records, even though all per-source ``_seeds_<SRC>.json`` checkpoints exist
on disk.

This standalone merger reads every ``_seeds_*.json`` it finds and writes a
correctly-merged ``seeds.json``.  Idempotent — safe to re-run.

Sub-task ρ aggregation (rev 2026-05-18)
----------------------------------------
Each ``_seeds_<SRC>.json`` may carry a ``sub_task_rhos`` map (added by
:func:`search.seed_generator.generate_seeds` at the end of each noisy stage):
``{window_id: spearman_ρ}``.  This merger consolidates them into one file:

  - ``seeds_meta.json`` — sibling of ``seeds.json``.  Contains
    ``{"noisy_sub_task_rhos": {window_id: ρ}, "n_sources": N, ...}``.

The comparator trainer (``run_pretrain_comparator.py``) loads
``seeds_meta.json`` when present and applies the
``comparator.noisy_rho_threshold`` filter to drop low-quality noisy buckets
(see ``Debug/noisy_rho_audit_2026_05_16.md §5 #1``).

Back-fill from logs (rev 2026-05-18)
-------------------------------------
For runs whose seed checkpoints predate the ρ-persistence change (e.g.
``outputs/zac_2026_05_12/seeds`` generated on 2026-05-12), the ρ values
exist only in the per-source ``run_array_*_<SRC>.log`` files.  Passing
``--log_dir`` enables a one-shot back-fill: missing ρ values are extracted
from the matching log lines and injected into the checkpoints in-place
(atomic rewrite), so the next merge picks them up like any other run.

Usage
-----
::

    # Vanilla merge (no ρ back-fill).
    python scripts/merge_seed_checkpoints.py \\
        --ckpt_dir outputs/zac_2026_05_12/seeds \\
        --out      outputs/zac_2026_05_12/seeds/seeds.json

    # With back-fill from existing logs (one-shot for pre-rho runs).
    python scripts/merge_seed_checkpoints.py \\
        --ckpt_dir outputs/zac_2026_05_12/seeds \\
        --out      outputs/zac_2026_05_12/seeds/seeds.json \\
        --log_dir  outputs/zac_2026_05_12
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple


# Log-line regexes for ρ back-fill from ``run_array_*_<SRC>.log``.  Both
# regexes deliberately anchor on the ``search.seed_generator`` log prefix so
# we never pick up unrelated lines that happen to contain similar tokens.
_RE_WINDOW = re.compile(
    r"\[(?P<ds>[A-Za-z0-9_]+)\] sub-task (?P<sub>\d+)/\d+\s+"
    r"window_id=(?P<window>\S+)"
)
_RE_RHO = re.compile(
    r"\[(?P<ds>[A-Za-z0-9_]+)\] sub-task (?P<sub>\d+)/\d+\s+"
    r"noisy-vs-clean Spearman rho on \d+ shared candidates "
    r"= (?P<rho>[+-]?\d+\.\d+)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge per-source _seeds_<SRC>.json into a single seeds.json.",
    )
    p.add_argument(
        "--ckpt_dir", required=True,
        help="Directory containing _seeds_*.json checkpoints "
             "(e.g. outputs/zac_2026_05_12/seeds).",
    )
    p.add_argument(
        "--out", default=None,
        help="Output path for the merged seeds.json. "
             "Defaults to {ckpt_dir}/seeds.json.",
    )
    p.add_argument(
        "--meta_out", default=None,
        help="Output path for seeds_meta.json (sub-task ρ aggregate). "
             "Defaults to a sibling of --out.",
    )
    p.add_argument(
        "--log_dir", default=None,
        help="If given, scan ``run_array_*_<SRC>.log`` files in this dir for "
             "noisy↔clean Spearman ρ lines and back-fill any sub_task_rhos "
             "missing from the per-source checkpoints (in-place, atomic). "
             "One-shot; no-op once all checkpoints carry their ρ values.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print per-checkpoint counts, do not write the merged file.",
    )
    return p.parse_args()


def _extract_rhos_from_log(log_path: str) -> Dict[str, float]:
    """Parse one ``run_array_*_<SRC>.log`` and return ``{window_id: ρ}``.

    Walks the file line by line; tracks the most recent ``window_id=…``
    seen per ``(dataset, sub_idx)`` pair, then attributes each subsequent
    ``noisy-vs-clean Spearman rho ... = +X.XXX`` line to that window_id.
    Resume retries that re-print the window-id banner are handled by
    "last writer wins" — the freshest ρ for a sub-task ends up in the map.
    """
    window_map: Dict[Tuple[str, str], str] = {}
    rho_map:    Dict[str, float] = {}
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _RE_WINDOW.search(line)
                if m:
                    window_map[(m["ds"], m["sub"])] = m["window"]
                    continue
                m = _RE_RHO.search(line)
                if m:
                    key = (m["ds"], m["sub"])
                    wid = window_map.get(key)
                    if wid is not None:
                        rho_map[wid] = float(m["rho"])
    except OSError as exc:
        print(f"[merge_seeds]   WARNING: cannot read {log_path}: {exc}",
              file=sys.stderr)
    return rho_map


def _backfill_rhos(ckpts: List[str], log_dir: str) -> int:
    """Inject any missing ρ values from ``log_dir`` into each checkpoint.

    Returns the number of (checkpoint, window_id) ρ values newly written.
    Idempotent: checkpoints that already carry every ρ their logs mention
    are left untouched.
    """
    if not os.path.isdir(log_dir):
        print(f"[merge_seeds] --log_dir {log_dir} does not exist — skipping "
              f"back-fill", file=sys.stderr)
        return 0

    # Find every ``run_array_*_<SRC>.log`` and group by source.
    logs_by_src: Dict[str, List[str]] = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "run_array_*_*.log"))):
        # Filename pattern: run_array_<idx>_<SRC>.log
        stem = os.path.basename(path)[len("run_array_"):-len(".log")]
        # stem is now "<idx>_<SRC>"; SRC may contain underscores so split once.
        if "_" not in stem:
            continue
        _, src = stem.split("_", 1)
        logs_by_src.setdefault(src, []).append(path)

    if not logs_by_src:
        print(f"[merge_seeds] --log_dir {log_dir} has no run_array_*_*.log "
              f"files — back-fill is a no-op", file=sys.stderr)
        return 0

    n_added = 0
    for ckpt in ckpts:
        # Match the checkpoint to its source via the _seeds_<SRC>.json name.
        fname = os.path.basename(ckpt)
        if not (fname.startswith("_seeds_") and fname.endswith(".json")):
            continue
        src = fname[len("_seeds_"):-len(".json")]
        src_logs = logs_by_src.get(src, [])
        if not src_logs:
            continue

        # Load existing checkpoint payload.
        try:
            with open(ckpt, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"[merge_seeds]   {fname}: cannot read for back-fill "
                  f"({exc}); skipping", file=sys.stderr)
            continue
        existing = dict(payload.get("sub_task_rhos") or {})

        # Union of ρ values from every log file for this source (last writer
        # wins, mirroring resume semantics).
        scraped: Dict[str, float] = {}
        for log_path in src_logs:
            scraped.update(_extract_rhos_from_log(log_path))

        new_count = 0
        for wid, rho in scraped.items():
            if wid not in existing:
                existing[wid] = float(rho)
                new_count += 1

        if new_count > 0:
            payload["sub_task_rhos"] = existing
            tmp = f"{ckpt}.tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, ckpt)
            print(f"[merge_seeds]   {fname}: back-filled {new_count} ρ "
                  f"value(s) from {len(src_logs)} log file(s)")
            n_added += new_count
        elif scraped:
            print(f"[merge_seeds]   {fname}: all {len(scraped)} log ρ "
                  f"value(s) already present in checkpoint")

    return n_added


def main() -> int:
    args = parse_args()

    ckpts: List[str] = sorted(glob.glob(os.path.join(args.ckpt_dir, "_seeds_*.json")))
    if not ckpts:
        print(f"[merge_seeds] no _seeds_*.json found in {args.ckpt_dir}",
              file=sys.stderr)
        return 1

    # ── ρ back-fill (optional) ─────────────────────────────────────────
    if args.log_dir:
        print(f"[merge_seeds] back-fill ρ from logs in {args.log_dir}:")
        n_added = _backfill_rhos(ckpts, args.log_dir)
        print(f"[merge_seeds] back-fill complete: {n_added} new ρ value(s) "
              f"injected across {len(ckpts)} checkpoint(s)")

    # ── Merge records + aggregate ρ ────────────────────────────────────
    print(f"[merge_seeds] found {len(ckpts)} checkpoint(s):")
    all_records: list = []
    aggregated_rhos: Dict[str, float] = {}
    rho_source_counts: Counter = Counter()
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
        rhos = payload.get("sub_task_rhos") or {}
        # Newer-writer-wins is fine: per-source checkpoints don't share keys
        # (window_ids are namespaced by source) so this is effectively a
        # union without collision.
        for wid, rho in rhos.items():
            aggregated_rhos[str(wid)] = float(rho)
            rho_source_counts[ds] += 1
        print(f"  {os.path.basename(p):30s}  {len(recs):5d} records  "
              f"(ds_name={ds}, ρ entries={len(rhos)})")
        all_records.extend(recs)

    print(f"[merge_seeds] total: {len(all_records)} records, "
          f"{len(aggregated_rhos)} ρ entries across {len(rho_source_counts)} "
          f"sources")

    # Sanity: report per-source counts in the merged set so the user can
    # eyeball whether anything is missing.
    by_base: Counter = Counter()
    for r in all_records:
        tid = r.get("task_id", "")
        base = tid.split(":", 1)[0] if ":" in tid else tid
        by_base[base] += 1
    print("[merge_seeds] per-source breakdown:")
    for base, n in sorted(by_base.items()):
        n_rho = rho_source_counts.get(base, 0)
        print(f"  {base:20s}  {n:5d} records   ρ entries: {n_rho}")

    if args.dry_run:
        print("[merge_seeds] --dry_run set — no file written")
        return 0

    out = args.out or os.path.join(args.ckpt_dir, "seeds.json")
    meta_out = args.meta_out or os.path.join(
        os.path.dirname(out) or args.ckpt_dir, "seeds_meta.json",
    )

    # PID-suffixed tmp path so multiple array tasks running merger in parallel
    # each write to their own tmp file.  See block comment below for why this
    # matters under SLURM array fan-out.
    _atomic_write_json(out, all_records)
    print(f"[merge_seeds] wrote {out}")

    # Always write seeds_meta.json — even when empty — so downstream readers
    # have an unambiguous "no ρ map available" signal (empty dict) rather
    # than a missing-file ambiguity ("did the merger not run, or is the map
    # genuinely empty?").
    meta_payload = {
        "noisy_sub_task_rhos": aggregated_rhos,
        "n_sources":           len(rho_source_counts),
        "n_sub_tasks_with_rho": len(aggregated_rhos),
        "rho_by_source_count": dict(rho_source_counts),
    }
    _atomic_write_json(meta_out, meta_payload)
    print(f"[merge_seeds] wrote {meta_out}  "
          f"({len(aggregated_rhos)} ρ entries)")
    return 0


def _atomic_write_json(path: str, payload) -> None:
    """PID-suffixed tmp → ``os.replace`` for cross-process safety.

    PID-suffixed tmp path avoids the collision where two array tasks open
    the same ``${path}.tmp``, the second truncates the first's content,
    then both call ``os.replace`` — the first wins atomically, the second
    hits ``FileNotFoundError`` because the tmp is gone.  Per-process tmp
    files sidestep this entirely.

    ``os.replace`` itself is a POSIX ``rename(2)`` which is atomic at the
    destination, so whichever writer's rename runs last is what
    downstream readers see.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


if __name__ == "__main__":
    sys.exit(main())
