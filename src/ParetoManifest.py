"""
src/ParetoManifest.py
=====================

Run-level metadata: what was configured, what completed, what schema.

The manifest is the single source of truth for a pareto run. It records
the run's parameters, the schema version of its outputs, when it ran,
and which slices completed. It lives at the root of the run folder
and is updated incrementally as slices finish.

Why bother:
  - Lets you tell two runs apart without parsing folder names.
  - Lets downstream readers validate they're consuming the schema
    they expect, not silently breaking on layout changes.
  - Lets a re-run resume cleanly: skip slices the manifest says are
    done, redo the rest. Safer than file-existence checks alone.
  - Survives partial deletions: if someone wipes one slice's data
    folder, the manifest still records that the run was attempted.

Concurrency note:
  Slice-completion writes use a tempfile + atomic rename so two
  processes finishing slices simultaneously won't corrupt the file.
  We do NOT support two processes editing the same field — the design
  assumes one orchestrator per run, with parallel slices appending
  their own keys.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional

from src.ParetoPaths import (
    SCHEMA_VERSION,
    manifest_path,
    slice_slug,
    SliceKey,
)


# ============================================================================
# Schema
# ============================================================================

@dataclass
class SliceRecord:
    """One slice's completion record. Written when a slice finishes."""
    bucket: str
    budget: float
    loft: float
    completed_at: str                # ISO-8601 UTC
    n_equity_floors_solved: int
    n_equity_floors_infeasible: int
    epc_mode: bool
    notes: str = ''                  # free-form, e.g. "infeasible at 75%+"


@dataclass
class RunManifest:
    """Top-level manifest. Serialised as the run's manifest.json."""
    schema_version: int
    run_id: str                      # e.g. timestamp + short hash
    created_at: str
    last_updated_at: str

    # Configuration that defines the run
    cost_scenarios: list[str]
    budgets: list[float]
    loft_probs: list[float]
    equity_floors: list[float]
    mip_gap: float
    epc_mode: bool
    test_mode: bool
    test_sample_size: Optional[int]
    test_seed: Optional[int]

    # Provenance
    git_sha: Optional[str]
    hostname: str
    input_base_dir: str

    # Per-slice completion log, keyed by slice slug
    slices: dict[str, SliceRecord] = field(default_factory=dict)


# ============================================================================
# Construction
# ============================================================================

def _git_sha() -> Optional[str]:
    """Best-effort git SHA. None if not in a repo or git unavailable."""
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        return None


def _now_iso() -> str:
    """UTC ISO-8601 timestamp with second precision."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec='seconds',
    )


def _make_run_id() -> str:
    """A run ID that's sortable and human-readable. No collision needed."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        '%Y%m%d_%H%M%S',
    )


def new_manifest(cfg) -> RunManifest:
    """
    Build a manifest from a RunConfig at the start of a run.

    cfg is the RunConfig from pareto.py — typed loosely here to avoid
    a circular import. The fields read are documented above.
    """
    now = _now_iso()
    return RunManifest(
        schema_version=SCHEMA_VERSION,
        run_id=_make_run_id(),
        created_at=now,
        last_updated_at=now,
        cost_scenarios=list(cfg.cost_scenarios),
        budgets=list(cfg.budgets),
        loft_probs=list(cfg.loft_probs),
        equity_floors=list(cfg.equity_floors),
        mip_gap=cfg.mip_gap,
        epc_mode=cfg.epc_run,
        test_mode=cfg.test_mode,
        test_sample_size=cfg.test_sample_size if cfg.test_mode else None,
        test_seed=cfg.test_seed if cfg.test_mode else None,
        git_sha=_git_sha(),
        hostname=socket.gethostname(),
        input_base_dir=cfg.input_base_dir,
    )


# ============================================================================
# I/O
# ============================================================================

def write_manifest(root: str, manifest: RunManifest) -> None:
    """Atomic write to <root>/manifest.json."""
    os.makedirs(root, exist_ok=True)
    target = manifest_path(root)
    manifest.last_updated_at = _now_iso()
    payload = _to_json_dict(manifest)

    # Write to a tempfile in the same directory, then rename. Rename is
    # atomic on POSIX, which means a reader can never see a half-written
    # file even if two writes race.
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=root, prefix='.manifest_', suffix='.tmp',
    )
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_path, target)
    except Exception:
        # Don't leave the tempfile if the write failed.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def read_manifest(root: str) -> Optional[RunManifest]:
    """
    Load <root>/manifest.json. Returns None if missing.

    Raises ManifestSchemaError if the file exists but its schema_version
    doesn't match what this code expects — silently consuming a stale
    manifest is worse than crashing.
    """
    path = manifest_path(root)
    if not os.path.exists(path):
        return None

    with open(path) as f:
        payload = json.load(f)

    found_version = payload.get('schema_version')
    if found_version != SCHEMA_VERSION:
        raise ManifestSchemaError(
            f"Manifest at {path} is schema v{found_version}; "
            f"this code expects v{SCHEMA_VERSION}. "
            f"Re-run from scratch or migrate the manifest."
        )

    return _from_json_dict(payload)


class ManifestSchemaError(RuntimeError):
    """Raised when an on-disk manifest's schema doesn't match this code."""


# ============================================================================
# Slice updates
# ============================================================================

def record_slice(
    root: str,
    key: SliceKey,
    *,
    n_equity_floors_solved: int,
    n_equity_floors_infeasible: int,
    epc_mode: bool,
    notes: str = '',
) -> None:
    """
    Mark a slice as complete in the manifest. Atomic read-modify-write.

    Safe to call from a worker that just finished a slice. If two workers
    record different slices simultaneously, the last writer wins on the
    file but no slice record is lost (each writes its own key).
    """
    manifest = read_manifest(root)
    if manifest is None:
        raise RuntimeError(
            f"Cannot record slice: no manifest at {manifest_path(root)}. "
            f"Call new_manifest() and write_manifest() at run start."
        )

    record = SliceRecord(
        bucket=key.bucket,
        budget=key.budget,
        loft=key.loft,
        completed_at=_now_iso(),
        n_equity_floors_solved=n_equity_floors_solved,
        n_equity_floors_infeasible=n_equity_floors_infeasible,
        epc_mode=epc_mode,
        notes=notes,
    )
    manifest.slices[key.slug] = record
    write_manifest(root, manifest)


def is_slice_recorded(root: str, key: SliceKey) -> bool:
    """True if the manifest says this slice has been completed."""
    manifest = read_manifest(root)
    if manifest is None:
        return False
    return key.slug in manifest.slices


# ============================================================================
# Compatibility checks
# ============================================================================

def assert_compatible_with_cfg(manifest: RunManifest, cfg) -> None:
    """
    Verify a re-run's config matches the existing manifest. If a user
    re-runs with different budgets, equity floors, or buckets, we want
    to fail loudly rather than silently produce a tree mixing two runs.

    Mip gap and verbosity changes are tolerated — they don't change
    what's on disk.
    """
    mismatches = []
    for field_name in ('cost_scenarios', 'budgets', 'loft_probs',
                       'equity_floors', 'epc_mode', 'test_mode'):
        manifest_val = getattr(manifest, field_name)
        cfg_attr = {
            'cost_scenarios': 'cost_scenarios',
            'budgets': 'budgets',
            'loft_probs': 'loft_probs',
            'equity_floors': 'equity_floors',
            'epc_mode': 'epc_run',
            'test_mode': 'test_mode',
        }[field_name]
        cfg_val = getattr(cfg, cfg_attr)
        if list(manifest_val) != list(cfg_val):
            mismatches.append(
                f"  {field_name}: manifest={manifest_val} cfg={cfg_val}"
            )

    if mismatches:
        raise ManifestConfigMismatch(
            "Re-run config doesn't match existing manifest:\n"
            + "\n".join(mismatches)
            + "\n\nEither delete the run folder and start fresh, or "
              "re-run with the original config."
        )


class ManifestConfigMismatch(RuntimeError):
    """Raised when a re-run's config conflicts with an existing manifest."""


# ============================================================================
# Serialisation helpers — keep dataclass <-> dict at the edges
# ============================================================================

def _to_json_dict(manifest: RunManifest) -> dict:
    """Dataclass to JSON-safe dict. Slices flatten to their dict form."""
    payload = asdict(manifest)
    # asdict() already recurses into nested dataclasses; nothing extra
    # needed unless we add non-trivial types (datetime, np.float, etc.)
    return payload


def _from_json_dict(payload: dict) -> RunManifest:
    """JSON dict back to dataclass. Reconstructs SliceRecord values."""
    slices_raw = payload.pop('slices', {}) or {}
    slices = {
        slug: SliceRecord(**rec) for slug, rec in slices_raw.items()
    }
    return RunManifest(slices=slices, **payload)