from __future__ import annotations

import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lmdb
import numpy as np
import torch

from .preprocess import resolve_nifti_path, process_nifti_to_payloads, subject_id_from_path
from .index import (
    basename_entry,
    bump_shard_stats,
    get_shard_stats,
    load_manifest,
    save_manifest,
    set_basename_entry,
)
from .types import (
    DEFAULT_MAP_SIZE,
    DEFAULT_MAX_SHARD_SIZE_BYTES,
    KEY_KEYS,
    SHARD_PREFIX,
    SHARD_SUFFIX,
    SUFFIX_ISO,
    SUFFIX_META,
    SUFFIX_NATIVE,
    IngestStats,
)

LOG = logging.getLogger(__name__)


def _subject_keys(sid: str) -> Tuple[bytes, bytes, bytes]:
    b = sid.encode("ascii")
    return (b + SUFFIX_ISO.encode("ascii"), b + SUFFIX_NATIVE.encode("ascii"), b + SUFFIX_META.encode("ascii"))


def _list_shard_names(manifest: Dict[str, Any]) -> List[str]:
    names = sorted(manifest.get("shards", {}).keys())
    if names:
        return names
    return [f"{SHARD_PREFIX}000{SHARD_SUFFIX}"]


def _next_shard_name(last: str) -> str:
    stem = last[: -len(SHARD_SUFFIX)] if last.endswith(SHARD_SUFFIX) else last
    idx = int(stem.split("_")[-1])
    return f"{SHARD_PREFIX}{idx + 1:03d}{SHARD_SUFFIX}"


def _select_writable_shard(manifest: Dict[str, Any], max_shard_size_bytes: int, estimated_new_bytes: int) -> str:
    names = _list_shard_names(manifest)
    cur = names[-1]
    s = get_shard_stats(manifest, cur)
    used = int(s.get("estimated_bytes", 0))
    if used + estimated_new_bytes <= max_shard_size_bytes:
        return cur
    nxt = _next_shard_name(cur)
    get_shard_stats(manifest, nxt)
    return nxt


def _open_env(path: str, map_size: int, readonly: bool = False) -> lmdb.Environment:
    os.makedirs(path, exist_ok=True)
    if readonly:
        return lmdb.open(path, readonly=True, lock=False, readahead=False, subdir=True)
    kw = dict(map_size=map_size, subdir=True, lock=True, readahead=False, meminit=False, map_async=True)
    try:
        return lmdb.open(path, writemap=True, **kw)
    except Exception:
        return lmdb.open(path, writemap=False, **kw)


def _process_worker(nifti_abs: str) -> Tuple[str, Optional[Tuple[bytes, bytes, bytes]], Optional[str]]:
    try:
        return nifti_abs, process_nifti_to_payloads(nifti_abs), None
    except Exception as e:
        return nifti_abs, None, str(e)


def ingest_json_incremental(
    json_path: str,
    root_dir: str,
    lmdb_folder: str,
    *,
    max_shard_size_bytes: int = DEFAULT_MAX_SHARD_SIZE_BYTES,
    map_size: int = DEFAULT_MAP_SIZE,
    workers: int = 0,
    log_every: int = 1000,
    dry_run: bool = False,
) -> Dict[str, Any]:
    import json

    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    lmdb_folder = os.path.abspath(os.path.expanduser(lmdb_folder))
    manifest = load_manifest(lmdb_folder)

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    stats = IngestStats(total_inputs=len(raw))
    log_every = max(1, int(log_every))

    seen_local: set[str] = set()
    candidates: List[Tuple[str, str, str]] = []  # (basename, path, sid)

    for item in raw:
        p = resolve_nifti_path(item, root_dir)
        p = os.path.abspath(p)
        base = os.path.basename(p)
        if base in seen_local:
            continue
        seen_local.add(base)
        stats.unique_basenames_seen += 1
        if basename_entry(manifest, base) is not None:
            stats.already_present += 1
            continue
        if not os.path.isfile(p):
            stats.failed += 1
            continue
        sid = subject_id_from_path(p, root_dir)
        candidates.append((base, p, sid))

    if dry_run:
        out = stats.as_dict()
        out.update({"n_candidates": len(candidates), "dry_run": True})
        return out

    def _log(done: int, total: int) -> None:
        if done % log_every != 0 and done != total:
            return
        LOG.info("LMDB shard ingest: %d/%d (%.2f%%)", done, total, 100.0 * done / max(1, total))

    done = 0
    total = len(candidates)

    if workers <= 1:
        for base, p, sid in candidates:
            try:
                iso_b, nat_b, meta_b = process_nifti_to_payloads(p)
            except Exception:
                stats.failed += 1
                done += 1
                _log(done, total)
                continue
            add = len(iso_b) + len(nat_b) + len(meta_b)
            shard_name = _select_writable_shard(manifest, max_shard_size_bytes, add)
            shard_dir = os.path.join(lmdb_folder, shard_name)
            env = _open_env(shard_dir, map_size=map_size, readonly=False)
            try:
                k_iso, k_nat, k_meta = _subject_keys(sid)
                with env.begin(write=True) as txn:
                    txn.put(k_iso, iso_b)
                    txn.put(k_nat, nat_b)
                    txn.put(k_meta, meta_b)
                    # append sid in per-shard __keys__
                    current = txn.get(KEY_KEYS)
                    ids = [] if current is None else pickle.loads(current)
                    if sid not in ids:
                        ids.append(sid)
                    txn.put(KEY_KEYS, pickle.dumps(ids, protocol=pickle.HIGHEST_PROTOCOL))
                set_basename_entry(manifest, base, shard=shard_name, sid=sid, source_path=p)
                bump_shard_stats(manifest, shard_name, add)
                stats.newly_ingested += 1
            except Exception:
                stats.failed += 1
            finally:
                env.sync()
                env.close()
            done += 1
            _log(done, total)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_process_worker, p): (base, p, sid) for base, p, sid in candidates}
            for fut in as_completed(futs):
                base, p, sid = futs[fut]
                try:
                    _, payload, err = fut.result()
                except Exception:
                    payload, err = None, "future failed"
                if err or payload is None:
                    stats.failed += 1
                    done += 1
                    _log(done, total)
                    continue
                iso_b, nat_b, meta_b = payload
                add = len(iso_b) + len(nat_b) + len(meta_b)
                shard_name = _select_writable_shard(manifest, max_shard_size_bytes, add)
                shard_dir = os.path.join(lmdb_folder, shard_name)
                env = _open_env(shard_dir, map_size=map_size, readonly=False)
                try:
                    k_iso, k_nat, k_meta = _subject_keys(sid)
                    with env.begin(write=True) as txn:
                        txn.put(k_iso, iso_b)
                        txn.put(k_nat, nat_b)
                        txn.put(k_meta, meta_b)
                        current = txn.get(KEY_KEYS)
                        ids = [] if current is None else pickle.loads(current)
                        if sid not in ids:
                            ids.append(sid)
                        txn.put(KEY_KEYS, pickle.dumps(ids, protocol=pickle.HIGHEST_PROTOCOL))
                    set_basename_entry(manifest, base, shard=shard_name, sid=sid, source_path=p)
                    bump_shard_stats(manifest, shard_name, add)
                    stats.newly_ingested += 1
                except Exception:
                    stats.failed += 1
                finally:
                    env.sync()
                    env.close()
                done += 1
                _log(done, total)

    save_manifest(lmdb_folder, manifest)
    out = stats.as_dict()
    out["lmdb_folder"] = lmdb_folder
    out["n_shards"] = len(manifest.get("shards", {}))
    return out


def load_volume_from_shards(
    lmdb_folder: str,
    *,
    basename: str,
    mode: str = "iso1mm",
) -> torch.Tensor:
    """按 basename 从 manifest 路由到分片并读取体数据。"""
    manifest = load_manifest(lmdb_folder)
    e = basename_entry(manifest, basename)
    if e is None:
        raise KeyError(f"basename not found: {basename}")
    shard_name = e["shard"]
    sid = e["sid"]
    key = sid.encode("ascii") + (SUFFIX_NATIVE if mode == "native" else SUFFIX_ISO).encode("ascii")
    meta_key = sid.encode("ascii") + SUFFIX_META.encode("ascii")
    env = _open_env(os.path.join(lmdb_folder, shard_name), map_size=1 << 30, readonly=True)
    try:
        with env.begin(write=False) as txn:
            raw = txn.get(key)
            meta_raw = txn.get(meta_key)
            if raw is None or meta_raw is None:
                raise KeyError(f"missing key for basename={basename}")
            meta = pickle.loads(meta_raw)
            shape = meta["shape_native" if mode == "native" else "shape_iso"]
            arr = np.frombuffer(raw, dtype=np.float16).copy().reshape(shape)
            return torch.from_numpy(arr).float()
    finally:
        env.close()
