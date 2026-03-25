from __future__ import annotations

import json
import logging
import os
import pickle
from uuid import uuid4
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

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
    INGEST_UNSYNCED_PATHS_JSON,
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


def _items_from_ingest_json(raw: Any) -> List[Any]:
    """顶层为 list（训练列表 JSON）。

    或为 ingest_unsynced_paths.json 形态：
    - legacy：`{\"paths\": [...]}`，或直接 list
    - 新：`{\"pending_by_run\": {\"run_id\": [...]}}`
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        paths = raw.get("paths")
        if isinstance(paths, list):
            return paths
        pending_by_run = raw.get("pending_by_run")
        if isinstance(pending_by_run, dict):
            out: List[Any] = []
            for v in pending_by_run.values():
                if isinstance(v, list):
                    out.extend(v)
            return out
    raise ValueError(
        'ingest JSON 须为 list，或包含 "paths"/"pending_by_run" 的对象（如 ingest_unsynced_paths.json）'
    )

def _read_unsynced_state(lmdb_folder: str) -> Dict[str, Any]:
    path = os.path.join(lmdb_folder, INGEST_UNSYNCED_PATHS_JSON)
    if not os.path.isfile(path):
        return {"pending_by_run": {}}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and isinstance(raw.get("pending_by_run"), dict):
        return raw
    # legacy format: {"paths":[...]}
    if isinstance(raw, dict) and isinstance(raw.get("paths"), list):
        return {"pending_by_run": {"__legacy__": raw["paths"]}}
    if isinstance(raw, list):
        return {"pending_by_run": {"__legacy__": raw}}
    # unknown schema -> reset (best-effort)
    return {"pending_by_run": {}}


def _update_unsynced_paths_json(lmdb_folder: str, run_id: str, paths: List[str]) -> None:
    path = os.path.join(lmdb_folder, INGEST_UNSYNCED_PATHS_JSON)
    tmp = path + ".tmp"
    state = _read_unsynced_state(lmdb_folder)
    pending_by_run = state.setdefault("pending_by_run", {})
    if paths:
        pending_by_run[run_id] = paths
    else:
        pending_by_run.pop(run_id, None)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _write_paths_json(path: str, paths: List[str], *, meta: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    payload: Dict[str, Any] = {"paths": paths}
    if meta:
        payload["meta"] = meta
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


class _BatchSyncWriter:
    """复用各 shard 的 LMDB Environment。

    每 sync_every 次成功写入后 sync + save_manifest，并只清空当前 run_id 的未 sync 记录。
    """

    def __init__(
        self,
        lmdb_folder: str,
        manifest: Dict[str, Any],
        *,
        run_id: str,
        map_size: int,
        max_shard_size_bytes: int,
        sync_every: int,
    ) -> None:
        self.lmdb_folder = lmdb_folder
        self.manifest = manifest
        self.run_id = run_id
        self.map_size = map_size
        self.max_shard_size_bytes = max_shard_size_bytes
        self.sync_every = max(1, int(sync_every))
        self._envs: Dict[str, lmdb.Environment] = {}
        self._unsynced_paths: List[str] = []
        self._writes_since_sync = 0

    def _get_env(self, shard_name: str) -> lmdb.Environment:
        if shard_name not in self._envs:
            shard_dir = os.path.join(self.lmdb_folder, shard_name)
            self._envs[shard_name] = _open_env(shard_dir, map_size=self.map_size, readonly=False)
        return self._envs[shard_name]

    def commit_payload(
        self,
        base: str,
        p: str,
        sid: str,
        iso_b: bytes,
        nat_b: bytes,
        meta_b: bytes,
    ) -> None:
        add = len(iso_b) + len(nat_b) + len(meta_b)
        shard_name = _select_writable_shard(self.manifest, self.max_shard_size_bytes, add)
        env = self._get_env(shard_name)
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
        set_basename_entry(self.manifest, base, shard=shard_name, sid=sid, source_path=p)
        bump_shard_stats(self.manifest, shard_name, add)
        self._unsynced_paths.append(p)
        _update_unsynced_paths_json(self.lmdb_folder, self.run_id, self._unsynced_paths)
        self._writes_since_sync += 1
        if self._writes_since_sync >= self.sync_every:
            self._flush_sync()

    def _flush_sync(self) -> None:
        for env in self._envs.values():
            env.sync()
        save_manifest(self.lmdb_folder, self.manifest)
        self._unsynced_paths.clear()
        _update_unsynced_paths_json(self.lmdb_folder, self.run_id, [])
        self._writes_since_sync = 0

    def finalize(self) -> None:
        for env in self._envs.values():
            env.sync()
        save_manifest(self.lmdb_folder, self.manifest)
        self._unsynced_paths.clear()
        _update_unsynced_paths_json(self.lmdb_folder, self.run_id, [])
        self._writes_since_sync = 0
        for env in self._envs.values():
            env.close()
        self._envs.clear()


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
    sync_every: int = 64,
    max_ingest: Optional[int] = None,
    max_error_logs: int = 20,
) -> Dict[str, Any]:
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    lmdb_folder = os.path.abspath(os.path.expanduser(lmdb_folder))
    manifest = load_manifest(lmdb_folder)

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = _items_from_ingest_json(raw)
    print(f"Number of list items: {len(items)}")

    stats = IngestStats(total_inputs=len(items))
    log_every = max(1, int(log_every))
    cap = max_ingest if max_ingest is not None and int(max_ingest) > 0 else None

    seen_local: set[str] = set()
    candidates: List[Tuple[str, str, str]] = []  # (basename, path, sid)
    dup_in_list = 0

    for item in items:
        p = resolve_nifti_path(item, root_dir)
        p = os.path.abspath(p)
        base = os.path.basename(p)
        if base in seen_local:
            dup_in_list += 1
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
        if cap is not None and len(candidates) >= cap:
            break

    LOG.info(
        "LMDB ingest precheck: list_items=%d unique_basenames_seen=%d dup_in_list=%d already_present(skip)=%d failed(skip/missing)=%d n_candidates=%d%s",
        len(items),
        stats.unique_basenames_seen,
        dup_in_list,
        stats.already_present,
        stats.failed,
        len(candidates),
        f" max_ingest={cap}" if cap is not None else "",
    )

    if dry_run:
        out = stats.as_dict()
        out.update(
            {
                "n_candidates": len(candidates),
                "n_list_items": len(items),
                "dry_run": True,
                "dup_in_list": dup_in_list,
            }
        )
        if cap is not None:
            out["max_ingest"] = cap
        return out

    def _log(done: int, total: int) -> None:
        if done % log_every != 0 and done != total:
            return
        remaining = max(0, total - done)
        pct = 100.0 * done / max(1, total)
        LOG.info(
            "LMDB ingest progress: processed=%d/%d (%.2f%%) newly_ingested=%d failed=%d already_present(skip)=%d remaining=%d",
            done,
            total,
            pct,
            stats.newly_ingested,
            stats.failed,
            stats.already_present,
            remaining,
        )

    done = 0
    total = len(candidates)
    error_logged = 0
    success_paths: List[str] = []

    run_id = str(uuid4())
    writer = _BatchSyncWriter(
        lmdb_folder,
        manifest,
        run_id=run_id,
        map_size=map_size,
        max_shard_size_bytes=max_shard_size_bytes,
        sync_every=sync_every,
    )
    try:
        if workers <= 1:
            for base, p, sid in candidates:
                try:
                    iso_b, nat_b, meta_b = process_nifti_to_payloads(p)
                except Exception as e:
                    stats.failed += 1
                    if error_logged < max_error_logs:
                        LOG.warning("LMDB ingest failed(process): base=%s path=%s err=%r", base, p, e)
                        error_logged += 1
                    done += 1
                    _log(done, total)
                    continue
                try:
                    writer.commit_payload(base, p, sid, iso_b, nat_b, meta_b)
                    stats.newly_ingested += 1
                    success_paths.append(p)
                except Exception as e:
                    stats.failed += 1
                    if error_logged < max_error_logs:
                        LOG.warning("LMDB ingest failed(commit): base=%s path=%s err=%r", base, p, e)
                        error_logged += 1
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
                        if error_logged < max_error_logs:
                            LOG.warning(
                                "LMDB ingest failed(process_worker): base=%s path=%s err=%r",
                                base,
                                p,
                                err,
                            )
                            error_logged += 1
                        done += 1
                        _log(done, total)
                        continue
                    iso_b, nat_b, meta_b = payload
                    try:
                        writer.commit_payload(base, p, sid, iso_b, nat_b, meta_b)
                        stats.newly_ingested += 1
                        success_paths.append(p)
                    except Exception as e:
                        stats.failed += 1
                        if error_logged < max_error_logs:
                            LOG.warning("LMDB ingest failed(commit): base=%s path=%s err=%r", base, p, e)
                            error_logged += 1
                    done += 1
                    _log(done, total)
    finally:
        writer.finalize()

    if cap is not None:
        success_json_path = os.path.join(lmdb_folder, f"ingest_success_paths_{run_id}.json")
        _write_paths_json(
            success_json_path,
            success_paths,
            meta={"run_id": run_id, "max_ingest": cap, "n_success": len(success_paths)},
        )
        LOG.info("LMDB ingest success paths written: %s (n=%d)", success_json_path, len(success_paths))
    out = stats.as_dict()
    out["lmdb_folder"] = lmdb_folder
    out["n_shards"] = len(manifest.get("shards", {}))
    out["n_list_items"] = len(items)
    if cap is not None:
        out["max_ingest"] = cap
        out["n_success_paths"] = len(success_paths)
        out["success_json"] = os.path.join(lmdb_folder, f"ingest_success_paths_{run_id}.json")
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
