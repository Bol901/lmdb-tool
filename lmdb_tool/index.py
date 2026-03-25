from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from .types import MANIFEST_NAME, MANIFEST_VERSION


def manifest_path(lmdb_folder: str) -> str:
    return os.path.join(lmdb_folder, MANIFEST_NAME)


def load_manifest(lmdb_folder: str) -> Dict[str, Any]:
    os.makedirs(lmdb_folder, exist_ok=True)
    mp = manifest_path(lmdb_folder)
    if not os.path.isfile(mp):
        return {
            "version": MANIFEST_VERSION,
            "basename_index": {},
            "shards": {},
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
        }
    with open(mp, "r", encoding="utf-8") as f:
        m = json.load(f)
    m.setdefault("version", MANIFEST_VERSION)
    m.setdefault("basename_index", {})
    m.setdefault("shards", {})
    m.setdefault("created_at", int(time.time()))
    m.setdefault("updated_at", int(time.time()))
    return m


def save_manifest(lmdb_folder: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(lmdb_folder, exist_ok=True)
    manifest["updated_at"] = int(time.time())
    mp = manifest_path(lmdb_folder)
    tmp = mp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, mp)


def basename_entry(manifest: Dict[str, Any], basename: str) -> Optional[Dict[str, Any]]:
    return manifest.get("basename_index", {}).get(basename)


def set_basename_entry(
    manifest: Dict[str, Any],
    basename: str,
    *,
    shard: str,
    sid: str,
    source_path: str,
) -> None:
    manifest.setdefault("basename_index", {})[basename] = {
        "shard": shard,
        "sid": sid,
        "source_path": source_path,
    }


def get_shard_stats(manifest: Dict[str, Any], shard_name: str) -> Dict[str, Any]:
    shards = manifest.setdefault("shards", {})
    if shard_name not in shards:
        shards[shard_name] = {
            "n_samples": 0,
            "estimated_bytes": 0,
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
        }
    return shards[shard_name]


def bump_shard_stats(manifest: Dict[str, Any], shard_name: str, add_bytes: int) -> None:
    s = get_shard_stats(manifest, shard_name)
    s["n_samples"] = int(s.get("n_samples", 0)) + 1
    s["estimated_bytes"] = int(s.get("estimated_bytes", 0)) + int(add_bytes)
    s["updated_at"] = int(time.time())
