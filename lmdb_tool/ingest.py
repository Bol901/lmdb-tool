from __future__ import annotations

import argparse
import logging

from .core import ingest_json_incremental
from .types import DEFAULT_MAP_SIZE, DEFAULT_MAX_SHARD_SIZE_BYTES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Incremental sharded LMDB ingester")
    p.add_argument("--json", required=True, help="input JSON list")
    p.add_argument("--root-dir", required=True, help="root dir for relative paths")
    p.add_argument("--lmdb-folder", required=True, help="sharded lmdb folder")
    p.add_argument("--max-shard-size-gb", type=float, default=500.0)
    p.add_argument("--map-size", type=int, default=DEFAULT_MAP_SIZE)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--sync-every", type=int, default=64, help="每 N 条成功写入后 env.sync + 保存 manifest；崩溃时可查 ingest_unsynced_paths.json")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    max_bytes = int(args.max_shard_size_gb * 1024 * 1024 * 1024)
    if max_bytes <= 0:
        max_bytes = DEFAULT_MAX_SHARD_SIZE_BYTES
    stats = ingest_json_incremental(
        json_path=args.json,
        root_dir=args.root_dir,
        lmdb_folder=args.lmdb_folder,
        max_shard_size_bytes=max_bytes,
        map_size=args.map_size,
        workers=args.workers,
        sync_every=args.sync_every,
        log_every=args.log_every,
        dry_run=args.dry_run,
    )
    print(stats)


if __name__ == "__main__":
    main()
