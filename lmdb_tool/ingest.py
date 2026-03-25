from __future__ import annotations

import argparse
import logging

from .core import ingest_json_incremental
from .types import DEFAULT_MAP_SIZE, DEFAULT_MAX_SHARD_SIZE_BYTES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Incremental sharded LMDB ingester")
    p.add_argument(
        "--json",
        required=True,
        help="输入：训练列表 JSON（顶层 list）；或 ingest_unsynced_paths.json（{\"paths\":[...]}）",
    )
    p.add_argument("--root-dir", required=True, help="root dir for relative paths")
    p.add_argument("--lmdb-folder", required=True, help="sharded lmdb folder")
    p.add_argument("--max-shard-size-gb", type=float, default=500.0)
    p.add_argument("--map-size", type=int, default=DEFAULT_MAP_SIZE)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--sync-every", type=int, default=64, help="每 N 条成功写入后 env.sync + 保存 manifest；崩溃时可查 ingest_unsynced_paths.json")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--h5-root-folder",
        type=str,
        default=None,
        help="如果提供，且输入 nifti 路径存在对应 h5（按 root_dir 映射规则），则从 h5 读取并仅补全 0-1 归一化。",
    )
    p.add_argument(
        "--max-error-logs",
        type=int,
        default=20,
        metavar="N",
        help="最多打印前 N 条失败样本的异常日志（避免刷屏）",
    )
    p.add_argument(
        "--max-ingest",
        type=int,
        default=None,
        metavar="N",
        help="最多 ingest 的新样本条数（不含 manifest 已存在项；用于小批试跑）",
    )
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
        max_ingest=args.max_ingest,
        h5_root_folder=args.h5_root_folder,
        max_error_logs=args.max_error_logs,
    )
    print(stats)


if __name__ == "__main__":
    main()
