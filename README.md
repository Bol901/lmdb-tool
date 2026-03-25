# lmdb_tool

Incremental sharded LMDB maintainer for MRI NIfTI datasets.

## Install

```bash
pip install "git+https://github.com/Bol901/lmdb-tool.git"
```

## CLI

```bash
lmdb-tool \
  --json /path/to/list.json \
  --root-dir /path/to/nifti_root \
  --lmdb-folder /path/to/cache \
  --max-shard-size-gb 500 \
  --workers 128 \
  --log-every 1000
```

- dedup key: basename (global)
- shards: `shard_000.lmdb`, `shard_001.lmdb`, ...
- manifest: `manifest.json`
