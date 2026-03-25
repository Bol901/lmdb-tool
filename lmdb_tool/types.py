from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

MANIFEST_NAME = "manifest.json"
INGEST_UNSYNCED_PATHS_JSON = "ingest_unsynced_paths.json"
SHARD_PREFIX = "shard_"
SHARD_SUFFIX = ".lmdb"
DEFAULT_MAX_SHARD_SIZE_BYTES = 500 * 1024 * 1024 * 1024
DEFAULT_MAP_SIZE = 5 << 40
KEY_KEYS = b"__keys__"
SUFFIX_ISO = "_iso1mm"
SUFFIX_NATIVE = "_native"
SUFFIX_META = "_meta"
MANIFEST_VERSION = 1


@dataclass
class IngestStats:
    total_inputs: int = 0
    unique_basenames_seen: int = 0
    already_present: int = 0
    newly_ingested: int = 0
    failed: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_total_inputs": self.total_inputs,
            "n_unique_basenames_seen": self.unique_basenames_seen,
            "n_already_present": self.already_present,
            "n_newly_ingested": self.newly_ingested,
            "n_failed": self.failed,
        }
