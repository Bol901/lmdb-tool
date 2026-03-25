from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, Orientationd, Spacingd

# 预处理阶段优先稳定性：分位数一律走 numpy（必要时抽样），避免 torch.quantile 的尺寸限制与潜在不一致。
MAX_PERCENTILE_NUMPY = 2_000_000

INT16_MIN = -32768.0
INT16_MAX = 32767.0
INT16_RANGE = INT16_MAX - INT16_MIN  # 65535.0


def resolve_nifti_path(item: Any, data_root: Optional[str]) -> str:
    if isinstance(item, dict):
        if "path" in item:
            raw = str(item["path"]).strip()
        elif "image" in item:
            raw = str(item["image"]).strip()
        else:
            raw = str(item).strip()
    else:
        raw = str(item).strip()
    if os.path.isabs(raw):
        return raw
    rel = raw.lstrip("/")
    if data_root is not None:
        return os.path.join(data_root.rstrip("/"), rel)
    return rel


def _normalize_rel_key(nifti_abs: str, data_root: Optional[str]) -> str:
    ap = os.path.normpath(os.path.abspath(nifti_abs))
    if data_root:
        dr = os.path.normpath(os.path.abspath(data_root.rstrip("/")))
        try:
            rel = os.path.relpath(ap, dr)
        except ValueError:
            rel = ap
    else:
        rel = ap
    return rel.replace(os.sep, "/")


def subject_id_from_path(nifti_abs: str, data_root: Optional[str]) -> str:
    rel = _normalize_rel_key(nifti_abs, data_root)
    return hashlib.sha256(rel.encode("utf-8")).hexdigest()[:32]


def _percentile_clip_normalize_f16(t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    x = t.detach().float()
    flat = x.flatten()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        z = torch.zeros_like(x, dtype=torch.float16)
        return z, {"p_low": 0.0, "p_high": 1.0}
    # 稳定优先：分位数使用 numpy.percentile；大样本用等间隔抽样控制内存与耗时。
    if flat.numel() > MAX_PERCENTILE_NUMPY:
        step = max(1, flat.numel() // MAX_PERCENTILE_NUMPY)
        flat = flat[::step]
    arr = flat.detach().cpu().numpy()
    pl = float(np.percentile(arr, 0.5))
    ph = float(np.percentile(arr, 99.5))
    y = torch.clamp(x, pl, ph)
    denom = ph - pl + 1e-8
    y = (y - pl) / denom
    y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    return y.half(), {"p_low": float(pl), "p_high": float(ph)}


def _affine_from_meta(t: Any) -> np.ndarray:
    if hasattr(t, "meta") and t.meta is not None and "affine" in t.meta:
        aff = t.meta["affine"]
        if isinstance(aff, torch.Tensor):
            return aff.detach().cpu().numpy().astype(np.float64)
        return np.asarray(aff, dtype=np.float64)
    return np.eye(4, dtype=np.float64)


def process_h5_to_payloads(
    h5_path: str,
    *,
    source_nifti_path: Optional[str] = None,
) -> Tuple[bytes, bytes, bytes]:
    """Read precomputed H5 and convert back to float16 normalized tensors (0-1).

    Your legacy `nifti_to_h5.py` stores:
      - image/native and image/iso1mm as int16 clipped then mapped to full int16 range
    So this function only needs the inverse mapping from int16 range to [0, 1].
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(h5_path)

    h5_abs = os.path.abspath(h5_path)
    src = os.path.abspath(source_nifti_path) if source_nifti_path else h5_abs

    with h5py.File(h5_abs, "r") as f:
        nat_i16 = f["image/native"][:]  # (D,H,W), int16
        iso_i16 = f["image/iso1mm"][:]  # (D,H,W), int16

        aff_native = f["meta/native/affine"][:]  # (4,4), float32
        aff_iso = f["meta/iso1mm/affine"][:]  # (4,4), float32

        nat_clip_p = f["meta/native/intensity/clip_p"][:]  # (2,)
        iso_clip_p = f["meta/iso1mm/intensity/clip_p"][:]  # (2,)

    # int16 full-range -> [0,1] normalized float
    nat_f = (nat_i16.astype(np.float32) - INT16_MIN) / INT16_RANGE
    iso_f = (iso_i16.astype(np.float32) - INT16_MIN) / INT16_RANGE
    nat_f = np.clip(nat_f, 0.0, 1.0)
    iso_f = np.clip(iso_f, 0.0, 1.0)

    # Match nifti path preprocessing output shape: (1,D,H,W) and dtype float16
    nat_np = nat_f.astype(np.float16)[None, ...]
    iso_np = iso_f.astype(np.float16)[None, ...]

    meta = {
        "affine_iso": aff_iso.astype(np.float64),
        "affine_native": aff_native.astype(np.float64),
        "shape_iso": tuple(int(x) for x in iso_np.shape),
        "shape_native": tuple(int(x) for x in nat_np.shape),
        "stats": {
            "iso": {"p_low": float(iso_clip_p[0]), "p_high": float(iso_clip_p[1])},
            "native": {"p_low": float(nat_clip_p[0]), "p_high": float(nat_clip_p[1])},
        },
        "source_path": src,
    }

    return (
        iso_np.tobytes(),
        nat_np.tobytes(),
        pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL),
    )


def process_nifti_to_payloads(nifti_path: str) -> Tuple[bytes, bytes, bytes]:
    if not os.path.isfile(nifti_path):
        raise FileNotFoundError(nifti_path)

    load_ras = Compose(
        [
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            Orientationd(keys=["image"], axcodes="RAS"),
        ]
    )
    d = load_ras({"image": nifti_path})
    vol_ras: torch.Tensor = d["image"]
    aff_native = _affine_from_meta(vol_ras)

    native_t = vol_ras.detach().clone()
    native_f16, st_nat = _percentile_clip_normalize_f16(native_t)

    spacing_iso = Compose(
        [
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ]
    )
    d_iso = spacing_iso({"image": vol_ras.detach().clone()})
    vol_iso = d_iso["image"]
    aff_iso = _affine_from_meta(vol_iso)
    iso_f16, st_iso = _percentile_clip_normalize_f16(vol_iso)

    iso_np = iso_f16.detach().cpu().numpy()
    nat_np = native_f16.detach().cpu().numpy()
    if iso_np.ndim != 4 or nat_np.ndim != 4:
        raise RuntimeError(f"bad shape iso={iso_np.shape} native={nat_np.shape}")

    meta = {
        "affine_iso": aff_iso,
        "affine_native": aff_native,
        "shape_iso": tuple(int(x) for x in iso_np.shape),
        "shape_native": tuple(int(x) for x in nat_np.shape),
        "stats": {"iso": st_iso, "native": st_nat},
        "source_path": os.path.abspath(nifti_path),
    }
    return (
        iso_np.tobytes(),
        nat_np.tobytes(),
        pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL),
    )
