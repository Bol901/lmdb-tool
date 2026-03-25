from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, Orientationd, Spacingd


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
    pl = torch.quantile(flat, 0.005).item()
    ph = torch.quantile(flat, 0.995).item()
    y = torch.clamp(x, pl, ph)
    denom = ph - pl + 1e-8
    y = (y - pl) / denom
    y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    return y.half(), {"p_low": float(pl), "p_high": float(ph)}


def _foreground_bbox_zyx(t: torch.Tensor, thr: float = 1e-3) -> List[int]:
    if t.ndim != 4 or t.shape[0] != 1:
        raise ValueError(f"expected (1,D,H,W), got {tuple(t.shape)}")
    m = (t[0] > thr).nonzero(as_tuple=False)
    if m.numel() == 0:
        d, h, w = t.shape[1:]
        return [0, d - 1, 0, h - 1, 0, w - 1]
    mn = m.min(dim=0).values.cpu().tolist()
    mx = m.max(dim=0).values.cpu().tolist()
    return [mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]]


def _affine_from_meta(t: Any) -> np.ndarray:
    if hasattr(t, "meta") and t.meta is not None and "affine" in t.meta:
        aff = t.meta["affine"]
        if isinstance(aff, torch.Tensor):
            return aff.detach().cpu().numpy().astype(np.float64)
        return np.asarray(aff, dtype=np.float64)
    return np.eye(4, dtype=np.float64)


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
    bbox_nat = _foreground_bbox_zyx(native_f16)

    spacing_iso = Compose(
        [
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ]
    )
    d_iso = spacing_iso({"image": vol_ras.detach().clone()})
    vol_iso = d_iso["image"]
    aff_iso = _affine_from_meta(vol_iso)
    iso_f16, st_iso = _percentile_clip_normalize_f16(vol_iso)
    bbox_iso = _foreground_bbox_zyx(iso_f16)

    iso_np = iso_f16.detach().cpu().numpy()
    nat_np = native_f16.detach().cpu().numpy()
    if iso_np.ndim != 4 or nat_np.ndim != 4:
        raise RuntimeError(f"bad shape iso={iso_np.shape} native={nat_np.shape}")

    meta = {
        "affine_iso": aff_iso,
        "affine_native": aff_native,
        "shape_iso": tuple(int(x) for x in iso_np.shape),
        "shape_native": tuple(int(x) for x in nat_np.shape),
        "bbox_iso": bbox_iso,
        "bbox_native": bbox_nat,
        "stats": {"iso": st_iso, "native": st_nat},
        "source_path": os.path.abspath(nifti_path),
    }
    return (
        iso_np.tobytes(),
        nat_np.tobytes(),
        pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL),
    )
