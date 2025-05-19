#!/usr/bin/env python3
#  extract_int8_weights.py  — Q-/K-/V + single-FFN weights (INT-8)

from pathlib import Path
import re, sys, onnx, numpy as np

# ───────── helper ------------------------------------------------------------
def tensor_by_name(g, name):
    return next((t for t in g.graph.initializer if t.name == name), None)

_used = set()
_pat  = re.compile(r"onnx::MatMul_\d+_quantized")

def find_anonymous(g, shape):
    """Return next unused MatMul_*_quantized tensor of given shape."""
    for t in g.graph.initializer:
        if t.name in _used:
            continue
        if _pat.fullmatch(t.name) and tuple(t.dims) == shape:
            sc = tensor_by_name(g, t.name.replace("_quantized", "_scale"))
            zp = tensor_by_name(g, t.name.replace("_quantized", "_zero_point"))
            if sc and zp:
                _used.add(t.name)
                return t, sc, zp
    return None, None, None

def get_quant(g, prefix, shape):
    """Return weight/scale/zp tensor triple – named OR anonymous fallback."""
    wq = tensor_by_name(g, prefix + "weight_quantized")
    if wq is None:
        return find_anonymous(g, shape)
    sc = tensor_by_name(g, prefix + "weight_scale")
    zp = tensor_by_name(g, prefix + "weight_zero_point")
    return wq, sc, zp

def export_tensor(wq, sc, zp, stem: Path):
    stem = stem.with_suffix("")
    arr  = np.frombuffer(wq.raw_data, dtype=np.int8)
    arr.tofile(stem.with_suffix(".bin"))
    np.frombuffer(sc.raw_data, dtype=np.float32).tofile(stem.with_suffix(".scale"))
    np.frombuffer(zp.raw_data,  dtype=np.uint8).tofile(stem.with_suffix(".zp"))
    with open(stem.parent / "weights_index.txt", "a") as f:
        f.write(f"{stem.name:<20} {list(wq.dims)} int8\n")
    print(f"✅ {stem.name}: {arr.size} weights   head→ {arr[:8].tolist()}")

# ───────── extraction routine ------------------------------------------------
def extract_weights(model_path: Path, dst_dir: Path):
    print(f"🔍  scanning {model_path.name} …")
    dst_dir.mkdir(parents=True, exist_ok=True)
    g = onnx.load_model(str(model_path))

    # ---- Q / K / V for 6 encoder layers ---------------------------------
    for layer in range(6):
        base = f"layers.{layer}.self_attn."
        for tag in ("q_proj.", "k_proj.", "v_proj."):
            wq, sc, zp = get_quant(g, base + tag, (512, 512))
            if wq is None:
                raise RuntimeError(f"❌ missing {base+tag} weights")
            export_tensor(wq, sc, zp, dst_dir / f"layer{layer}_{tag[:-1]}")

    # ---- Single FFN (fc1 / fc2) – use first layer only ------------------
    base = "layers.0."
    wq1, sc1, zp1 = get_quant(g, base + "fc1.", (512, 2048))
    wq2, sc2, zp2 = get_quant(g, base + "fc2.", (2048, 512))
    if not (wq1 and wq2):
        raise RuntimeError("❌ could not locate fc1 / fc2 weights")
    export_tensor(wq1, sc1, zp1, dst_dir / "fc1")
    export_tensor(wq2, sc2, zp2, dst_dir / "fc2")

    print(f"✏️  all weights dumped – see {dst_dir/'weights_index.txt'}")

# ───────── main --------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage:  extract_int8_weights.py <model.int8.onnx> <output_dir>")
    extract_weights(Path(sys.argv[1]).expanduser(),
                    Path(sys.argv[2]).expanduser())
