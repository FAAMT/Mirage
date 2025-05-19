#!/usr/bin/env python3
# ------------------------------------------------------------
#  Extract INT-8 Q / K / V  (64×64)
#          FC-1 hidden      (128×64)
#          FC-2 output      ( 64×64)
#  from a quantised Helsinki *encoder_model.int8.onnx*
#  and dump every weight in human-readable form
# ------------------------------------------------------------
from pathlib import Path
import sys, re, onnx, numpy as np

_pat  = re.compile(r"onnx::MatMul_\d+_quantized")
_used = set()                       # anonymous MatMul already taken
_full_dump_file = None              # initialised in extract()

# ---------- helpers ----------------------------------------------------------
def tensor_by_name(g, name):
    for t in g.graph.initializer:
        if t.name == name:
            return t
    return None

def is_int8(t):  return t.data_type == onnx.TensorProto.INT8
def shape(t):    return tuple(int(d) for d in t.dims)

def slice_2d(t, rows, cols):
    full = np.frombuffer(t.raw_data, dtype=np.int8).reshape(shape(t))
    return full[:rows, :cols]

def dump_matrix(arr: np.ndarray, name: str):
    """Print and append the full matrix with row/col labels."""
    global _full_dump_file
    lines = [f"\n── {name}  {arr.shape[0]}×{arr.shape[1]} ──"]
    for r in range(arr.shape[0]):
        row_elems = " ".join(f"{int(x):4d}" for x in arr[r])
        lines.append(f"row{r:02d}: {row_elems}")
    text = "\n".join(lines) + "\n"
    print(text)
    _full_dump_file.write(text)

def save_tensor(data: np.ndarray, stem: Path, sc, zp):
    """binary/scale/zp + preview + full dump"""
    stem = stem.with_suffix("")
    data.astype(np.int8 ).tofile(stem.with_suffix(".bin"))
    np.frombuffer(sc.raw_data, dtype=np.float32).tofile(stem.with_suffix(".scale"))
    np.frombuffer(zp.raw_data, dtype=np.uint8 ).tofile(stem.with_suffix(".zp"))

    head = data.flatten()[:8]
    with open(stem.parent / "weights_index.txt", "a") as cat:
        cat.write(f"{stem.name:<10} {data.shape!s:<13} int8  head→ {head.tolist()}\n")
    print(f"✅ {stem.name:<10} {data.size:6} weights  head→ {head.tolist()}")

    dump_matrix(data, stem.name)

def get_triplet(g, base):
    wq = tensor_by_name(g, base + "weight_quantized")
    sc = tensor_by_name(g, base + "weight_scale")
    zp = tensor_by_name(g, base + "weight_zero_point")
    return wq, sc, zp if all((wq, sc, zp)) else (None, None, None)

def find_anonymous(g, wanted_shape):
    for t in g.graph.initializer:
        if t.name in _used or not _pat.fullmatch(t.name) or shape(t) != wanted_shape:
            continue
        sc = tensor_by_name(g, t.name.replace("_quantized", "_scale"))
        zp = tensor_by_name(g, t.name.replace("_quantized", "_zero_point"))
        if sc and zp:
            _used.add(t.name)
            return t, sc, zp
    return None, None, None

# ---------- main -------------------------------------------------------------
def extract(model: Path, out_dir: Path):
    global _full_dump_file
    print(f"🔍  scanning {model.name} …")
    out_dir.mkdir(parents=True, exist_ok=True)
    _full_dump_file = open(out_dir / "weights_full_dump.txt", "w", encoding="utf-8")

    g = onnx.load(str(model))

    # -------- Q / K / V ------------------------------------------------------
    prefix = "layers.0.self_attn."
    for proj in ("q_proj.", "k_proj.", "v_proj."):
        wq, sc, zp = get_triplet(g, prefix + proj)
        if wq is None:               # optimiser renamed – anonymous fallback
            wq, sc, zp = find_anonymous(g, (512, 512))
        if wq is None or not is_int8(wq):
            raise RuntimeError(f"❌ cannot locate INT-8 weights for {proj}")
        sliced = slice_2d(wq, 64, 256)     # 64 × 256
        save_tensor(sliced, out_dir / proj[:-1], sc, zp)

    # -------- FC-1 128×64  and  FC-2 64×128 ----------------------------------
    fc_spec = (("fc1.", (512, 2048), 128, 64),
               ("fc2.", (2048, 512),  64, 128))
    for tag, full_shape, keep_r, keep_c in fc_spec:
        wq, sc, zp = get_triplet(g, "layers.0." + tag)
        if wq is None:
            wq, sc, zp = find_anonymous(g, full_shape)
        if wq is None or not is_int8(wq):
            raise RuntimeError(f"❌ cannot locate INT-8 weights for {tag}")
        sliced = slice_2d(wq, keep_r, keep_c)
        save_tensor(sliced, out_dir / tag.rstrip("."), sc, zp)

    _full_dump_file.close()
    print(f"✏️  Done – see {out_dir/'weights_full_dump.txt'}  and  weights_index.txt")

# ---------- CLI --------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage:  extract_int8_weights.py  model.int8.onnx  out_dir")
    extract(Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser())
