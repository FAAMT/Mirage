#!/usr/bin/env python3
# slice_teacher.py  ----------------------------------------------------------
# Copy sliced weights from the INT8 teacher used in translate.py
#   Q,K,V : 64×256
#   fc1   : 128×64
#   fc2   : 256×128
# and store them in   student_ckpt/sliced_init/
# ---------------------------------------------------------------------------
import sys, torch
from pathlib import Path
from transformers import MarianMTModel

# ---------- pick up the teacher path from translate.py ---------------------
TEACHER_DIR = Path("/Users/faamt/Desktop/Nomadic Inc/Helsinki/opus-mt-en-ar")

if not TEACHER_DIR.exists():
    sys.exit(f"❌ Teacher directory not found: {TEACHER_DIR}")

def slice_mat(t: torch.Tensor, rows: int, cols: int):
    return t[:rows, :cols].clone()

def main(student_ckpt: Path):
    teacher = MarianMTModel.from_pretrained(TEACHER_DIR, torch_dtype=torch.float32)
    student = MarianMTModel.from_pretrained(student_ckpt, torch_dtype=torch.float32)

    sd_t = teacher.state_dict()
    sd_s = student.state_dict()
    
    # --- helper ---------------------------------------------------------------
    def copy_slice(dst: torch.Tensor, src: torch.Tensor):
        """in-place copy of src into the top-left corner of dst (no size mismatch)."""
        r, c = src.shape
        dst[:r, :c] = src        # all other elements of dst stay as-is
    # --------------------------------------------------------------------------

    # ---- Q / K / V -----------------------------------------------------------
    for proj in ("q_proj", "k_proj", "v_proj"):
        k = f"model.encoder.layers.0.self_attn.{proj}.weight"
        copy_slice(sd_s[k], slice_mat(sd_t[k], 64, 256))   # 64×256 into 256×256

    # ---- fc1 : 128×64  (student tensor is 128×256) ---------------------------
    k = "model.encoder.layers.0.fc1.weight"
    copy_slice(sd_s[k], slice_mat(sd_t[k], 128, 64))

    # ---- fc2 : 256×128 (student tensor is 256×256) ---------------------------
    k = "model.encoder.layers.0.fc2.weight"
    copy_slice(sd_s[k], slice_mat(sd_t[k], 256, 128))

    student.load_state_dict(sd_s, strict=False)
    out = student_ckpt / "sliced_init"
    student.save_pretrained(out)

    print(f"✅  sliced weights saved to {out.resolve()}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage:  python slice_teacher.py  student_ckpt")
    main(Path(sys.argv[1]).expanduser())
