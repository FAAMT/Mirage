#!/usr/bin/env python3
# ------------------------------------------------------------
# Extract and slice INT-8 weights from a quantized Helsinki
# encoder_model.int8.onnx for a custom Transformer accelerator.
#
# Enhanced name searching and detailed debug output for anonymous search.
# ------------------------------------------------------------
from pathlib import Path
import sys
import re
import onnx
import numpy as np

# Regex for anonymous MatMul quantized weights
ANONYMOUS_MATMUL_PATTERN = re.compile(r"onnx::MatMul\d+_quantized")

class WeightExtractor:
    def __init__(self, model_path: Path, out_dir_path: Path):
        self.model_path = model_path
        self.out_dir = out_dir_path
        try:
            self.onnx_model = onnx.load(str(model_path))
            self.onnx_graph = self.onnx_model.graph
        except Exception as e:
            sys.exit(f"ERROR: Failed to load ONNX model from {model_path}. Details: {e}")
            
        self.used_anonymous_tensors = set()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.full_dump_file_path = self.out_dir / "weights_full_dump.txt"
        
        index_file_path = self.out_dir / "weights_index.txt"
        with open(index_file_path, "w", encoding="utf-8") as f:
            f.write(f"# Weights index from {self.model_path.name}\n")
            f.write("# Columns: Weight Name | Extracted Shape | Precision | Preview of first 8 values\n")
            f.write("----------------------|-----------------|-----------|--------------------------\n")
        print(f"Index file initialized: {index_file_path}")

    def _tensor_by_name(self, name: str):
        for t in self.onnx_graph.initializer:
            if t.name == name:
                return t
        return None

    @staticmethod
    def _is_int8(tensor: onnx.TensorProto) -> bool:
        return tensor.data_type == onnx.TensorProto.INT8

    @staticmethod
    def _get_shape(tensor: onnx.TensorProto) -> tuple:
        return tuple(int(d) for d in tensor.dims)

    def _slice_2d(self, tensor: onnx.TensorProto, rows: int = None, cols: int = None) -> np.ndarray:
        full_shape = self._get_shape(tensor)
        full_matrix = np.frombuffer(tensor.raw_data, dtype=np.int8).reshape(full_shape)
        if rows is None:
            rows = full_shape[0]
        if cols is None:
            cols = full_shape[1]
        return full_matrix[:rows, :cols].copy()

    def _dump_matrix(self, arr: np.ndarray, name: str, file_handle):
        lines = [f"\n── {name}  {arr.shape[0]}×{arr.shape[1]} ──"]
        for r_idx in range(arr.shape[0]):
            row_elems = " ".join(f"{int(x):4d}" for x in arr[r_idx])
            lines.append(f"row{r_idx:03d}: {row_elems}")
        text = "\n".join(lines) + "\n"
        file_handle.write(text)
        
        preview_rows = min(4, arr.shape[0])
        preview_cols = min(8, arr.shape[1])


    def _save_tensor_files(self, data: np.ndarray, name_stem_str: str,
                       scale_tensor: onnx.TensorProto, zp_tensor: onnx.TensorProto,
                       bias_tensor: onnx.TensorProto, full_dump_fh):
        name_stem_path = self.out_dir / name_stem_str

        # 1) Quantized ints
        data.astype(np.int8).tofile(name_stem_path.with_suffix(".bin"))

        # 2) Scale: accept ONNX TensorProto **or** sliced ndarray
        if isinstance(scale_tensor, np.ndarray):
            scale_arr = scale_tensor.astype(np.float32)
        elif scale_tensor.raw_data:
            scale_arr = np.frombuffer(scale_tensor.raw_data, dtype=np.float32)
        else:
            scale_arr = np.array(scale_tensor.float_data, dtype=np.float32)
        scale_arr.tofile(name_stem_path.with_suffix(".scale"))

        # 3) Zero point: same dual handling
        if isinstance(zp_tensor, np.ndarray):
            zp_arr = zp_tensor.astype(np.uint8)
        elif zp_tensor.raw_data:
            zp_arr = np.frombuffer(zp_tensor.raw_data, dtype=np.uint8)
        elif zp_tensor.int32_data:
            zp_arr = np.array(zp_tensor.int32_data, dtype=np.uint8)
        else:
            zp_arr = np.array(zp_tensor.int64_data, dtype=np.uint8)
        zp_arr.tofile(name_stem_path.with_suffix(".zp"))

        # 4) Save bias if present (FIXED: Added None check)
        if bias_tensor is not None:
            if hasattr(bias_tensor, 'raw_data') and bias_tensor.raw_data:
                bias_arr = np.frombuffer(bias_tensor.raw_data, dtype=np.float32)
            else:
                bias_arr = np.array(bias_tensor.float_data, dtype=np.float32)
            bias_arr.tofile(name_stem_path.with_suffix(".bias"))

        head_preview = data.flatten()[:8]
        with open(self.out_dir / "weights_index.txt", "a", encoding="utf-8") as cat_file:
            cat_file.write(f"{name_stem_path.name:<21} | {str(data.shape):<15} | int8      | {head_preview.tolist()}\n")

        print(f"✅ Saved {name_stem_path.name:<20} {data.shape!s:<15} weights. Head: {head_preview.tolist()}")
        self._dump_matrix(data, name_stem_path.name, full_dump_fh)

    def _get_quant_triplet(self, base_name: str):
        wq = self._tensor_by_name(base_name + "weight_quantized")
        sc = self._tensor_by_name(base_name + "weight_scale")
        zp = self._tensor_by_name(base_name + "weight_zero_point")
        bias = self._tensor_by_name(base_name + "bias")
        if all((wq, sc, zp)):
            return wq, sc, zp, bias
        return None, None, None, None

    def _find_anonymous_quant_triplet(self, desired_shape: tuple):
        print(f"    Searching for anonymous triplet with desired_shape: {desired_shape}")
        for t_wq_candidate in self.onnx_graph.initializer:
            cand_name = t_wq_candidate.name
            cand_shape = self._get_shape(t_wq_candidate)
            cand_dtype_val = t_wq_candidate.data_type
            cand_dtype_str = onnx.TensorProto.DataType.Name(cand_dtype_val)

            if cand_name in self.used_anonymous_tensors:
                continue

            if cand_shape != desired_shape:
                continue
                
            if not self._is_int8(t_wq_candidate):
                continue
            
            is_pattern_match = ANONYMOUS_MATMUL_PATTERN.fullmatch(cand_name) is not None
            if is_pattern_match:
                 print(f"        Candidate: {cand_name} (INT8, shape {cand_shape}) matches anonymous pattern.")
            else:
                 print(f"        Candidate: {cand_name} (INT8, shape {cand_shape}) does NOT match pattern '{ANONYMOUS_MATMUL_PATTERN.pattern}', but checking for scale/zp due to shape/type match.")

            expected_scale_name = cand_name.replace("_quantized", "_scale")
            expected_zp_name = cand_name.replace("_quantized", "_zero_point")
            
            t_sc = self._tensor_by_name(expected_scale_name)
            t_zp = self._tensor_by_name(expected_zp_name)

            if t_sc and t_zp:
                print(f"        SUCCESS: Found full triplet for {cand_name}!")
                self.used_anonymous_tensors.add(cand_name)
                return t_wq_candidate, t_sc, t_zp, None  # No bias for anonymous tensors
            else:
                fail_reason = []
                if not t_sc: fail_reason.append(f"Scale '{expected_scale_name}' not found")
                if not t_zp: fail_reason.append(f"ZP '{expected_zp_name}' not found")
                print(f"        SKIPPING {cand_name}: {'; '.join(fail_reason)}.")
        
        print(f"    Finished searching for anonymous triplet with shape {desired_shape}. No suitable triplet found.")
        return None, None, None, None
        
    def _print_available_initializers_and_exit(self, failed_component: str, attempted_shape_for_anon_search):
        print(f"\n--- ERROR: Failed to find INT-8 weights for {failed_component} ---")
        if attempted_shape_for_anon_search:
            print(f"--- Last attempted anonymous search with shape: {attempted_shape_for_anon_search}")
        print("\n--- AVAILABLE INITIALIZERS (POTENTIAL WEIGHTS) IN ONNX GRAPH ---")
        if not self.onnx_graph.initializer:
            print("  No initializers found in the ONNX graph! Model might be structured differently or empty.")
        
        sorted_initializers = sorted(self.onnx_graph.initializer, key=lambda t: t.name)

        for i, t in enumerate(sorted_initializers):
            tensor_type_str = onnx.TensorProto.DataType.Name(t.data_type)
            print(f"  Name: {t.name:<70} Shape: {str(self._get_shape(t)):<20} DataType: {tensor_type_str} ({t.data_type})")
            if i > 200 and len(sorted_initializers) > 220:
                print(f"  ... (and {len(sorted_initializers) - (i+1)} more initializers)")
                break
        print("--- END AVAILABLE INITIALIZERS ---\n")
        sys.exit(f"❌ Cannot locate INT-8 weights for {failed_component}. Please review the initializer list above to find the correct names/shapes and adjust the script if necessary.")

    def extract_weights(self):
        """
        Dump per-layer Q, K, V, Wo, FC1 and FC2 weights (full and per-head where relevant)
        ― each written as layer{idx}_<name>*.{bin,scale,zp,bias}.
        """
        import re
        print(f"🔍 Scanning {self.model_path.name}…")
        with open(self.full_dump_file_path, "w", encoding="utf-8") as full_dump_fh:
            full_dump_fh.write(f"# Full weight dump from {self.model_path.name}\n")
            full_dump_fh.write("# Target d_model=256, num_heads=8, head_dim=64, d_ff=512\n")

            # ── discover how many encoder layers are present ─────────────────────
            layer_indices = sorted(
                {
                    int(m.group(1))
                    for t in self.onnx_model.graph.initializer
                    if (m := re.match(r"(?:encoder\.)?layers\.(\d+)\.", t.name))
                }
            )
            if not layer_indices:                           # fallback → single layer model
                layer_indices = [0]

            NUM_HEADS, HEAD_DIM = 8, 64

            # ───────────────────────── main per-layer loop ──────────────────────
            for layer_idx in layer_indices:
                print(f"\n================= Layer {layer_idx} =================")
                current_onnx_qkv_full_shape = (512, 512)

                # ---------- QKV --------------------------------------------------
                print("\n--- Extracting QKV Weights ---")
                for proj_type in ("query", "key", "value"):
                    print(f"  Processing {proj_type} projection…")
                    helsinki_proj_suffix = proj_type[0] + "_proj."
                    wq_full = sc_full = zp_full = bias = None
                    found_named = False

                    qkv_prefixes_to_try = [
                        f"layers.{layer_idx}.self_attn.",
                        f"encoder.layers.{layer_idx}.self_attn.",
                    ]
                    for prefix_base in qkv_prefixes_to_try:
                        full_name_base_attempt = prefix_base + helsinki_proj_suffix
                        wq_full, sc_full, zp_full, bias = self._get_quant_triplet(full_name_base_attempt)
                        if wq_full is not None:
                            print(f"    ✓ {full_name_base_attempt} → {self._get_shape(wq_full)}")
                            current_onnx_qkv_full_shape = self._get_shape(wq_full)
                            found_named = True
                            break
                    if not found_named:
                        print("    ↪ anonymous fallback …")
                        wq_full, sc_full, zp_full, bias = self._find_anonymous_quant_triplet(
                            current_onnx_qkv_full_shape
                        )
                    if wq_full is None or not self._is_int8(wq_full):
                        self._print_available_initializers_and_exit(
                            f"Layer {layer_idx} QKV {proj_type}", current_onnx_qkv_full_shape
                        )

                    full_matrix = np.frombuffer(wq_full.raw_data, dtype=np.int8).reshape(
                        self._get_shape(wq_full)
                    )
                    scale_full = (
                        np.frombuffer(sc_full.raw_data, dtype=np.float32)
                        if sc_full.raw_data
                        else np.asarray(sc_full.float_data, dtype=np.float32)
                    )
                    zp_full_arr = (
                        np.frombuffer(zp_full.raw_data, dtype=np.uint8)
                        if zp_full.raw_data
                        else np.asarray(
                            zp_full.int32_data or zp_full.int64_data, dtype=np.uint8
                        )
                    )

                    # per-head slices
                    for head_idx in range(NUM_HEADS):
                        row_slice = slice(head_idx * HEAD_DIM, (head_idx + 1) * HEAD_DIM)
                        self._save_tensor_files(
                            full_matrix[row_slice, :],
                            f"layer{layer_idx}_{proj_type}_head{head_idx}",
                            scale_full.copy(),
                            zp_full_arr.copy(),
                            bias,
                            full_dump_fh,
                        )

                    # full projection
                    self._save_tensor_files(
                        full_matrix,
                        f"layer{layer_idx}_{proj_type}_full",
                        sc_full,
                        zp_full,
                        bias,
                        full_dump_fh,
                    )

                # ---------- Wo ---------------------------------------------------
                print("\n--- Extracting Attention Output Projection (Wo) ---")
                wo_prefixes_to_try = [
                    f"layers.{layer_idx}.self_attn.out_proj.",
                    f"encoder.layers.{layer_idx}.self_attn.out_proj.",
                    f"encoder.layers.{layer_idx}.attention.output.dense.",
                ]
                wq_wo = sc_wo = zp_wo = bias_wo = None
                for prefix_base in wo_prefixes_to_try:
                    wq_wo, sc_wo, zp_wo, bias_wo = self._get_quant_triplet(prefix_base)
                    if wq_wo is not None:
                        print(f"    ✓ {prefix_base} → {self._get_shape(wq_wo)}")
                        break
                if wq_wo is None:
                    print("    ↪ anonymous fallback …")
                    wq_wo, sc_wo, zp_wo, bias_wo = self._find_anonymous_quant_triplet(
                        current_onnx_qkv_full_shape
                    )
                if wq_wo is None or not self._is_int8(wq_wo):
                    self._print_available_initializers_and_exit(
                        f"Layer {layer_idx} Wo", current_onnx_qkv_full_shape
                    )
                self._save_tensor_files(
                    np.frombuffer(wq_wo.raw_data, dtype=np.int8).reshape(
                        self._get_shape(wq_wo)
                    ),
                    f"layer{layer_idx}_attn_out_proj_full",
                    sc_wo,
                    zp_wo,
                    bias_wo,
                    full_dump_fh,
                )

                # ---------- FFN (FC1 / FC2) --------------------------------------
                print("\n--- Extracting FFN Weights ---")
                ffn_specs = [
                    (
                        "FC1",
                        [
                            f"layers.{layer_idx}.fc1.",
                            f"encoder.layers.{layer_idx}.intermediate.dense.",
                        ],
                        (2048, 512),
                        f"layer{layer_idx}_ffn_fc1",
                    ),
                    (
                        "FC2",
                        [
                            f"layers.{layer_idx}.fc2.",
                            f"encoder.layers.{layer_idx}.output.dense.",
                        ],
                        (512, 2048),
                        f"layer{layer_idx}_ffn_fc2",
                    ),
                ]
                for layer_name, prefixes, default_shape, file_stem in ffn_specs:
                    print(f"  Processing FFN {layer_name}…")
                    wq_fc = sc_fc = zp_fc = bias_fc = None
                    for prefix_base in prefixes:
                        wq_fc, sc_fc, zp_fc, bias_fc = self._get_quant_triplet(prefix_base)
                        if wq_fc is not None:
                            print(f"    ✓ {prefix_base} → {self._get_shape(wq_fc)}")
                            break
                    if wq_fc is None:
                        print("    ↪ anonymous fallback …")
                        wq_fc, sc_fc, zp_fc, bias_fc = self._find_anonymous_quant_triplet(
                            default_shape
                        )
                    if wq_fc is None or not self._is_int8(wq_fc):
                        self._print_available_initializers_and_exit(
                            f"Layer {layer_idx} FFN {layer_name}", default_shape
                        )
                    self._save_tensor_files(
                        np.frombuffer(wq_fc.raw_data, dtype=np.int8).reshape(
                            self._get_shape(wq_fc)
                        ),
                        f"{file_stem}_full",
                        sc_fc,
                        zp_fc,
                        bias_fc,
                        full_dump_fh,
                    )

        print(
            f"\n✏️ Done. Details dumped to:\n  {self.full_dump_file_path}\n  {self.out_dir / 'weights_index.txt'}"
        )
        print(f"Binary files are in: {self.out_dir}")

def  main():
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} model.int8.onnx output_directory")
    
    model_file = Path(sys.argv[1]).expanduser().resolve()
    output_dir = Path(sys.argv[2]).expanduser().resolve()

    if not model_file.is_file():
        sys.exit(f"ERROR: Model file not found: {model_file}")

    print(f"Input ONNX model: {model_file}")
    print(f"Output directory: {output_dir}")

    extractor = WeightExtractor(model_file, output_dir)
    extractor.extract_weights()

if __name__ == "__main__":
    main()