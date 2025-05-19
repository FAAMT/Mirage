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

    def _slice_2d(self, tensor: onnx.TensorProto, rows: int, cols: int) -> np.ndarray:
        full_shape = self._get_shape(tensor)
        if not (rows <= full_shape[0] and cols <= full_shape[1]):
            # This call will print initializers and exit if slicing is invalid
            self._print_available_initializers_and_exit(
                f"Slicing {tensor.name} (shape {full_shape}) with target ({rows},{cols})",
                full_shape 
            )
        full_matrix = np.frombuffer(tensor.raw_data, dtype=np.int8).reshape(full_shape)
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
        # Ensure console output for dump preview is minimal but informative
        # print(f"    Preview of {name} (shape {arr.shape}):") # Can be noisy, let save_tensor_files handle main log
        # for r_idx in range(preview_rows):
        #     row_elems_preview = " ".join(f"{int(x):4d}" for x in arr[r_idx, :preview_cols])
        #     ellipsis = " ..." if arr.shape[1] > preview_cols else ""
        #     print(f"      row{r_idx:03d}: {row_elems_preview}{ellipsis}")
        # if arr.shape[0] > preview_rows:
        #     print("      ...")


    def _save_tensor_files(self, data: np.ndarray, name_stem_str: str,
                           scale_tensor: onnx.TensorProto, zp_tensor: onnx.TensorProto,
                           full_dump_fh):
        name_stem_path = self.out_dir / name_stem_str

        # 1) Quantized ints
        data.astype(np.int8).tofile(name_stem_path.with_suffix(".bin"))

        # 2) Scale: try raw_data, else float_data
        if scale_tensor.raw_data:
            scale_arr = np.frombuffer(scale_tensor.raw_data, dtype=np.float32)
        else:
            scale_arr = np.array(scale_tensor.float_data, dtype=np.float32)
        scale_arr.tofile(name_stem_path.with_suffix(".scale"))

        # 3) Zero point: try raw_data, else int32_data or int64_data
        if zp_tensor.raw_data:
            zp_arr = np.frombuffer(zp_tensor.raw_data, dtype=np.uint8)
        elif zp_tensor.int32_data:
            zp_arr = np.array(zp_tensor.int32_data, dtype=np.uint8)
        else:
            zp_arr = np.array(zp_tensor.int64_data, dtype=np.uint8)
        zp_arr.tofile(name_stem_path.with_suffix(".zp"))

        head_preview = data.flatten()[:8]
        with open(self.out_dir / "weights_index.txt", "a", encoding="utf-8") as cat_file:
            cat_file.write(f"{name_stem_path.name:<21} | {str(data.shape):<15} | int8      | {head_preview.tolist()}\n")
        
        print(f"✅ Saved {name_stem_path.name:<20} {data.shape!s:<15} weights. Head: {head_preview.tolist()}")
        self._dump_matrix(data, name_stem_path.name, full_dump_fh)

    def _get_quant_triplet(self, base_name: str):
        # print(f"Attempting to get triplet for base name: {base_name}") # DEBUG
        wq = self._tensor_by_name(base_name + "weight_quantized")
        sc = self._tensor_by_name(base_name + "weight_scale")
        zp = self._tensor_by_name(base_name + "weight_zero_point")
        if all((wq, sc, zp)):
            # print(f"  Found triplet for {base_name}") # DEBUG
            return wq, sc, zp
        # print(f"  Triplet not found for {base_name}") # DEBUG
        return None, None, None

    def _find_anonymous_quant_triplet(self, desired_shape: tuple):
        print(f"    Searching for anonymous triplet with desired_shape: {desired_shape}")
        for t_wq_candidate in self.onnx_graph.initializer:
            cand_name = t_wq_candidate.name
            cand_shape = self._get_shape(t_wq_candidate)
            cand_dtype_val = t_wq_candidate.data_type
            cand_dtype_str = onnx.TensorProto.DataType.Name(cand_dtype_val)

            # Uncomment for extreme verbosity if needed:
            # print(f"      Considering cand: {cand_name}, shape: {cand_shape}, dtype: {cand_dtype_str} ({cand_dtype_val})")

            if cand_name in self.used_anonymous_tensors:
                # print(f"        SKIP: {cand_name} (shape {cand_shape}) already used.")
                continue

            if cand_shape != desired_shape:
                # print(f"        SKIP: {cand_name} shape {cand_shape} != desired {desired_shape}.")
                continue
                
            if not self._is_int8(t_wq_candidate):
                # print(f"        SKIP: {cand_name} (shape {cand_shape}) is not INT8, it's {cand_dtype_str}.")
                continue
            
            # At this point, shape matches and it's INT8 and not used.
            # Now, check if the name matches the specific anonymous pattern, primarily for logging.
            # The core requirement is finding its scale and zero_point.
            
            is_pattern_match = ANONYMOUS_MATMUL_PATTERN.fullmatch(cand_name) is not None
            if is_pattern_match:
                 print(f"        Candidate: {cand_name} (INT8, shape {cand_shape}) matches anonymous pattern.")
            else:
                 print(f"        Candidate: {cand_name} (INT8, shape {cand_shape}) does NOT match pattern '{ANONYMOUS_MATMUL_PATTERN.pattern}', but checking for scale/zp due to shape/type match.")

            expected_scale_name = cand_name.replace("_quantized", "_scale")
            expected_zp_name = cand_name.replace("_quantized", "_zero_point")
            
            # print(f"          Looking for scale: '{expected_scale_name}'") # Verbose
            t_sc = self._tensor_by_name(expected_scale_name)
            # print(f"          Looking for ZP: '{expected_zp_name}'") # Verbose
            t_zp = self._tensor_by_name(expected_zp_name)

            if t_sc and t_zp:
                print(f"        SUCCESS: Found full triplet for {cand_name}!")
                self.used_anonymous_tensors.add(cand_name)
                return t_wq_candidate, t_sc, t_zp
            else:
                # Log why this candidate failed if it was promising
                fail_reason = []
                if not t_sc: fail_reason.append(f"Scale '{expected_scale_name}' not found")
                if not t_zp: fail_reason.append(f"ZP '{expected_zp_name}' not found")
                print(f"        SKIPPING {cand_name}: {'; '.join(fail_reason)}.")
        
        print(f"    Finished searching for anonymous triplet with shape {desired_shape}. No suitable triplet found.")
        return None, None, None
        
    def _print_available_initializers_and_exit(self, failed_component: str, attempted_shape_for_anon_search):
        print(f"\n--- ERROR: Failed to find INT-8 weights for {failed_component} ---")
        if attempted_shape_for_anon_search:
            print(f"--- Last attempted anonymous search with shape: {attempted_shape_for_anon_search}")
        print("\n--- AVAILABLE INITIALIZERS (POTENTIAL WEIGHTS) IN ONNX GRAPH ---")
        if not self.onnx_graph.initializer:
            print("  No initializers found in the ONNX graph! Model might be structured differently or empty.")
        
        # Sort initializers by name for easier reading
        sorted_initializers = sorted(self.onnx_graph.initializer, key=lambda t: t.name)

        for i, t in enumerate(sorted_initializers):
            tensor_type_str = onnx.TensorProto.DataType.Name(t.data_type)
            print(f"  Name: {t.name:<70} Shape: {str(self._get_shape(t)):<20} DataType: {tensor_type_str} ({t.data_type})")
            if i > 200 and len(sorted_initializers) > 220 : # Limit print for very large models
                print(f"  ... (and {len(sorted_initializers) - (i+1)} more initializers)")
                break
        print("--- END AVAILABLE INITIALIZERS ---\n")
        sys.exit(f"❌ Cannot locate INT-8 weights for {failed_component}. Please review the initializer list above to find the correct names/shapes and adjust the script if necessary.")

    def extract_weights(self):
        print(f"🔍 Scanning {self.model_path.name}…")
        with open(self.full_dump_file_path, "w", encoding="utf-8") as full_dump_fh:
            full_dump_fh.write(f"# Full weight dump from {self.model_path.name}\n")
            full_dump_fh.write(f"# Target d_model=256, num_heads=4, head_dim=64, d_ff=512\n")

            d_model_accelerator = 256
            head_dim_accelerator = 64
            num_heads_accelerator = 1
            
            current_onnx_qkv_full_shape = (512, 512) 

            qkv_prefixes_to_try = [
                "layers.0.self_attn.",            
                "encoder.layers.0.self_attn.",    
            ]
            print("\n--- Extracting QKV Weights ---")
            for proj_type in ("query", "key", "value"):
                print(f"  Processing {proj_type} projection...")
                helsinki_proj_suffix = proj_type[0] + "_proj." 
                wq_full, sc_full, zp_full = None, None, None
                found_named = False

                for prefix_base in qkv_prefixes_to_try:
                    full_name_base_attempt = prefix_base + helsinki_proj_suffix
                    wq_full, sc_full, zp_full = self._get_quant_triplet(full_name_base_attempt)
                    if wq_full is not None:
                        print(f"    Found named {proj_type} weights using base: {full_name_base_attempt} (Actual ONNX shape: {self._get_shape(wq_full)})")
                        current_onnx_qkv_full_shape = self._get_shape(wq_full) 
                        found_named = True
                        break
                
                if not found_named:
                    print(f"    Named {proj_type} weights not found. Trying anonymous search with fallback shape {current_onnx_qkv_full_shape}...")
                    wq_full, sc_full, zp_full = self._find_anonymous_quant_triplet(current_onnx_qkv_full_shape)
                    if wq_full is not None:
                        print(f"    Found anonymous {proj_type} weights: {wq_full.name} (Actual ONNX shape: {self._get_shape(wq_full)})")
                        current_onnx_qkv_full_shape = self._get_shape(wq_full)

                if wq_full is None or not self._is_int8(wq_full):
                    self._print_available_initializers_and_exit(f"QKV {proj_type}", current_onnx_qkv_full_shape)

                for head_idx in range(num_heads_accelerator):
                    row_start = head_idx * head_dim_accelerator
                    row_end = (head_idx + 1) * head_dim_accelerator
                    
                    full_onnx_matrix_np = np.frombuffer(wq_full.raw_data, dtype=np.int8).reshape(self._get_shape(wq_full))
                    
                    if not (row_end <= full_onnx_matrix_np.shape[0] and d_model_accelerator <= full_onnx_matrix_np.shape[1]):
                        self._print_available_initializers_and_exit(
                            f"Slicing QKV {proj_type}_head{head_idx} from {wq_full.name} (shape {full_onnx_matrix_np.shape}) " +
                            f"with target slice [{row_start}:{row_end}, :{d_model_accelerator}] is out of bounds.",
                            current_onnx_qkv_full_shape
                        )

                    sliced_head_weights = full_onnx_matrix_np[row_start:row_end, :d_model_accelerator].copy()
                    
                    file_name_stem = f"{proj_type}_head{head_idx}"
                    self._save_tensor_files(sliced_head_weights, file_name_stem, sc_full, zp_full, full_dump_fh)

            print("\n--- Extracting Attention Output Projection (Wo) ---")
            wo_onnx_expected_shape = current_onnx_qkv_full_shape 
            wo_prefixes_to_try = [
                "layers.0.self_attn.out_proj.",
                "encoder.layers.0.self_attn.out_proj.",
                "encoder.layers.0.attention.output.dense." 
            ]
            wq_wo, sc_wo, zp_wo = None, None, None
            found_named_wo = False
            for prefix_base in wo_prefixes_to_try:
                wq_wo, sc_wo, zp_wo = self._get_quant_triplet(prefix_base)
                if wq_wo is not None:
                    print(f"  Found named Wo weights using base: {prefix_base} (Actual ONNX shape: {self._get_shape(wq_wo)})")
                    wo_onnx_expected_shape = self._get_shape(wq_wo)
                    found_named_wo = True
                    break
            if not found_named_wo:
                print(f"  Named Wo weights not found. Trying anonymous search with shape {wo_onnx_expected_shape}...")
                wq_wo, sc_wo, zp_wo = self._find_anonymous_quant_triplet(wo_onnx_expected_shape)
                if wq_wo is not None:
                     print(f"  Found anonymous Wo weights: {wq_wo.name} (Actual ONNX shape: {self._get_shape(wq_wo)})")
                     wo_onnx_expected_shape = self._get_shape(wq_wo)
            
            if wq_wo is None or not self._is_int8(wq_wo):
                 self._print_available_initializers_and_exit("Attention Output Projection (Wo)", wo_onnx_expected_shape)
            
            sliced_wo = self._slice_2d(wq_wo, d_model_accelerator, d_model_accelerator)
            self._save_tensor_files(sliced_wo, "attn_out_proj", sc_wo, zp_wo, full_dump_fh)

            print("\n--- Extracting FFN Weights ---")
            ffn_specs = [
                ("FC1", ["layers.0.fc1.", "encoder.layers.0.intermediate.dense."], (2048, 512), 512, d_model_accelerator, "ffn_fc1"),
                ("FC2", ["layers.0.fc2.", "encoder.layers.0.output.dense."], (512, 2048), d_model_accelerator, 512, "ffn_fc2")
            ]

            for layer_name, prefixes, default_onnx_shape, r_slice, c_slice, file_stem in ffn_specs:
                print(f"  Processing FFN {layer_name}...")
                wq_fc, sc_fc, zp_fc = None, None, None
                current_ffn_onnx_shape = default_onnx_shape
                found_named_ffn = False
                for prefix_base in prefixes:
                    wq_fc, sc_fc, zp_fc = self._get_quant_triplet(prefix_base)
                    if wq_fc is not None:
                        print(f"    Found named {layer_name} weights using base: {prefix_base} (Actual ONNX shape: {self._get_shape(wq_fc)})")
                        current_ffn_onnx_shape = self._get_shape(wq_fc)
                        found_named_ffn = True
                        break
                if not found_named_ffn:
                    print(f"    Named {layer_name} weights not found. Trying anonymous search with shape {current_ffn_onnx_shape}...")
                    wq_fc, sc_fc, zp_fc = self._find_anonymous_quant_triplet(current_ffn_onnx_shape)
                    if wq_fc is not None:
                        print(f"    Found anonymous {layer_name} weights: {wq_fc.name} (Actual ONNX shape: {self._get_shape(wq_fc)})")
                        current_ffn_onnx_shape = self._get_shape(wq_fc)

                if wq_fc is None or not self._is_int8(wq_fc):
                    self._print_available_initializers_and_exit(f"FFN {layer_name}", current_ffn_onnx_shape)
                
                actual_onnx_shape = self._get_shape(wq_fc)
                if not (r_slice <= actual_onnx_shape[0] and c_slice <= actual_onnx_shape[1]):
                    self._print_available_initializers_and_exit(
                        f"Slicing FFN {layer_name} from {wq_fc.name} (shape {actual_onnx_shape}) " +
                        f"with target slice ({r_slice},{c_slice}) is out of bounds.",
                        current_ffn_onnx_shape
                    )

                sliced_fc = self._slice_2d(wq_fc, r_slice, c_slice)
                self._save_tensor_files(sliced_fc, file_stem, sc_fc, zp_fc, full_dump_fh)

        print(f"\n✏️ Done. Weight details dumped to:\n  {self.full_dump_file_path}\n  {self.out_dir / 'weights_index.txt'}")
        print(f"Binary .bin, .scale, .zp files are in: {self.out_dir}")


def main():
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