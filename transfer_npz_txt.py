import numpy as np
from pathlib import Path

# === config (modify here for your verilog format) ===
NPZ_PATH = "weights/mnist_weights_quantized.npz"
BIT_WIDTH = 8  # 8 for int8, or 32 for bias if needed

# For int8, use two hex digits (00~FF)
HEX_DIGITS = (BIT_WIDTH + 3) // 4  # 8 bit -> 2 hex chars


def int_to_hex_signed(x: int, bits: int) -> str:
    """Convert signed integer to two's complement hex string."""
    mask = (1 << bits) - 1
    # Adjust format width based on bits (e.g., 32-bit bias needs 8 hex chars)
    width = (bits + 3) // 4
    return format(x & mask, f"0{width}X")


def find_int_weights_by_shape(
    data: np.lib.npyio.NpzFile,
    target_shape,
    prefer_key: str | None = None,
) -> tuple[str, np.ndarray]:
    """Find integer weight tensor with given shape, optionally preferring a specific key."""
    if prefer_key is not None and prefer_key in data.files:
        arr = data[prefer_key]
        # Check shape match
        if arr.shape == target_shape:
            print(f"[Info] Found key '{prefer_key}' matching shape {target_shape}")
            return prefer_key, np.asarray(arr)

    # Fallback: search by shape
    for key in data.files:
        arr = data[key]
        if arr.shape == target_shape:
            print(f"[Info] Found key '{key}' by shape {target_shape}, dtype={arr.dtype}")
            return key, np.asarray(arr)

    raise RuntimeError(f"Cannot find integer weights with shape {target_shape} in {data.files}.")


def save_flat_hex(flat: np.ndarray, out_path: Path, bits=8):
    """Save 1D integer array as hex file, one value per line."""
    # Convert numpy types to python int for formatting
    lines = [int_to_hex_signed(int(v), bits) for v in flat.ravel(order="C")]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Save] Saved {len(flat)} values to {out_path} ({bits}-bit)")


def main():
    if not Path(NPZ_PATH).exists():
        print(f"Error: {NPZ_PATH} not found.")
        return

    data = np.load(NPZ_PATH)
    print("Keys in npz:", list(data.keys()))
    print("-" * 40)

    # =========================================================================
    # 1. Layer 1: Conv2d (1 -> 8, 3x3)
    # Target File: conv1_relu.txt
    # =========================================================================
    # Shape: (Out, In, K, K) = (8, 1, 3, 3)
    _, w1 = find_int_weights_by_shape(data, (8, 1, 3, 3), prefer_key="conv1.weight")
    save_flat_hex(w1, Path("weights/conv1_relu.txt"), bits=8)

    # =========================================================================
    # 2. Layer 2: Conv2d (8 -> 16, 3x3)
    # Target File: conv2_selu.txt
    # =========================================================================
    # Shape: (Out, In, K, K) = (16, 8, 3, 3)
    _, w2 = find_int_weights_by_shape(data, (16, 8, 3, 3), prefer_key="conv2.weight")
    save_flat_hex(w2, Path("weights/conv2_selu.txt"), bits=8)

    # =========================================================================
    # 3. Layer 3: Conv2d (16 -> 32, 3x3) (Your "Layer 4" in PDF)
    # Target File: conv3_gelu.txt
    # =========================================================================
    # Shape: (Out, In, K, K) = (32, 16, 3, 3)
    _, w3 = find_int_weights_by_shape(data, (32, 16, 3, 3), prefer_key="conv3.weight")
    save_flat_hex(w3, Path("weights/conv3_gelu.txt"), bits=8)

    # =========================================================================
    # 4. FC Layer Weights (32 -> 10)
    # Target File: fc_weights.txt
    # =========================================================================
    # Shape: (Out_features, In_features) = (10, 32)
    # Note: PyTorch Linear layer stores weights as (Out, In)
    # Your Verilog reads it linearly: weights[i*32 + j] where i is class(0..9)
    # This matches numpy's C-order flatten of (10, 32).
    _, w_fc = find_int_weights_by_shape(data, (10, 32), prefer_key="fc.weight")
    save_flat_hex(w_fc, Path("weights/fc_weights.txt"), bits=8)

    # =========================================================================
    # 5. FC Layer Biases (10)
    # Target File: fc_biases.txt
    # =========================================================================
    # Shape: (10,)
    # Note: Biases are often 32-bit integers in quantized models to handle accumulation
    # Check your Verilog: reg signed [31:0] biases [0:9]; -> Needs 32-bit hex
    _, b_fc = find_int_weights_by_shape(data, (10,), prefer_key="fc.bias")
    save_flat_hex(b_fc, Path("weights/fc_biases.txt"), bits=32)

    print("-" * 40)
    print("All weight files generated successfully.")


if __name__ == "__main__":
    main()