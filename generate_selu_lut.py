# Python script to generate selu_lut.txt

import numpy as np

def selu(x):
    alpha = 1.67326324
    scale = 1.05070098
    return scale * x if x > 0 else scale * alpha * (np.exp(x) - 1)

# Input range: int8 (-128 to 127)
# Output range: int8 (-128 to 127)

with open("selu_lut.txt", "w") as f:
    for i in range(256):
        # Convert unsigned index 0..255 back to signed int8 -128..127
        val_in = i if i < 128 else i - 256

        # Apply SELU function (input is usually scaled, here assumes raw mapping)
        # Note: Depending on your quantization scheme, input '1' might mean 1.0 or 0.01
        # Assuming direct mapping for simplicity or pre-computed fixed point
        val_out = selu(val_in)

        # Clip and round to int8
        val_out_int = int(np.clip(np.round(val_out), -128, 127))

        # Convert back to 2's complement hex
        hex_val = val_out_int & 0xFF
        f.write(f"{hex_val:02x}\n")

print("selu_lut.txt generated.")

