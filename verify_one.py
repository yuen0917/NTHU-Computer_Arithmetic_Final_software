import numpy as np

# ==========================================
# 1. Parameter Settings
# ==========================================
IMG_W = 28
IMG_H = 28
QUANT_SHIFT = 6  # Hardware Setting

# File names
FILE_IMG = "test_image.txt"
FILE_W_L1 = "weights/conv1_relu.txt"
FILE_W_L2 = "weights/conv2_selu.txt"
FILE_W_L3 = "weights/conv3_gelu.txt"
FILE_W_FC = "weights/fc_weights.txt"
FILE_B_FC = "weights/fc_biases.txt"
FILE_SELU_LUT = "selu_lut.txt"
FILE_GELU_LUT = "gelu_lut.txt"

# ==========================================
# 2. Hardware Functions (Same as before)
# ==========================================
def to_signed8(val):
    val = val & 0xFF
    if val > 127: return val - 256
    return val

def to_signed32(val):
    val = val & 0xFFFFFFFF
    if val > 0x7FFFFFFF: return val - 0x100000000
    return val

def load_hex_file(filename, is_bias=False):
    data = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    val = int(line, 16)
                    if "image" in filename: pass
                    elif is_bias: val = to_signed32(val)
                    else: val = to_signed8(val)
                    data.append(val)
        return data
    except FileNotFoundError:
        print(f"[Error] File {filename} not found!")
        exit()

def load_lut(filename):
    lut = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                val = to_signed8(int(line.strip(), 16))
                lut.append(val)
        return lut
    except: return []

def int8_to_lut_index(x_int):
    x_clipped = max(-128, min(127, x_int))
    return x_clipped if x_clipped >= 0 else x_clipped + 256

def selu_func(x_int, selu_lut):
    return selu_lut[int8_to_lut_index(x_int)]

def gelu_func(x_int, gelu_lut):
    return gelu_lut[int8_to_lut_index(x_int)]

def conv2d_fixed(input_vol, weights, ch_out, ch_in, img_size, padding=1, act_type='relu', selu_lut=None, gelu_lut=None):
    H, W = img_size, img_size
    output_vol = np.zeros((ch_out, H, W), dtype=int)
    padded_input = np.zeros((ch_in, H + 2*padding, W + 2*padding), dtype=int)
    padded_input[:, padding:H+padding, padding:W+padding] = input_vol

    for oc in range(ch_out):
        for r in range(H):
            for c in range(W):
                sum_val = 0
                for ic in range(ch_in):
                    sub_img = padded_input[ic, r:r+3, c:c+3]
                    w_kernel = weights[oc][ic]
                    sum_val += np.sum(sub_img * w_kernel)

                sum_val = sum_val >> QUANT_SHIFT
                if sum_val > 127: sum_val = 127
                elif sum_val < -128: sum_val = -128

                if act_type == 'relu': val = max(0, sum_val)
                elif act_type == 'selu': val = selu_func(sum_val, selu_lut)
                elif act_type == 'gelu': val = gelu_func(sum_val, gelu_lut)
                else: val = sum_val

                output_vol[oc, r, c] = val
    return output_vol

def max_pool_2x2(input_vol):
    ch, h, w = input_vol.shape
    output_vol = np.zeros((ch, h // 2, w // 2), dtype=int)
    for c in range(ch):
        for r in range(h // 2):
            for col in range(w // 2):
                output_vol[c, r, col] = np.max(input_vol[c, r*2:r*2+2, col*2:col*2+2])
    return output_vol

def global_avg_pool_approx(input_vol):
    ch, h, w = input_vol.shape
    output_vec = np.zeros(ch, dtype=int)
    for c in range(ch):
        current_sum = np.sum(input_vol[c])
        val = (current_sum * 167) >> 15
        output_vec[c] = val
    return output_vec

def fully_connected(gap_out, weights, biases):
    W = np.array(weights).reshape(10, 32)
    B = np.array(biases)
    outputs = []
    for i in range(10):
        acc = np.dot(W[i], gap_out) + B[i]
        outputs.append(acc)
    return np.array(outputs)

# ==========================================
# 3. Smart Debug Functions
# ==========================================
def inspect_layer(name, data):
    print(f"\n--- Inspecting {name} ---")
    ch, h, w = data.shape

    # 1. Print first non-zero value
    non_zero_indices = np.transpose(np.nonzero(data[0])) # Check Channel 0
    if len(non_zero_indices) > 0:
        r, c = non_zero_indices[0]
        val = data[0, r, c]
        print(f"  [First Non-Zero] Ch0 @ (Row {r}, Col {c}) = {val}")
    else:
        print(f"  [Warning] Channel 0 is all Zeros!")

    # 2. Print Center Row (Middle of the image)
    mid_r = h // 2
    print(f"  [Center Row {mid_r}] Ch0 Data (First 10 cols):")
    print(f"  {data[0, mid_r, 0:10]}")

# ==========================================
# 4. Main
# ==========================================
def main():
    print("=== Python Hardware Verification Model (Smart Debug) ===")

    selu_lut = load_lut(FILE_SELU_LUT)
    gelu_lut = load_lut(FILE_GELU_LUT)

    raw_img = load_hex_file(FILE_IMG)
    input_img = np.array(raw_img).reshape(1, IMG_H, IMG_W)

    # Check Input
    print(f"\n[Input Image] Row 14 (Center): {input_img[0, 14, 10:20]}")

    # L1
    w_l1 = np.array(load_hex_file(FILE_W_L1)).reshape(8, 1, 3, 3)
    l1_out = conv2d_fixed(input_img, w_l1, 8, 1, 28, act_type='relu')
    inspect_layer("Layer 1 (Conv+ReLU)", l1_out)

    # L2
    w_l2 = np.array(load_hex_file(FILE_W_L2)).reshape(16, 8, 3, 3)
    l2_out = conv2d_fixed(l1_out, w_l2, 16, 8, 28, act_type='selu', selu_lut=selu_lut)
    inspect_layer("Layer 2 (Conv+SELU)", l2_out)

    # MP
    mp_out = max_pool_2x2(l2_out)

    # L3
    w_l3 = np.array(load_hex_file(FILE_W_L3)).reshape(32, 16, 3, 3)
    l3_out = conv2d_fixed(mp_out, w_l3, 32, 16, 14, act_type='gelu', gelu_lut=gelu_lut)
    inspect_layer("Layer 3 (Conv+GELU)", l3_out)

    # GAP
    gap_out = global_avg_pool_approx(l3_out)
    print("\n--- GAP Output (All 32 Ch) ---")
    print(gap_out)

    # FC
    w_fc = load_hex_file(FILE_W_FC)
    b_fc = load_hex_file(FILE_B_FC, is_bias=True)
    fc_out = fully_connected(gap_out, w_fc, b_fc)

    print("\n--- Final FC Scores ---")
    for i, score in enumerate(fc_out):
        print(f"Class {i}: {score}")

    print(f"\n[Prediction] Class: {np.argmax(fc_out)}")

if __name__ == "__main__":
    main()