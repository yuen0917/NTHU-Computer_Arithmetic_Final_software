import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import sys

# ==========================================
# 1. Parameter Settings
# ==========================================
IMG_W = 28
IMG_H = 28
QUANT_SHIFT = 7       # Hardware Setting
BATCH_SIZE  = 100    # Number of images to test (adjust as needed)
SAVE_IMAGE_INDEX = 10

# File paths
FILE_W_L1 = "weights/conv1_relu.txt"
FILE_W_L2 = "weights/conv2_selu.txt"
FILE_W_L3 = "weights/conv3_gelu.txt"
FILE_W_FC = "weights/fc_weights.txt"
FILE_B_FC = "weights/fc_biases.txt"
FILE_SELU_LUT = "selu_lut.txt"
FILE_GELU_LUT = "gelu_lut.txt"

# ==========================================
# 2. Hardware Simulation Functions (Reused)
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
                    if is_bias: val = to_signed32(val)
                    else: val = to_signed8(val)
                    data.append(val)
        return data
    except FileNotFoundError:
        print(f"[Error] File {filename} not found!")
        sys.exit(1)

def load_lut(filename):
    lut = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    val = int(line, 16)
                    val = to_signed8(val)
                    lut.append(val)
        return lut
    except FileNotFoundError:
        print(f"[Error] LUT {filename} not found!")
        sys.exit(1)

def int8_to_lut_index(x_int):
    x_clipped = max(-128, min(127, x_int))
    if x_clipped >= 0: return x_clipped
    else: return x_clipped + 256

def selu_func(x_int, selu_lut):
    idx = int8_to_lut_index(x_int)
    return selu_lut[idx]

def gelu_func(x_int, gelu_lut):
    idx = int8_to_lut_index(x_int)
    return gelu_lut[idx]

def conv2d_fixed(input_vol, weights, ch_out, ch_in, img_size, padding=1, act_type='relu', selu_lut=None, gelu_lut=None):
    H, W = img_size, img_size
    output_vol = np.zeros((ch_out, H, W), dtype=int)
    padded_input = np.zeros((ch_in, H + 2*padding, W + 2*padding), dtype=int)
    padded_input[:, padding:H+padding, padding:W+padding] = input_vol

    # Optimization: Pre-fetch weights to avoid repeated indexing
    # This is purely for simulation speed, logic remains bit-true
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
# 3. Batch Testing Logic
# ==========================================

def run_hardware_model(img_int, w_l1, w_l2, w_l3, w_fc, b_fc, selu_lut, gelu_lut):
    """Run the full hardware pipeline on a single image"""
    # L1
    l1_out = conv2d_fixed(img_int, w_l1, 8, 1, 28, act_type='relu')
    # L2
    l2_out = conv2d_fixed(l1_out, w_l2, 16, 8, 28, act_type='selu', selu_lut=selu_lut)
    # MP
    mp_out = max_pool_2x2(l2_out)
    # L3
    l3_out = conv2d_fixed(mp_out, w_l3, 32, 16, 14, act_type='gelu', gelu_lut=gelu_lut)
    # GAP
    gap_out = global_avg_pool_approx(l3_out)
    # FC
    fc_out = fully_connected(gap_out, w_fc, b_fc)

    return np.argmax(fc_out), fc_out

def main():
    print(f"=== MNIST Hardware Batch Test (N={BATCH_SIZE}, Shift={QUANT_SHIFT}) ===")

    # 1. Load Weights & LUTs
    print("Loading weights and LUTs...")
    selu_lut = load_lut(FILE_SELU_LUT)
    gelu_lut = load_lut(FILE_GELU_LUT)

    w_l1 = np.array(load_hex_file(FILE_W_L1)).reshape(8, 1, 3, 3)
    w_l2 = np.array(load_hex_file(FILE_W_L2)).reshape(16, 8, 3, 3)
    w_l3 = np.array(load_hex_file(FILE_W_L3)).reshape(32, 16, 3, 3)
    w_fc = load_hex_file(FILE_W_FC)
    b_fc = load_hex_file(FILE_B_FC, is_bias=True)

    # 2. Prepare Dataset
    print("Downloading/Loading MNIST...")
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    correct = 0
    total = 0

    print(f"\nStart testing...")

    for i in range(BATCH_SIZE):
        image_tensor, label = testset[i]

        # --- Preprocessing: Convert to 0-255 Integer ---
        # Assuming your hardware expects raw pixel values 0-255
        img_np = image_tensor.numpy() * 255.0
        img_int = img_np.astype(int).reshape(1, IMG_H, IMG_W) # (1, 28, 28)

        # Run Model
        pred, scores = run_hardware_model(img_int, w_l1, w_l2, w_l3, w_fc, b_fc, selu_lut, gelu_lut)

        if pred == label:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"

        total += 1
        print(f"Img {i}: True {label} -> Pred {pred} | {status} | Acc: {correct/total:.2%}", end='\r')

    print(f"\n\n=== Final Result ===")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct/total:.2%}")


    img_tensor, label = testset[SAVE_IMAGE_INDEX]
    img_int = (img_tensor.numpy() * 255).astype(int)
    fname = f"test_image_label_{label}.txt"
    np.savetxt(fname, img_int.flatten(), fmt="%02x")
    print(f"Saved Img {SAVE_IMAGE_INDEX} (Label {label}) to {fname} for Verilog.")

if __name__ == "__main__":
    main()