import numpy as np

def gelu_approx(x):
    # using PDF provided approximation formula
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    # here x is considered as direct input integer.
    # if your design has decimal point (e.g. Q3.5), x needs to be divided by 2^5 here.
    # but according to your Conv2d code, you do Quantization after Activation,
    # so it is usually considered as integer range -128 ~ 127.

    sqrt_2_over_pi = np.sqrt(2 / np.pi)
    term = sqrt_2_over_pi * (x + 0.044715 * np.power(x, 3))
    return 0.5 * x * (1 + np.tanh(term))

def generate_gelu_hex():
    filename = "gelu_lut.txt"
    with open(filename, 'w') as f:
        # Verilog's readmemh reading order:
        # for 8-bit signed number:
        # index 0    corresponds to Hex 00 (decimal 0)
        # ...
        # index 127 corresponds to Hex 7F (decimal 127)
        # index 128 corresponds to Hex 80 (decimal -128)
        # ...
        # index 255 corresponds to Hex FF (decimal -1)

        # therefore we loop from 0 to 255
        for i in range(256):
            # convert 0~255 to signed integer -128~127
            if i < 128:
                input_val = i
            else:
                input_val = i - 256

            # calculate GELU
            res_float = gelu_approx(input_val)

            # round and truncate to -128 ~ 127
            res_int = int(round(res_float))
            if res_int > 127:
                res_int = 127
            elif res_int < -128:
                res_int = -128

            # convert back to 8-bit Hex (handle negative numbers)
            hex_val = res_int & 0xFF

            f.write(f"{hex_val:02x}\n")

    print(f"Generated {filename}")

if __name__ == "__main__":
    generate_gelu_hex()