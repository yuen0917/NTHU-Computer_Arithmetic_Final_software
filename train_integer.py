import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --------- PyTorch Model ---------
class GlobalAvgPoolCNN_PyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Conv2d(1->8, kernel=3, padding=1)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        # 2. Conv2d(8->16, kernel=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # 3. MaxPool 2x2
        self.pool = nn.MaxPool2d(2, 2)
        # 4. Conv2d(16->32, kernel=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 5. Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 6. FC(32->10)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        # 注意：这里的输入 x 已经是 0.0 ~ 255.0 的数值范围
        x = torch.relu(self.conv1(x))
        x = torch.selu(self.conv2(x))
        x = self.pool(x)
        x = torch.nn.functional.gelu(self.conv3(x))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def compute_quantization_params(tensor, dtype=np.int8):
    """Simple Min-Max Quantization to find scale and zero-point (Symmetric for weights usually)"""
    # For simplicity in this project, we assume symmetric quantization for weights around 0
    # This matches the Verilog behavior of simple fixed-point multiplication
    min_val = tensor.min()
    max_val = tensor.max()
    abs_max = max(abs(min_val), abs(max_val))

    # Target range for int8 is -128 to 127
    scale = abs_max / 127.0 if abs_max != 0 else 1.0
    zero_point = 0
    return scale, zero_point

def quantize_tensor(tensor, scale, zero_point, dtype=np.int8):
    # x_q = clamp(round(x / scale) + zero_point)
    q_tensor = np.round(tensor / scale) + zero_point
    q_tensor = np.clip(q_tensor, -128, 127)
    return q_tensor.astype(dtype)

def save_weights_to_txt(data, filename, is_bias=False):
    """Save weights to hex txt format for Verilog $readmemh"""
    # Create directory if not exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        # Flatten data
        flat_data = data.flatten()
        for val in flat_data:
            # Convert to Python int to avoid overflow issues with numpy int8/int32
            val = int(val)
            # Handle negative numbers for hex representation (two's complement)
            if is_bias:
                # 32-bit for Bias
                if val < 0: val = val + (1 << 32)
                f.write(f"{val:08x}\n")
            else:
                # 8-bit for Weights
                if val < 0: val = val + (1 << 8)
                f.write(f"{val:02x}\n")
    print(f"Saved {filename}")

def train_model(epochs=5, batch_size=64, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # [關鍵修改] Data Transforms: 0-1 -> 0-255
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts to [0.0, 1.0]
        # Scale back to [0.0, 255.0] to simulate integer input
        transforms.Lambda(lambda x: x * 255.0)
    ])

    print("Downloading MNIST dataset...")
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = GlobalAvgPoolCNN_PyTorch().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\nStarting training (Integer-Aware Mode)...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)

        print(f"Epoch {epoch+1} Test Acc: {100.*test_correct/test_total:.2f}%")

    # --- Export Weights for Verilog ---
    print("\nExporting weights to 'weights/' directory...")
    os.makedirs("weights", exist_ok=True)

    # Get weights as numpy arrays
    W1 = model.conv1.weight.data.cpu().numpy() # (8, 1, 3, 3)
    # PyTorch Bias is usually float, but hardware expects integer accumulation
    # However, since we simulate training with float, we need to decide how to quantize bias.
    # Often bias is quantized to 32-bit.
    b1 = model.conv1.bias.data.cpu().numpy()

    W2 = model.conv2.weight.data.cpu().numpy()
    b2 = model.conv2.bias.data.cpu().numpy()
    W3 = model.conv3.weight.data.cpu().numpy()
    b3 = model.conv3.bias.data.cpu().numpy()
    W4 = model.fc.weight.data.cpu().numpy()
    b4 = model.fc.bias.data.cpu().numpy()

    # --- Quantization Simulation (Float -> Int8) ---
    # We simply scale and clip to get rough int8 weights for Verilog
    # Note: A proper quantization aware training (QAT) is better, but this post-training quantization
    # usually works fine for MNIST.

    # We assume weights are small floats. We need to scale them up to fit -128~127.
    # Strategy: Find max absolute value per layer, map it to 127.

    def quantize_layer(w, b, name):
        w_scale, _ = compute_quantization_params(w)
        # Use the SAME scale for bias? Or bias is usually accumulated result.
        # For simplicity in this specific project flow, let's quantize bias to int32
        # assuming it matches the accumulation scale.
        # But realistically, bias scale = input_scale * weight_scale.
        # Since input is 0-255 (scale 1.0), bias scale ~= weight scale.

        # Quantize Weights to Int8
        w_q = quantize_tensor(w, w_scale, 0)

        # Quantize Bias to Int32 (using weight scale for simplicity, often 2^10 factor involved in hardware)
        # In your hardware, bias is added to the MAC sum.
        # MAC sum = (pixel * weight). Pixel is large (0-255). Weight is int8.
        # So bias should be roughly same scale as (255 * 127).
        # Let's trust the float training magnitude and scale it similarly.
        # But actually, bias in Verilog is added directly.
        # Let's scale bias by (1/w_scale) * (some factor?)
        # Let's try simple scaling: scale bias to int32 using same scale factor as weights
        # BUT shifted by input magnitude?
        # Actually, the simplest way that works for this specific ad-hoc hardware:
        # Map float weights to int8 range. Map float bias to int32 range proportionally.

        b_q = np.round(b / w_scale).astype(np.int32)

        print(f"[{name}] Scale: {1/w_scale:.4f}, W_int range: [{w_q.min()}, {w_q.max()}]")
        return w_q, b_q

    W1_q, b1_q = quantize_layer(W1, b1, "Layer1")
    W2_q, b2_q = quantize_layer(W2, b2, "Layer2")
    W3_q, b3_q = quantize_layer(W3, b3, "Layer3")
    W4_q, b4_q = quantize_layer(W4, b4, "FC")

    # --- Save to .txt for Verilog ---
    save_weights_to_txt(W1_q, "weights/conv1_relu.txt")
    # Biases in Verilog often not used in Conv layers if not implemented,
    # but your code seems to only implement bias for FC?
    # Checking your code: conv2d_layerX modules usually DON'T have bias input ports in your RTL.
    # ONLY FC unit has bias input.

    save_weights_to_txt(W2_q, "weights/conv2_selu.txt")
    save_weights_to_txt(W3_q, "weights/conv3_gelu.txt")

    # FC Layer has Bias
    save_weights_to_txt(W4_q, "weights/fc_weights.txt")
    save_weights_to_txt(b4_q, "weights/fc_biases.txt", is_bias=True)

    print("\nTraining and Export Complete!")
    print("New weights are in 'weights/' folder.")

if __name__ == "__main__":
    train_model(epochs=30, batch_size=64, lr=0.001)