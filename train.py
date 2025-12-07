import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --------- PyTorch Model (matching the NumPy architecture) ---------


class GlobalAvgPoolCNN_PyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Conv2d(1→8, kernel=3, padding=1)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)

        # 2. Conv2d(8→16, kernel=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        # 3. MaxPool 2×2
        self.pool = nn.MaxPool2d(2, 2)

        # 4. Conv2d(16→32, kernel=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # 5. Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 6. FC(32→10)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        # 1. Conv(1→8) + ReLU -> (N, 8, 28, 28)
        x = torch.relu(self.conv1(x))

        # 2. Conv(8→16) + SELU -> (N, 16, 28, 28)
        x = torch.selu(self.conv2(x))

        # 3. MaxPool 2×2 -> (N, 16, 14, 14)
        x = self.pool(x)

        # 4. Conv(16→32) + GELU -> (N, 32, 14, 14)
        x = torch.nn.functional.gelu(self.conv3(x))

        # 5. Global Average Pooling -> (N, 32, 1, 1)
        x = self.global_avg_pool(x)

        # 6. Flatten -> (N, 32)
        x = x.view(x.size(0), -1)

        # 7. FC(32→10) -> (N, 10)
        x = self.fc(x)
        return x


def train_model(epochs=5, batch_size=64, lr=0.001):
    """Train the model and save weights as NumPy arrays"""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # MNIST is already normalized to [0, 1], but we can add normalization
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        ]
    )

    # Download and load MNIST dataset
    print("Downloading MNIST dataset...")
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create model
    model = GlobalAvgPoolCNN_PyTorch().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print("\nStarting training...")
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
                print(
                    f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%"
                )

        # Evaluate on test set
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

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {running_loss/len(train_loader):.4f}, "
            f"Train Acc: {100.*correct/total:.2f}%, Test Acc: {100.*test_correct/test_total:.2f}%"
        )

    # Save weights as NumPy arrays
    print("\nSaving weights as NumPy arrays...")
    os.makedirs("weights", exist_ok=True)

    # Convert PyTorch weights to NumPy format (matching the NumPy model structure)
    # Note: PyTorch conv weights are (C_out, C_in, H, W), which matches our NumPy format
    # PyTorch linear weights are (out_features, in_features), which matches our NumPy format

    W1 = model.conv1.weight.data.cpu().numpy()  # (8, 1, 3, 3)
    b1 = model.conv1.bias.data.cpu().numpy()  # (8,)

    W2 = model.conv2.weight.data.cpu().numpy()  # (16, 8, 3, 3)
    b2 = model.conv2.bias.data.cpu().numpy()  # (16,)

    W3 = model.conv3.weight.data.cpu().numpy()  # (32, 16, 3, 3)
    b3 = model.conv3.bias.data.cpu().numpy()  # (32,)

    W4 = model.fc.weight.data.cpu().numpy()  # (10, 32)
    b4 = model.fc.bias.data.cpu().numpy()  # (10,)

    # Save float32 weights
    np.savez(
        "weights/mnist_weights.npz",
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        W3=W3,
        b3=b3,
        W4=W4,
        b4=b4,
    )

    print("Float32 weights saved to weights/mnist_weights.npz")
    print("\nWeight shapes:")
    print(f"W1: {W1.shape}, b1: {b1.shape}")
    print(f"W2: {W2.shape}, b2: {b2.shape}")
    print(f"W3: {W3.shape}, b3: {b3.shape}")
    print(f"W4: {W4.shape}, b4: {b4.shape}")

    # Quantize and save quantized weights
    print("\nQuantizing weights...")
    from quantized_model_improved import compute_quantization_params, quantize_tensor

    dtype = np.int8

    # Quantize all weights and biases
    W1_scale, W1_zp = compute_quantization_params(W1, dtype)
    b1_scale, b1_zp = compute_quantization_params(b1, dtype)
    W1_q = quantize_tensor(W1, W1_scale, W1_zp, dtype)
    b1_q = quantize_tensor(b1, b1_scale, b1_zp, dtype)

    W2_scale, W2_zp = compute_quantization_params(W2, dtype)
    b2_scale, b2_zp = compute_quantization_params(b2, dtype)
    W2_q = quantize_tensor(W2, W2_scale, W2_zp, dtype)
    b2_q = quantize_tensor(b2, b2_scale, b2_zp, dtype)

    W3_scale, W3_zp = compute_quantization_params(W3, dtype)
    b3_scale, b3_zp = compute_quantization_params(b3, dtype)
    W3_q = quantize_tensor(W3, W3_scale, W3_zp, dtype)
    b3_q = quantize_tensor(b3, b3_scale, b3_zp, dtype)

    W4_scale, W4_zp = compute_quantization_params(W4, dtype)
    b4_scale, b4_zp = compute_quantization_params(b4, dtype)
    W4_q = quantize_tensor(W4, W4_scale, W4_zp, dtype)
    b4_q = quantize_tensor(b4, b4_scale, b4_zp, dtype)

    # Save quantized weights with scales and zero points
    np.savez(
        "weights/mnist_weights_quantized.npz",
        # Quantized weights (int8)
        W1_q=W1_q,
        b1_q=b1_q,
        W2_q=W2_q,
        b2_q=b2_q,
        W3_q=W3_q,
        b3_q=b3_q,
        W4_q=W4_q,
        b4_q=b4_q,
        # Quantization parameters
        W1_scale=W1_scale,
        W1_zp=W1_zp,
        b1_scale=b1_scale,
        b1_zp=b1_zp,
        W2_scale=W2_scale,
        W2_zp=W2_zp,
        b2_scale=b2_scale,
        b2_zp=b2_zp,
        W3_scale=W3_scale,
        W3_zp=W3_zp,
        b3_scale=b3_scale,
        b3_zp=b3_zp,
        W4_scale=W4_scale,
        W4_zp=W4_zp,
        b4_scale=b4_scale,
        b4_zp=b4_zp,
    )

    print("Quantized weights saved to weights/mnist_weights_quantized.npz")

    # Calculate size comparison
    float_size = (
        W1.size * 4
        + b1.size * 4
        + W2.size * 4
        + b2.size * 4
        + W3.size * 4
        + b3.size * 4
        + W4.size * 4
        + b4.size * 4
    )
    quantized_size = (
        W1_q.size * 1
        + b1_q.size * 1
        + W2_q.size * 1
        + b2_q.size * 1
        + W3_q.size * 1
        + b3_q.size * 1
        + W4_q.size * 1
        + b4_q.size * 1
    )

    print(f"\nModel size comparison:")
    print(f"  Float32:   {float_size / 1024:.2f} KB")
    print(f"  Quantized: {quantized_size / 1024:.2f} KB")
    print(f"  Compression: {float_size / quantized_size:.2f}x")

    return model


if __name__ == "__main__":
    train_model(epochs=10, batch_size=64, lr=0.001)
