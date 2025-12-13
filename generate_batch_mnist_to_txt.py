import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os

BATCH_SIZE = 100  # 設定要測試的張數

def export_batch_data():
    print(f"=== Exporting {BATCH_SIZE} images for Verilog Batch Test ===")

    # Load MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Files
    f_img = open("all_test_images.txt", "w")
    f_lbl = open("all_labels.txt", "w")

    for i in range(BATCH_SIZE):
        image, label = testset[i]

        # Convert to 0-255 int
        img_int = (image.numpy() * 255).astype(int).flatten()

        # Write Image Data (Hex)
        for val in img_int:
            f_img.write(f"{val:02x}\n")

        # Write Label (Decimal is fine for Verilog readmemh if formatted correctly, but hex is safer)
        f_lbl.write(f"{label:x}\n")

        print(f"Exported Image {i}/{BATCH_SIZE} (Label: {label})", end='\r')

    f_img.close()
    f_lbl.close()
    print("\n[Done] Generated 'all_test_images.txt' and 'all_labels.txt'")

if __name__ == "__main__":
    export_batch_data()