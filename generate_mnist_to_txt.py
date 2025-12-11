"""
Convert a single MNIST image to text file for testbench use.

This script loads a MNIST image and converts it to a text file format
suitable for Verilog/SystemVerilog testbench.
"""

import numpy as np
import torch
from torchvision import datasets, transforms
import argparse
import os


def convert_mnist_to_txt(
    image_idx=0,
    dataset_type="test",
    output_file="mnist_image.txt",
    format="hex",
    normalize=True,
    flatten=True,
):
    """
    Convert a single MNIST image to text file.

    Args:
        image_idx: Index of the image in the dataset (default: 0)
        dataset_type: "train" or "test" (default: "test")
        output_file: Output text file path (default: "mnist_image.txt")
        format: Output format - "hex", "dec", or "bin" (default: "hex")
        normalize: Whether to use normalized values (default: True)
                   If True: uses normalized values (0-1 range, then scaled)
                   If False: uses raw pixel values (0-255)
        flatten: Whether to flatten the image (default: True)
                 If True: outputs 784 values (28x28 flattened)
                 If False: outputs 28x28 matrix format
    """
    # Setup transforms
    if normalize:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        )
    else:
        transform = transforms.ToTensor()

    # Load MNIST dataset
    print(f"Loading MNIST {dataset_type} dataset...")
    dataset = datasets.MNIST(
        root="./data", train=(dataset_type == "train"), download=True, transform=transform
    )

    if image_idx >= len(dataset):
        print(f"Error: Image index {image_idx} is out of range (max: {len(dataset)-1})")
        return

    # Get image and label
    image, label = dataset[image_idx]
    print(f"Image index: {image_idx}")
    print(f"True label: {label}")
    print(f"Image shape: {image.shape}")

    # Convert to numpy array
    # Image shape: (1, 28, 28) -> (28, 28) after squeeze
    image_np = image.squeeze().numpy()

    # Handle normalization
    if normalize:
        # Denormalize to get original range approximately
        # Original normalization: (x - 0.1307) / 0.3081
        # Reverse: x * 0.3081 + 0.1307
        image_np = image_np * 0.3081 + 0.1307
        # Clip to [0, 1] range
        image_np = np.clip(image_np, 0, 1)
        # Scale to 0-255 for integer representation
        image_np = (image_np * 255).astype(np.uint8)
    else:
        # Already in 0-255 range
        image_np = image_np.astype(np.uint8)

    # Flatten if requested
    if flatten:
        image_flat = image_np.flatten()
    else:
        image_flat = image_np

    # Write to file
    print(f"Writing to {output_file}...")
    with open(output_file, "w") as f:
        if format == "hex":
            # Hexadecimal format (for $readmemh in Verilog)
            for val in image_flat:
                f.write(f"{val:02x}\n")
        elif format == "dec":
            # Decimal format
            for val in image_flat:
                f.write(f"{val}\n")
        elif format == "bin":
            # Binary format (8-bit)
            for val in image_flat:
                f.write(f"{val:08b}\n")
        else:
            raise ValueError(f"Unknown format: {format}")

    print(f"Successfully converted image to {output_file}")
    print(f"Format: {format}, Values: {len(image_flat)}")
    print(f"Value range: [{image_flat.min()}, {image_flat.max()}]")
    if not flatten:
        print(f"Image shape: {image_np.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MNIST image to text file for testbench"
    )
    parser.add_argument(
        "-i", "--index", type=int, default=0, help="Image index in dataset (default: 0)"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Dataset type: train or test (default: test)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="mnist_image.txt",
        help="Output file path (default: mnist_image.txt)",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["hex", "dec", "bin"],
        default="hex",
        help="Output format: hex, dec, or bin (default: hex)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Use raw pixel values (0-255) instead of normalized",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Output as 28x28 matrix instead of flattened (784 values)",
    )

    args = parser.parse_args()

    convert_mnist_to_txt(
        image_idx=args.index,
        dataset_type=args.dataset,
        output_file=args.output,
        format=args.format,
        normalize=not args.no_normalize,
        flatten=not args.matrix,
    )


if __name__ == "__main__":
    main()

