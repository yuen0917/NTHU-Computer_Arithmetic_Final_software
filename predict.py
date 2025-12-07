import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from main import GlobalAvgPoolCNN
from quantized_model_improved import QuantizedGlobalAvgPoolCNN_Improved


def predict_with_quantized_model(
    weights_path="weights/mnist_weights.npz",
    batch_size=64,
    use_lut=True,
):
    """
    Load trained weights and make predictions using quantized model
    """
    # Load float model first (needed for quantization)
    print("Loading float model...")
    float_model = GlobalAvgPoolCNN(weights_path=weights_path)

    # Create quantized model
    print("Creating quantized model...")
    model = QuantizedGlobalAvgPoolCNN_Improved(
        float_model, dtype=np.int8, use_lut=use_lut
    )

    # Load MNIST test dataset
    print("Loading MNIST test dataset...")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        ]
    )

    test_dataset_full = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Use only first 1000 samples
    test_dataset = Subset(test_dataset_full, range(1000))

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"Test samples: {len(test_dataset)} (using first 100 samples)")
    print("\nMaking predictions...")

    # Make predictions
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    # Use tqdm for progress bar
    for batch_idx, (data, target) in enumerate(
        tqdm(test_loader, desc="Predicting", unit="batch")
    ):
        # Convert PyTorch tensor to NumPy array
        # Data shape: (N, 1, 28, 28)
        data_np = data.numpy().astype(np.float32)
        target_np = target.numpy()

        # Forward pass
        probs = model.forward(data_np)
        predictions = np.argmax(probs, axis=1)

        # Calculate accuracy
        correct += np.sum(predictions == target_np)
        total += len(target_np)

        all_predictions.extend(predictions)
        all_labels.extend(target_np)

    accuracy = 100.0 * correct / total
    print(f"\n{'='*50}")
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")
    print(f"Model Type: Quantized (int8, {'LUT' if use_lut else 'Approximation'})")
    print(f"{'='*50}")

    # Show some example predictions
    print("\nExample predictions (first 10 samples):")
    for i in range(min(10, len(all_predictions))):
        print(
            f"  Sample {i+1}: Predicted={all_predictions[i]}, True={all_labels[i]}, "
            f"{'✓' if all_predictions[i] == all_labels[i] else '✗'}"
        )

    return accuracy, all_predictions, all_labels


def predict_with_numpy_model(weights_path="weights/mnist_weights.npz", batch_size=64):
    """
    Load trained weights and make predictions on MNIST test set (float32 version)
    """
    # Load model with trained weights
    print("Loading trained model (float32)...")
    model = GlobalAvgPoolCNN(weights_path=weights_path)

    # Load MNIST test dataset
    print("Loading MNIST test dataset...")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        ]
    )

    test_dataset_full = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Use only first 100 samples
    test_dataset = Subset(test_dataset_full, range(100))

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"Test samples: {len(test_dataset)} (using first 100 samples)")
    print("\nMaking predictions...")

    # Make predictions
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    # Use tqdm for progress bar
    for batch_idx, (data, target) in enumerate(
        tqdm(test_loader, desc="Predicting", unit="batch")
    ):
        # Convert PyTorch tensor to NumPy array
        # Data shape: (N, 1, 28, 28)
        data_np = data.numpy().astype(np.float32)
        target_np = target.numpy()

        # Forward pass
        probs = model.forward(data_np)
        predictions = np.argmax(probs, axis=1)

        # Calculate accuracy
        correct += np.sum(predictions == target_np)
        total += len(target_np)

        all_predictions.extend(predictions)
        all_labels.extend(target_np)

    accuracy = 100.0 * correct / total
    print(f"\n{'='*50}")
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")
    print(f"Model Type: Float32")
    print(f"{'='*50}")

    # Show some example predictions
    print("\nExample predictions (first 10 samples):")
    for i in range(min(10, len(all_predictions))):
        print(
            f"  Sample {i+1}: Predicted={all_predictions[i]}, True={all_labels[i]}, "
            f"{'✓' if all_predictions[i] == all_labels[i] else '✗'}"
        )

    return accuracy, all_predictions, all_labels


def predict_single_image(model, image):
    """
    Predict a single image
    Args:
        model: GlobalAvgPoolCNN model instance
        image: numpy array of shape (1, 28, 28) or (28, 28)
    Returns:
        predicted_class: int (0-9)
        probabilities: numpy array of shape (10,)
    """
    # Ensure correct shape
    if image.ndim == 2:
        image = image[np.newaxis, :, :]  # (28, 28) -> (1, 28, 28)
    if image.ndim == 3 and image.shape[0] != 1:
        image = image[np.newaxis, :, :, :]  # Add batch dimension

    # Forward pass
    probs = model.forward(image)
    predicted_class = np.argmax(probs, axis=1)[0]

    return predicted_class, probs[0]


if __name__ == "__main__":
    import os

    weights_path = "weights/mnist_weights.npz"

    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        print("Please run train.py first to train the model.")
    else:
        # Use quantized model by default (with approximation for better accuracy)
        print("Using quantized model for prediction...")
        accuracy, predictions, labels = predict_with_quantized_model(
            weights_path, use_lut=False  # Use approximation for better accuracy
        )
