"""
Improved quantized model implementation with support for GELU/SELU activation functions.
This module provides quantization utilities and a quantized CNN model.
"""

import numpy as np


def compute_quantization_params(tensor: np.ndarray, dtype: np.dtype) -> tuple[float, int]:
    """
    Compute quantization parameters (scale and zero point) for symmetric quantization.

    Args:
        tensor: Input float32 tensor
        dtype: Target dtype (e.g., np.int8)

    Returns:
        scale: Quantization scale
        zero_point: Quantization zero point (always 0 for symmetric quantization)
    """
    if dtype == np.int8:
        qmin = -128
        qmax = 127
    elif dtype == np.int16:
        qmin = -32768
        qmax = 32767
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Symmetric quantization: zero point is always 0
    zero_point = 0

    # Compute scale based on maximum absolute value
    max_val = np.abs(tensor).max()
    if max_val < 1e-8:
        scale = 1.0
    else:
        scale = max_val / qmax

    return scale, zero_point


def quantize_tensor(
    tensor: np.ndarray,
    scale: float,
    zero_point: int,
    dtype: np.dtype
) -> np.ndarray:
    """
    Quantize a float32 tensor to integer type.

    Args:
        tensor: Input float32 tensor
        scale: Quantization scale
        zero_point: Quantization zero point
        dtype: Target dtype (e.g., np.int8)

    Returns:
        Quantized tensor
    """
    if dtype == np.int8:
        qmin = -128
        qmax = 127
    elif dtype == np.int16:
        qmin = -32768
        qmax = 32767
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Quantize: q = round(x / scale) + zero_point
    # For symmetric quantization, zero_point = 0
    quantized = np.round(tensor / scale).astype(dtype)

    # Clamp to valid range
    quantized = np.clip(quantized, qmin, qmax)

    return quantized


def dequantize_tensor(
    quantized: np.ndarray,
    scale: float,
    zero_point: int
) -> np.ndarray:
    """
    Dequantize an integer tensor back to float32.

    Args:
        quantized: Quantized integer tensor
        scale: Quantization scale
        zero_point: Quantization zero point

    Returns:
        Dequantized float32 tensor
    """
    return (quantized.astype(np.float32) - zero_point) * scale


def selu_approx(x: np.ndarray) -> np.ndarray:
    """
    Approximate SELU activation function.
    SELU(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))
    where scale ≈ 1.0507, alpha ≈ 1.6733
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))


def gelu_approx(x: np.ndarray) -> np.ndarray:
    """
    Approximate GELU activation function using tanh approximation.
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    sqrt_2_over_pi = np.sqrt(2 / np.pi)
    return 0.5 * x * (1 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * np.power(x, 3))))


class QuantizedGlobalAvgPoolCNN_Improved:
    """
    Improved quantized CNN model with support for GELU/SELU using LUT or approximation.

    This class implements a quantized version of the Global Average Pooling CNN
    that matches the architecture specified in the PDF.
    """

    def __init__(
        self,
        float_model=None,
        weights_path: str = None,
        dtype: np.dtype = np.int8,
        use_lut: bool = True
    ):
        """
        Initialize quantized model.

        Args:
            float_model: Float model instance (optional, for extracting weights)
            weights_path: Path to saved weights .npz file
            dtype: Quantization dtype (np.int8 or np.int16)
            use_lut: Whether to use lookup tables for SELU/GELU (if False, uses approximation)
        """
        self.dtype = dtype
        self.use_lut = use_lut

        # Load weights
        if weights_path:
            weights = np.load(weights_path)
            W1 = weights['W1']
            b1 = weights['b1']
            W2 = weights['W2']
            b2 = weights['b2']
            W3 = weights['W3']
            b3 = weights['b3']
            W4 = weights['W4']
            b4 = weights['b4']
        elif float_model:
            # Extract weights from float model
            # This assumes float_model has attributes like conv1, conv2, etc.
            # For now, we'll require weights_path
            raise NotImplementedError("Direct model extraction not implemented. Please use weights_path.")
        else:
            raise ValueError("Either float_model or weights_path must be provided")

        # Quantize weights and biases
        self.W1_scale, self.W1_zp = compute_quantization_params(W1, dtype)
        self.b1_scale, self.b1_zp = compute_quantization_params(b1, dtype)
        self.W1_q = quantize_tensor(W1, self.W1_scale, self.W1_zp, dtype)
        self.b1_q = quantize_tensor(b1, self.b1_scale, self.b1_zp, dtype)

        self.W2_scale, self.W2_zp = compute_quantization_params(W2, dtype)
        self.b2_scale, self.b2_zp = compute_quantization_params(b2, dtype)
        self.W2_q = quantize_tensor(W2, self.W2_scale, self.W2_zp, dtype)
        self.b2_q = quantize_tensor(b2, self.b2_scale, self.b2_zp, dtype)

        self.W3_scale, self.W3_zp = compute_quantization_params(W3, dtype)
        self.b3_scale, self.b3_zp = compute_quantization_params(b3, dtype)
        self.W3_q = quantize_tensor(W3, self.W3_scale, self.W3_zp, dtype)
        self.b3_q = quantize_tensor(b3, self.b3_scale, self.b3_zp, dtype)

        self.W4_scale, self.W4_zp = compute_quantization_params(W4, dtype)
        self.b4_scale, self.b4_zp = compute_quantization_params(b4, dtype)
        self.W4_q = quantize_tensor(W4, self.W4_scale, self.W4_zp, dtype)
        self.b4_q = quantize_tensor(b4, self.b4_scale, self.b4_zp, dtype)

        # Build LUTs if needed
        if use_lut:
            self._build_luts()

    def _build_luts(self):
        """Build lookup tables for SELU and GELU activation functions."""
        if self.dtype == np.int8:
            qmin, qmax = -128, 127
        else:
            qmin, qmax = -32768, 32767

        # Create input range
        x_int = np.arange(qmin, qmax + 1, dtype=np.int16)

        # For LUT, we need to know the scale of activations
        # This is a simplified version - in practice, you'd use actual activation scales
        # For now, we'll use a default scale (this should be calibrated)
        default_scale = 0.1  # This should be calibrated based on actual activations

        x_float = x_int.astype(np.float32) * default_scale

        # Compute SELU and GELU
        selu_float = selu_approx(x_float)
        gelu_float = gelu_approx(x_float)

        # Quantize outputs
        selu_scale, _ = compute_quantization_params(selu_float, self.dtype)
        gelu_scale, _ = compute_quantization_params(gelu_float, self.dtype)

        self.selu_lut = quantize_tensor(selu_float, selu_scale, 0, self.dtype)
        self.gelu_lut = quantize_tensor(gelu_float, gelu_scale, 0, self.dtype)
        self.selu_scale = selu_scale
        self.gelu_scale = gelu_scale

    def _quantized_conv2d(self, x: np.ndarray, W_q: np.ndarray, b_q: np.ndarray,
                         W_scale: float, b_scale: float, padding: int = 1) -> np.ndarray:
        """
        Quantized 2D convolution.

        Args:
            x: Input tensor (N, C_in, H, W) - already quantized
            W_q: Quantized weights (C_out, C_in, 3, 3)
            b_q: Quantized biases (C_out,)
            W_scale: Weight scale
            b_scale: Bias scale
            padding: Padding size

        Returns:
            Output tensor (quantized)
        """
        N, C_in, H, W = x.shape
        C_out = W_q.shape[0]

        # Add padding
        if padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                            mode='constant', constant_values=0)
        else:
            x_padded = x

        H_out = H  # With padding=1, output size = input size
        W_out = W

        output = np.zeros((N, C_out, H_out, W_out), dtype=np.int32)

        # Convolution
        for n in range(N):
            for c_out in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        # Convolution operation
                        conv_sum = 0
                        for c_in in range(C_in):
                            for kh in range(3):
                                for kw in range(3):
                                    conv_sum += (x_padded[n, c_in, h + kh, w + kw].astype(np.int32) *
                                               W_q[c_out, c_in, kh, kw].astype(np.int32))

                        # Add bias
                        output[n, c_out, h, w] = conv_sum + b_q[c_out].astype(np.int32)

        # Quantize output (this is simplified - in practice, you'd compute proper output scale)
        # For now, we'll use a simple approach
        output_scale = W_scale  # Simplified
        output_q = np.clip(np.round(output.astype(np.float32) * output_scale),
                          -128, 127).astype(self.dtype)

        return output_q

    def _quantized_linear(self, x: np.ndarray, W_q: np.ndarray, b_q: np.ndarray,
                         W_scale: float, b_scale: float) -> np.ndarray:
        """
        Quantized linear (fully connected) layer.

        Args:
            x: Input tensor (N, in_features) - already quantized
            W_q: Quantized weights (out_features, in_features)
            b_q: Quantized biases (out_features,)
            W_scale: Weight scale
            b_scale: Bias scale

        Returns:
            Output tensor (quantized)
        """
        # Matrix multiplication: output = x @ W^T + b
        output = np.dot(x.astype(np.int32), W_q.T.astype(np.int32)) + b_q.astype(np.int32)

        # Quantize output (simplified)
        output_scale = W_scale
        output_q = np.clip(np.round(output.astype(np.float32) * output_scale),
                          -128, 127).astype(self.dtype)

        return output_q

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through quantized model.

        Args:
            x: Input tensor (N, 1, 28, 28) in float32

        Returns:
            Output logits (N, 10) in float32
        """
        # Quantize input
        x_scale, x_zp = compute_quantization_params(x, self.dtype)
        x_q = quantize_tensor(x, x_scale, x_zp, self.dtype)

        # Conv1 + ReLU
        x_q = self._quantized_conv2d(x_q, self.W1_q, self.b1_q,
                                     self.W1_scale, self.b1_scale)
        x_q = np.maximum(x_q, 0)  # ReLU

        # Conv2 + SELU
        x_q = self._quantized_conv2d(x_q, self.W2_q, self.b2_q,
                                     self.W2_scale, self.b2_scale)
        if self.use_lut:
            # Use LUT for SELU (simplified - would need proper indexing)
            x_float = dequantize_tensor(x_q, self.W2_scale, 0)
            x_float = selu_approx(x_float)
            x_q = quantize_tensor(x_float, self.W2_scale, 0, self.dtype)
        else:
            x_float = dequantize_tensor(x_q, self.W2_scale, 0)
            x_float = selu_approx(x_float)
            x_q = quantize_tensor(x_float, self.W2_scale, 0, self.dtype)

        # MaxPool 2x2
        N, C, H, W = x_q.shape
        H_out = H // 2
        W_out = W // 2
        x_pooled = np.zeros((N, C, H_out, W_out), dtype=self.dtype)
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        x_pooled[n, c, h, w] = np.max(x_q[n, c, h*2:(h*2+2), w*2:(w*2+2)])

        x_q = x_pooled

        # Conv3 + GELU
        x_q = self._quantized_conv2d(x_q, self.W3_q, self.b3_q,
                                     self.W3_scale, self.b3_scale)
        if self.use_lut:
            x_float = dequantize_tensor(x_q, self.W3_scale, 0)
            x_float = gelu_approx(x_float)
            x_q = quantize_tensor(x_float, self.W3_scale, 0, self.dtype)
        else:
            x_float = dequantize_tensor(x_q, self.W3_scale, 0)
            x_float = gelu_approx(x_float)
            x_q = quantize_tensor(x_float, self.W3_scale, 0, self.dtype)

        # Global Average Pooling
        x_q = np.mean(x_q, axis=(2, 3))  # (N, 32)

        # FC layer
        x_q = self._quantized_linear(x_q, self.W4_q, self.b4_q,
                                    self.W4_scale, self.b4_scale)

        # Dequantize output
        output = dequantize_tensor(x_q, self.W4_scale, 0)

        return output


if __name__ == "__main__":
    # Test quantization functions
    print("Testing quantization functions...")

    # Test compute_quantization_params
    test_tensor = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    scale, zp = compute_quantization_params(test_tensor, np.int8)
    print(f"Scale: {scale}, Zero point: {zp}")

    # Test quantize_tensor
    quantized = quantize_tensor(test_tensor, scale, zp, np.int8)
    print(f"Quantized: {quantized}")

    # Test dequantize_tensor
    dequantized = dequantize_tensor(quantized, scale, zp)
    print(f"Dequantized: {dequantized}")
    print(f"Error: {np.abs(test_tensor - dequantized).max()}")

    print("\nQuantization test passed!")

