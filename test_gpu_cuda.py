import torch
import sys

print("=" * 60)
print("PyTorch CUDA 診斷資訊")
print("=" * 60)

print(f"\nPython 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"PyTorch 安裝路徑: {torch.__file__}")

# Check if CUDA is compiled in PyTorch
print(f"\nCUDA 是否編譯進 PyTorch: {torch.cuda.is_available()}")
if hasattr(torch.version, 'cuda'):
    print(f"PyTorch 編譯時的 CUDA 版本: {torch.version.cuda}")
else:
    print("PyTorch 編譯時的 CUDA 版本: None (CPU 版本)")

# Check CUDA availability
print(f"\nCUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
    print(f"GPU 數量: {torch.cuda.device_count()}")
    print(f"當前設備: {torch.cuda.current_device()}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
else:
    print("\n[錯誤] 沒有檢測到 CUDA 支援")
    print("\n可能的原因：")
    print("1. PyTorch 安裝的是 CPU 版本（最常見）")
    print("2. NVIDIA GPU 驅動程式未安裝或版本過舊")
    print("3. CUDA Toolkit 未安裝或版本不匹配")
    print("4. 虛擬環境中安裝的 PyTorch 版本錯誤")

    print("\n解決方案：")
    print("1. 檢查是否有 NVIDIA GPU:")
    print("   - Windows: 開啟工作管理員 > 效能 > GPU")
    print("   - 或執行: nvidia-smi (如果有安裝驅動程式)")

    print("\n2. 如果使用 uv/pip，重新安裝 CUDA 版本的 PyTorch:")
    print("   pip uninstall torch torchvision")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("   (或 cu118, cu124 根據你的 CUDA 版本)")

    print("\n3. 如果使用虛擬環境，確保在正確的環境中安裝")
    print("4. 安裝後重新啟動 Python 環境")

print("\n" + "=" * 60)
