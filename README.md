# CNN MNIST 手寫數字識別

從零開始實現的卷積神經網絡（CNN），使用純 NumPy 進行推理，支援整數量化（Quantization）以提升效率。

## 目錄

- [項目簡介](#項目簡介)
- [模型架構](#模型架構)
- [量化架構](#量化架構)
- [安裝與使用](#安裝與使用)
- [查找表生成](#查找表生成)
- [文件結構](#文件結構)
- [性能比較](#性能比較)

## 項目簡介

本項目實現了一個用於 MNIST 手寫數字識別（0-9）的 CNN 模型，具有以下特點：

- **純 NumPy 實現**：不依賴深度學習框架進行推理
- **支援量化**：提供 int8 量化版本，模型大小減少 4 倍
- **多種激活函數**：使用 ReLU、SELU、GELU 等不同激活函數
- **完整訓練流程**：使用 PyTorch 訓練，權重轉換為 NumPy 格式

### 數據集規格

- **任務**：手寫數字分類（0-9，共 10 類）
- **圖像尺寸**：1×28×28（灰階，單通道）
- **數據分割**：訓練 60,000 張、測試 10,000 張
- **標籤**：整數 0-9
- **評估指標**：Top-1 Accuracy

## 模型架構

### 原始架構（Float32）

```text
Input: (N, 1, 28, 28)
  ↓
Conv2d(1→8, kernel=3, padding=1) + ReLU
  → (N, 8, 28, 28)
  ↓
Conv2d(8→16, kernel=3, padding=1) + SELU
  → (N, 16, 28, 28)
  ↓
MaxPool2d(2×2, stride=2)
  → (N, 16, 14, 14)
  ↓
Conv2d(16→32, kernel=3, padding=1) + GELU
  → (N, 32, 14, 14)
  ↓
Global Average Pooling
  → (N, 32, 1, 1)
  ↓
Flatten
  → (N, 32)
  ↓
Linear(32→10) + Softmax
  → (N, 10)
```

### 數學公式

#### 1. 卷積層（Convolution）

對於輸入 $x \in \mathbb{R}^{N \times C_{in} \times H \times W}$ 和權重 $W \in \mathbb{R}^{C_{out} \times C_{in} \times K_h \times K_w}$：

$$y[n, c_{out}, h, w] = \sum_{c_{in}} \sum_{i,j} x[n, c_{in}, h+i, w+j] \cdot W[c_{out}, c_{in}, i, j] + b[c_{out}]$$

#### 2. 激活函數

**ReLU:**
$$f(x) = \max(0, x)$$

**SELU:**

$$
f(x) = \lambda \begin{cases}
x & \text{if } x > 0 \\
\alpha(e^x - 1) & \text{if } x \leq 0
\end{cases}
$$

其中 $\lambda = 1.0507$, $\alpha = 1.6733$

**GELU:**
$$f(x) = 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)$$

#### 3. 最大池化（MaxPooling）

$$y[n, c, h, w] = \max_{i,j \in [0,1]} x[n, c, 2h+i, 2w+j]$$

#### 4. 全局平均池化（Global Average Pooling）

$$y[n, c] = \frac{1}{H \times W} \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} x[n, c, i, j]$$

#### 5. 全連接層（Linear）

$$y = xW^T + b$$

其中 $x \in \mathbb{R}^{N \times d_{in}}$, $W \in \mathbb{R}^{d_{out} \times d_{in}}$, $b \in \mathbb{R}^{d_{out}}$

#### 6. Softmax

$$p_i = \frac{e^{x_i - \max(x)}}{\sum_{j} e^{x_j - \max(x)}}$$

## 量化架構

### 量化原理

量化將浮點數轉換為整數，使用以下公式：

**量化（Quantization）:**
$$Q(x) = \text{round}\left(\frac{x}{S}\right) + Z$$

**反量化（Dequantization）:**
$$x = (Q(x) - Z) \times S$$

其中：

- $S$ (scale)：量化縮放因子
- $Z$ (zero_point)：量化零點
- $Q(x)$：量化後的整數值

### 量化架構公式

#### 1. 量化卷積層

對於量化權重 $W_q$ 和量化輸入 $x_q$：

**方法 A：反量化後計算（當前實現）**
$$y = \text{Dequantize}(W_q) \circledast \text{Dequantize}(x_q) + \text{Dequantize}(b_q)$$

**方法 B：純整數計算（硬體實現）**
$$y_q = \text{round}\left(\frac{W_q \circledast x_q}{S_{out}}\right) + Z_{out}$$

其中 $S_{out} = S_W \times S_x / S_y$

#### 2. 量化激活函數

**ReLU（簡單，無需量化）:**
$$y_q = \max(0, x_q)$$

**SELU / GELU（使用查找表 LUT）:**
$$y_q = \text{LUT}[\text{clip}(x_q, q_{min}, q_{max})]$$

查找表預先計算：
$$\text{LUT}[i] = Q(f(\text{Dequantize}(i)))$$

**SELU / GELU（使用分段線性近似）:**
$$y_q = Q(f_{approx}(\text{Dequantize}(x_q)))$$

其中 $f_{approx}$ 是分段線性近似函數。

#### 3. 量化池化

最大池化在整數域直接適用：
$$y_q[n, c, h, w] = \max_{i,j} x_q[n, c, 2h+i, 2w+j]$$

全局平均池化需要反量化：
$$y = \frac{1}{HW}\sum_{i,j} \text{Dequantize}(x_q[n, c, i, j])$$
$$y_q = Q(y)$$

### 量化參數計算

對於張量 $x$，量化參數計算如下：

$$S = \frac{r_{max} - r_{min}}{q_{max} - q_{min}}$$

$$Z = q_{min} - \frac{r_{min}}{S}$$

其中：

- $r_{min}, r_{max}$：浮點張量的最小值和最大值
- $q_{min}, q_{max}$：整數範圍（int8: -128 到 127）

### 量化架構流程

```text
Input: (N, 1, 28, 28) [float32]
  ↓ Quantize
Input_q: (N, 1, 28, 28) [int8]
  ↓
Conv2d_q(1→8) + ReLU
  → (N, 8, 28, 28) [int8]
  ↓
Conv2d_q(8→16) + SELU_q (LUT or Approx)
  → (N, 16, 28, 28) [int8]
  ↓
MaxPool2d(2×2)
  → (N, 16, 14, 14) [int8]
  ↓
Conv2d_q(16→32) + GELU_q (LUT or Approx)
  → (N, 32, 14, 14) [int8]
  ↓
Global Average Pooling + Quantize
  → (N, 32, 1, 1) [int8]
  ↓
Flatten
  → (N, 32) [int8]
  ↓
Linear_q(32→10) + Dequantize
  → (N, 10) [float32]
  ↓
Softmax
  → (N, 10) [float32]
```

## 安裝與使用

### 環境要求

- Python >= 3.13
- NumPy >= 2.3.5
- PyTorch >= 2.0.0（僅用於訓練）
- torchvision >= 0.15.0（僅用於數據加載）
- tqdm >= 4.66.0（用於進度條）

### 安裝

```bash
# 使用 uv（推薦）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 訓練模型（浮點）

```bash
python train.py
```

這會：

1. 自動下載 MNIST 數據集
2. 訓練模型 5 個 epochs
3. 將浮點權重保存為 `weights/mnist_weights.npz`
4. **自動量化權重並保存為 `weights/mnist_weights_quantized.npz`**

訓練完成後會顯示模型大小比較（浮點 vs 量化）。

### 使用量化模型進行預測（默認）

```bash
python predict.py
```

**默認使用量化模型**（int8，查找表方法）進行預測。這會：

1. 載入浮點權重
2. 創建量化模型
3. 在測試集上進行預測
4. 顯示準確率和模型類型

### 使用浮點模型進行預測

如果需要使用浮點模型，可以修改 `predict.py` 或直接調用：

```python
from predict import predict_with_numpy_model

accuracy, predictions, labels = predict_with_numpy_model("weights/mnist_weights.npz")
```

### 自定義量化模型使用

```python
from main import GlobalAvgPoolCNN
from quantized_model_improved import QuantizedGlobalAvgPoolCNN_Improved

# 載入浮點模型
float_model = GlobalAvgPoolCNN(weights_path="weights/mnist_weights.npz")

# 創建量化模型（使用查找表，精度更高）
quantized_model = QuantizedGlobalAvgPoolCNN_Improved(
    float_model,
    dtype=np.int8,
    use_lut=True
)

# 或使用近似方法（更省記憶體）
quantized_model = QuantizedGlobalAvgPoolCNN_Improved(
    float_model,
    dtype=np.int8,
    use_lut=False
)

# 進行預測
probs = quantized_model.forward(x)  # x: (N, 1, 28, 28) float32
```

### 比較量化方法

```bash
python quantized_model_improved.py
```

這會比較 LUT 方法和近似方法的性能差異。

### 診斷工具

```bash
# 檢查 PyTorch CUDA 環境
python test.py
```

此工具會顯示 PyTorch 版本、CUDA 可用性、GPU 資訊等診斷資訊，幫助排查訓練環境問題。

### 訓練與預測（整數域／硬體對齊）

整數感知訓練：輸入維持 0~255 範圍，便於硬體對齊

```bash
python train_integer.py
```

- 產出 Verilog 友善的權重文字檔：`weights/conv1_relu.txt`、`conv2_selu.txt`、`conv3_gelu.txt`、`fc_weights.txt`、`fc_biases.txt`

整數域批次預測（快速驗證權重與 LUT）：

```bash
python predict_integer.py
```

- 讀取上述權重與 `selu_lut.txt`、`gelu_lut.txt`，跑硬體等效流程 (Conv→ReLU/SELU/GELU→MaxPool→GAP→FC)，輸出整體正確率

單張除錯（查看中間層）：

```bash
python verify_one.py
```

- 讀取 `test_image.txt` 與權重/LUT，逐層列印特徵圖與 GAP/FC，方便與 RTL 波形比對

### 轉換權重檔案為文字檔（Testbench 用）

將量化權重 NPZ 檔案轉換為文字檔，方便在 Verilog/SystemVerilog testbench 中使用：

```bash
# 將量化權重轉換為文字檔
python transfer_npz_txt.py
```

此腳本會從 `weights/mnist_weights_quantized.npz` 讀取量化權重，並生成以下文字檔：

- `weights/conv1_relu.txt` - 第一層卷積權重（1→8, 3×3，共 72 個值）
- `weights/conv2_selu.txt` - 第二層卷積權重（8→16, 3×3，共 1152 個值）
- `weights/conv3_gelu.txt` - 第三層卷積權重（16→32, 3×3，共 4608 個值）
- `weights/fc_weights.txt` - 全連接層權重（32→10，共 320 個值）
- `weights/fc_biases.txt` - 全連接層偏置（10 個值，32-bit）

**輸出格式**：

- 所有權重以十六進制格式儲存（每行一個值）
- 權重使用 8-bit 有符號整數（int8）格式
- 偏置使用 32-bit 有符號整數格式（用於累加運算）
- 使用二補數表示法，適合 Verilog 的 `$readmemh` 指令

**使用範例**（在 Verilog 中讀取）：

```verilog
// 讀取卷積層權重
reg signed [7:0] conv1_weights [0:71];
initial $readmemh("weights/conv1_relu.txt", conv1_weights);

// 讀取全連接層權重
reg signed [7:0] fc_weights [0:319];
initial $readmemh("weights/fc_weights.txt", fc_weights);

// 讀取偏置（32-bit）
reg signed [31:0] fc_biases [0:9];
initial $readmemh("weights/fc_biases.txt", fc_biases);
```

### 轉換 MNIST 圖片為文字檔（Testbench 用）

單張輸出：

```bash
# 轉換測試集第 0 張圖片，十六進制
python generate_mnist_to_txt.py

# 指定索引、格式、輸出檔名
python generate_mnist_to_txt.py -i 5 -f bin -o img5_bin.txt

# 使用原始像素（不正規化）、保留矩陣格式
python generate_mnist_to_txt.py --no-normalize --matrix -o img5_mat.txt
```

批次輸出（前 100 張，含標籤）：

```bash
python generate_batch_mnist_to_txt.py
```

會產生 `all_test_images.txt`（每行 1 byte，十六進制）與 `all_labels.txt`（每行 1 label，十六進制），方便 Verilog/FPGA 批次測試。

**輸出格式**：

- **十六進制**（`hex`）：每行一個十六進制值（如 `3f`），適合 Verilog 的 `$readmemh` 指令
- **十進制**（`dec`）：每行一個十進制值（如 `63`）
- **二進制**（`bin`）：每行一個 8 位元二進制值（如 `00111111`）

**範例輸出**（十六進制格式，前 10 行）：

```text
00
00
00
3f
7f
...
```

## 查找表生成

查找表（Look-Up Table, LUT）用於在量化模型中實現非線性激活函數（SELU、GELU）。本項目提供了生成這些查找表的腳本。

### 生成查找表

如果需要重新生成 SELU 或 GELU 的查找表檔案，可以使用以下腳本：

```bash
# 生成 SELU 查找表
python generate_selu_lut.py

# 生成 GELU 查找表
python generate_gelu_lut.py
```

這些腳本會生成 `selu_lut.txt` 和 `gelu_lut.txt` 檔案，用於量化模型中的激活函數查找表。查找表以十六進制格式儲存，每個值對應 int8 範圍（-128 到 127）的輸入。

### 查找表格式

- **輸入範圍**：int8 範圍（-128 到 127，共 256 個值）
- **計算方式**：對每個整數輸入值，計算對應的激活函數輸出
- **量化輸出**：將浮點輸出量化回 int8 範圍
- **儲存格式**：十六進制格式（每行一個值），方便硬體實現讀取

**SELU 查找表**：

- 使用標準 SELU 公式：$\lambda = 1.0507$, $\alpha = 1.6733$
- 對負值使用指數函數：$\alpha(e^x - 1)$

**GELU 查找表**：

- 使用近似公式：$0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$
- 適合硬體實現的平滑激活函數

## 文件結構

```text
mnist/
├── main.py                      # 核心 CNN 實現（浮點版本）
├── train.py                     # PyTorch 訓練腳本（自動量化）
├── predict.py                   # 預測腳本（默認使用量化模型）
├── predict_integer.py           # 整數域批次預測（硬體等效流程）
├── test.py                      # PyTorch CUDA 診斷工具
├── quantized_model.py           # 基礎量化實現
├── quantized_model_improved.py  # 改進的量化實現（支援 GELU/SELU）
├── train.py                     # 浮點訓練（自動量化）
├── train_integer.py             # 整數感知訓練，直接輸出 Verilog 權重 txt
├── generate_selu_lut.py         # SELU 查找表生成腳本
├── generate_gelu_lut.py         # GELU 查找表生成腳本
├── transfer_npz_txt.py          # 權重 NPZ 轉文字檔工具（Testbench 用）
├── generate_mnist_to_txt.py     # 單張 MNIST 轉文字檔
├── generate_batch_mnist_to_txt.py # 批次 MNIST 轉文字檔（圖像+標籤）
├── verify_one.py                # 單張硬體流程除錯（列印中間層）
├── pyproject.toml               # 項目配置
├── .gitignore                   # Git 忽略檔案配置
├── README.md                    # 本文件
├── weights/                     # 權重檔案目錄（自動生成，不上傳）
│   ├── mnist_weights.npz              # 浮點權重
│   ├── mnist_weights_quantized.npz   # 量化權重（自動生成）
│   ├── conv1_relu.txt                # 第一層卷積權重（Verilog）
│   ├── conv2_selu.txt                # 第二層卷積權重（Verilog）
│   ├── conv3_gelu.txt                # 第三層卷積權重（Verilog）
│   ├── fc_weights.txt                # 全連接層權重（Verilog）
│   └── fc_biases.txt                 # 全連接層偏置（Verilog，32-bit）
├── data/                        # 數據集目錄（自動下載，不上傳）
│   └── MNIST/                   # MNIST 數據集
├── selu_lut.txt                 # SELU 查找表（自動生成，不上傳）
└── gelu_lut.txt                 # GELU 查找表（自動生成，不上傳）
```

**注意**：`weights/`、`data/`、`selu_lut.txt`、`gelu_lut.txt` 等生成的檔案已配置在 `.gitignore` 中，不會上傳到版本控制系統。

### 量化方法

1. **權重量化**：所有權重和偏置使用 int8 量化
2. **激活量化**：中間激活值可選量化（當前實現為混合精度）
3. **查找表（LUT）**：用於 GELU 和 SELU 的整數域計算
4. **分段線性近似**：替代 LUT 的輕量級方法

### 激活函數處理

- **ReLU**：直接適用於整數域
- **SELU**：使用 LUT 或分段線性近似
- **GELU**：使用 LUT 或分段線性近似
- **Softmax**：在浮點域計算（最後一層）

## 參考資料

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Quantization in Deep Learning](https://pytorch.org/docs/stable/quantization.html)
- [GELU Paper](https://arxiv.org/abs/1606.08415)
- [SELU Paper](https://arxiv.org/abs/1706.02515)

## 授權

本項目僅用於學習和研究目的。
