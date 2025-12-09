
# PyTorch Neural Network 教學範例

本專案為一系列 PyTorch 教學示範，從模型建構、資料餵入、Loss/Optimizer 選擇，到正則化、Dropout 及 Batch Normalization。可供課堂使用與自學練習。

---

## 📁 教學目錄與說明

| 檔名 | 說明 |
|------|------|
| `env.py` | 自動偵測運算設備（CPU / CUDA / Apple MPS） |
| `ex1-seq_build.py` | 使用 `nn.Sequential` 快速建立簡單模型 |
| `ex2-class_build.py` | 使用 `nn.Module` 自訂 class-based 模型建構方式 |
| `ex3-simple_seq.py` | `Sequential` 模型加上前向推論與輸出 |
| `ex4-simple_class.py` | 自訂 class 模型並執行 forward 與輸出 |
| `ex5-diff_loss.py` | 練習切換不同損失函數：CrossEntropy、MSE |
| `ex6-diff_opt.py` | 練習不同 Optimizer：SGD、Adam、RMSprop、Adagrad |
| `ex7-feed_data.py` | 練習不同餵資料方式：整批、手動 batch、DataLoader、逐筆 |
| `ex8-regularization.py` | 加入 L1 / L2 Regularization，學習模型正則化技巧 |
| `ex9-dropout.py` | 模型中加入 Dropout，觀察對訓練影響 |
| `ex10-BN.py` | 模型加入 Batch Normalization，觀察穩定訓練效果 |
| `ex11-simple_cnn.py` | 建立基本 CNN 架構（2層 conv + pooling），理解影像分類流程 |
| `ex12-cnn.py` | 完整 CNN 訓練流程，支援 Dropout、BatchNorm、Early Stopping 與模型儲存 |
| `ex13-transfer_resnet.py` | 使用預訓練 ResNet18 進行 Transfer Learning，微調最後分類層 |
| `ex13_0-GradCam.py` | Grad-CAM 視覺化實作，觀察 CNN 模型關注的影像區域熱力圖 |
| `ex13_1-face_mediapipe.py` | 使用 MediaPipe 進行即時人臉偵測（攝影機串流） |
| `ex13_2-ges_mediapipe.py` | 使用 MediaPipe 進行手勢辨識（比讚、愛心、數字 1-5） |
| `ex14-w2v.py` | 示範如何載入並使用 Word2Vec 詞向量於 NLP 任務中（如文本分類） |
| `ex15-rnn.py` | 基礎 RNN 模型，使用 Sine Wave 時間序列預測 |
| `ex16-stack_rnn.py` | 堆疊式 RNN 預測股價（AAPL），比較直接預測與差分預測方法 |
| `ex17-lstm.py` | LSTM 模型預測股價（AAPL），學習長短期記憶網路架構 |
| `ex18-lstm_midi_generator.py` | LSTM 音樂生成器，讀取 MIDI 檔案訓練並自動作曲（需下載 MAESTRO 資料集） |
| `ex19-transformer.py` | Transformer 模型預測加密貨幣價格（ETH-USD） |
| `ex20-gan.py` | GAN 生成對抗網路，使用 MNIST 資料集生成手寫數字圖像 |
| `requirements.txt` | 安裝相依套件清單（建議建立虛擬環境安裝） |

---

## 🔧 安裝環境

建議使用 Python 3.11 搭配虛擬環境：

```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 執行方式

```bash
python ex1-seq_build.py
```

或依序照章節進行練習。每支程式皆可獨立執行。

---

## 🎓 推薦學習順序

### 基礎篇
1. `ex1` → `ex2`：模型建立方式（Sequential vs Class）
2. `ex3` ~ `ex6`：基本訓練流程、損失函數與優化器選擇
3. `ex7`：餵資料方式理解
4. `ex8` ~ `ex10`：正則化與訓練穩定技巧（L1/L2、Dropout、BatchNorm）

### 影像處理篇
5. `ex11` → `ex12`：CNN 卷積神經網路從簡單到完整
6. `ex13`：Transfer Learning 遷移學習
7. `ex13_0`：Grad-CAM 模型可解釋性視覺化
8. `ex13_1` → `ex13_2`：MediaPipe 人臉/手勢偵測應用

### 序列模型篇
9. `ex14`：Word2Vec 詞向量應用
10. `ex15` → `ex16`：RNN 基礎與堆疊式 RNN
11. `ex17` → `ex18`：LSTM 股價預測與音樂生成
12. `ex19`：Transformer 架構入門

### 生成模型篇
13. `ex20`：GAN 生成對抗網路

---

## 📦 額外資料集下載

### MAESTRO 資料集（ex18 音樂生成用）

`ex18-lstm_midi_generator.py` 需要 MAESTRO 鋼琴 MIDI 資料集：

1. 前往 [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) 下載
2. 或使用指令下載：
   ```bash
   wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
   unzip maestro-v3.0.0-midi.zip
   ```
3. 解壓後將 `maestro-v3.0.0` 資料夾放在專案根目錄

---

歡迎用於課程教學、實驗設計與學生報告練習。

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
