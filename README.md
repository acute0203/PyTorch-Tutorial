
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
| `ex14-w2v.py` | 示範如何載入並使用 Word2Vec 詞向量於 NLP 任務中（如文本分類） |
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

1. `ex1-seq_build.py` → `ex2-class_build.py`：模型建立方式
2. `ex3` ~ `ex6`：基本訓練流程、損失與優化器選擇
3. `ex7`：餵資料方式理解
4. `ex8` ~ `ex10`：正則化與訓練穩定技巧

---

歡迎用於課程教學、實驗設計與學生報告練習。

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
