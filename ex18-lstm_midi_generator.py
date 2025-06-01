import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from music21 import converter, instrument, note, chord, stream
from collections import Counter

# 設定裝置（可支援 Mac M1/M2 的 GPU 加速）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"⚙️ 使用裝置：{device}")

# === Step 1: 讀取 MIDI 檔案並解析為音符序列 ===
def load_midi_notes(midi_folder):
    notes = []
    r = midi_folder
    for y in range(2004, 2005):  # 此範例只處理 2004 年資料夾，可自行調整
        r_midi_folder = f"{r}/{y}/"
        idx = 0
        for file in glob.glob(os.path.join(r_midi_folder, "*.midi")):
            idx += 1
            print(idx, file)
            midi = converter.parse(file)
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))  # 轉換單一音符為字串 pitch
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))  # 和弦轉為字串
    return notes

# === Step 2: 轉換音符為數值序列，準備訓練資料 ===
def prepare_sequences(notes, sequence_length=30, min_freq=1):
    note_counts = Counter(notes)
    frequent_notes = [note for note, count in note_counts.items() if count >= min_freq]
    note_to_int = {note: idx for idx, note in enumerate(frequent_notes)}
    int_to_note = {idx: note for note, idx in note_to_int.items()}

    network_input, network_output = [], []
    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        if all(n in note_to_int for n in sequence_in) and sequence_out in note_to_int:
            network_input.append([note_to_int[n] for n in sequence_in])
            network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(len(note_to_int))
    network_output = np.array(network_output)
    print(f"可用訓練樣本數量：{len(network_input)}，Note種類：{len(note_to_int)}")
    return network_input, network_output, note_to_int, int_to_note

# === Step 3: 定義 LSTM 模型架構 ===
class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MusicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])  # 只取最後一個時間點的輸出做分類
        return out

# === Step 4: 模型訓練流程 ===
def train(model, network_input, network_output, epochs=20, batch_size=32, lr=0.001):
    model.to(device)  # 模型丟到 GPU 或 CPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"EPOCH:{epoch}")
        epoch_loss = 0
        count = 0
        for i in range(0, len(network_input) - batch_size, batch_size):
            inputs = torch.tensor(network_input[i:i+batch_size], dtype=torch.float32).to(device)
            targets = torch.tensor(network_output[i:i+batch_size], dtype=torch.long).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            count += 1

        if count > 0:
            avg_loss = epoch_loss / count
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, ⚠️ 跳過：資料不足以形成一個 batch")

# === Step 5: 使用訓練好的模型生成音樂 MIDI ===
def generate_music(model, network_input, int_to_note, note_count, output_file='output.mid'):
    if len(network_input) < 1:
        print("❌ 無法生成音樂，network_input 太短")
        return

    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    pattern = torch.tensor(pattern, dtype=torch.float32).unsqueeze(0).to(device)

    generated_notes = []

    model.eval()
    with torch.no_grad():
        for _ in range(500):
            prediction = model(pattern)
            _, index = torch.max(prediction, 1)
            result = int_to_note[index.item()]
            generated_notes.append(result)

            new_input = torch.tensor([[index.item() / float(note_count)]], dtype=torch.float32).to(device)
            pattern = torch.cat((pattern[:, 1:, :], new_input.unsqueeze(0)), dim=1)

    offset = 0
    output_notes = []
    for pattern in generated_notes:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = [note.Note(int(n)) for n in notes_in_chord]
            for n in notes:
                n.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)
    print(f"✅ MIDI 檔已儲存為：{output_file}")

# === Step 6: 主程式流程 ===
if __name__ == "__main__":
    midi_dir = "maestro-v3.0.0"

    print("📥 開始讀取 MIDI...", flush=True)
    notes = load_midi_notes(midi_dir)
    print(f"✅ 讀取完成，共 {len(notes)} 個音符", flush=True)

    print("🔁 準備序列資料...", flush=True)
    net_in, net_out, note_to_int, int_to_note = prepare_sequences(notes)
    print(f"✅ 序列資料準備完成，共 {len(net_in)} 筆訓練樣本", flush=True)

    print("🧠 建立 LSTM 模型...", flush=True)
    model = MusicLSTM(input_size=1, hidden_size=256, output_size=len(note_to_int))
    print("✅ 模型建立完成", flush=True)

    print("🚀 開始訓練模型...", flush=True)
    train(model, net_in, net_out, epochs=30)
    print("✅ 訓練完成", flush=True)

    print("🎼 生成音樂...", flush=True)
    generate_music(model, net_in, int_to_note, note_count=len(note_to_int))
    print("✅ 音樂生成完成", flush=True)

# === 📚 練習題目 ===
# 1. 調整 sequence_length（例如：20、50）觀察訓練樣本數變化
# 2. 嘗試不同 dropout 值（如 0.1、0.5）觀察 overfitting 狀況
# 3. 修改 hidden_size（如 128 或 512）觀察生成音樂品質
# 4. 將 output 改為 top-k sampling 而非 max，增加音樂多樣性
# 5. 嘗試加入 BatchNorm 或 GRU 替代 LSTM，觀察訓練收斂性