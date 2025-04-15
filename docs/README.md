# WavTokenizer Project

## Project Structure

### Data Organization
```
data/
├── input/                 # 輸入音檔（不同材質）
│   ├── box/              # 箱子材質的音檔
│   ├── plastic/          # 塑膠材質的音檔
│   └── papercup/         # 紙杯材質的音檔
└── target/               # 目標音檔
    └── clean/            # 乾淨的目標音檔 (原 box2/)
```

### Model Files
- `tsne.py`: 主要模型實現，用於音頻特徵提取和增強
- `config/model_config/`: 模型配置文件
- `checkpoints/`: 模型檢查點保存目錄

### Training Data Flow
1. 輸入：從 data/input/ 讀取不同材質的音檔
2. 目標：使用 data/target/clean/ 中的乾淨音檔作為訓練目標
3. 輸出：增強後的音頻保存在 outputs/ 目錄

### Key Components
1. **輸入處理**
   - 支援多種材質的音頻輸入
   - 自動進行音頻正規化和預處理

2. **特徵提取**
   - 使用 WavTokenizer 進行特徵編碼
   - 增強層處理編碼特徵

3. **音頻重建**
   - 將增強後的特徵解碼為音頻
   - 保持音頻品質的同時去除材質影響

### Training Process
1. 數據加載：從 input/ 和 target/clean/ 加載配對的音頻
2. 特徵提取：使用 WavTokenizer 提取音頻特徵
3. 特徵增強：通過增強層處理特徵
4. 重建：將增強的特徵重建為音頻
5. 評估：對比重建音頻與目標音頻的品質