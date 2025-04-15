# WavTokenizer 專案結構

## 核心目錄

### src/
主要源代碼目錄
- `models/` - 模型相關代碼
  - `tsne_model.py` - TSNE 模型實現
  - `encodec/` - EnCodec 相關代碼
- `utils/` - 通用工具函數
- `tools/` - 輔助工具腳本

### config/
配置文件目錄
- `model_config/` - 模型配置文件
  - `wavtokenizer_config.yaml` - 主要配置文件

### data/
數據目錄
- `raw/` - 原始音頻文件
  - `box/` - 箱子材質音頻
  - `plastic/` - 塑膠材質音頻
  - `papercup/` - 紙杯材質音頻
- `processed/` - 處理後的數據

### outputs/
輸出結果目錄
- `enhanced/` - 增強後的音頻文件
- `tsne/` - TSNE 分析結果
- `visualization/` - 可視化結果

## 保留的核心文件
以下是專案中重要的核心文件：

1. 模型文件：
- `tsne_model.py` - 主要實現文件
- `decoder/` 和 `encoder/` - EnCodec 相關代碼

2. 配置文件：
- `wavtokenizer_config.yaml` - 主要配置文件

3. 模型權重：
- `wavtokenizer_large_speech_320_24k.ckpt`
- `wavtokenizer_medium_music_audio_320_24k_v2.ckpt`
- `byol_model_weights.pth`

## 移除的文件
為了保持專案整潔，以下類型的文件已被移除：

1. 測試文件：
- `test.py`, `test2.py`, `test3.py`
- `test_env.py`

2. 實驗文件：
- `try.py`, `try2.py`, `try3.py`, `try4.py`
- `tr_byol.py`, `tr_encodec.py`, `tr_w_t.py`

3. 其他輔助文件：
- `add.py`, `dd.py`, `finetune.py`
- `resume_training.py`, `resume_training1.py`

## 使用指南

1. 訓練模型：
```bash
python src/models/tsne_model.py --config config/model_config/wavtokenizer_config.yaml
```

2. 數據處理：
- 將原始音頻文件放入 `data/raw/` 對應的材質目錄
- 處理後的數據會自動保存到 `data/processed/`

3. 查看結果：
- 增強後的音頻文件在 `outputs/enhanced/`
- TSNE 分析結果在 `outputs/tsne/`
- 可視化圖表在 `outputs/visualization/`