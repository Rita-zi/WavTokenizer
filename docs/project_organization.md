# WavTokenizer 專案結構說明

## 核心文件結構

```
WavTokenizer/
├── src/
│   ├── models/
│   │   └── wavtokenizer/
│   │       ├── tsne_model.py      # 主要模型實現 (原 tsne.py)
│   │       └── dataset.py         # 數據集相關代碼 (從 try3.py 移出)
│   ├── utils/
│   │   └── visualization.py       # 視覺化工具 (從 add.py 移出)
│   └── tests/
│       └── test_model.py         # 測試代碼 (原 test.py)
├── config/
│   └── model_config/
│       └── wavtokenizer_config.yaml
├── data/
│   ├── raw/                      # 原始音頻數據
│   │   ├── box/
│   │   ├── plastic/
│   │   └── papercup/
│   └── processed/                # 處理後的數據
└── outputs/
    ├── enhanced/                 # 增強後的音頻
    ├── tsne/                     # TSNE 分析結果
    └── visualization/            # 可視化結果

## 保留的核心文件說明

1. tsne.py (移動到 src/models/wavtokenizer/tsne_model.py)
   - 主要的模型實現
   - TSNE 分析核心功能

2. test.py (移動到 src/tests/test_model.py)
   - 測試集程式碼
   - 用於驗證模型功能

3. try3.py
   - 包含重要的數據處理邏輯
   - 被 tsne.py 調用的關鍵組件
   - 主要功能移至 src/models/wavtokenizer/dataset.py

4. add.py (功能移至 src/utils/visualization.py)
   - 訓練過程中的 TSNE 圖形視覺化
   - 用於監控訓練進度

## 移除的非核心文件
以下文件將被移除，因為它們是臨時測試或實驗性質：
- try.py, try2.py, try4.py
- test2.py, test3.py
- 其他實驗性文件

## 數據目錄組織
- 所有原始音頻數據移至 data/raw/ 下對應的子目錄
- 處理後的數據統一存放在 data/processed/
- 訓練輸出和結果存放在 outputs/ 目錄下

## 配置文件管理
所有配置文件統一放在 config/model_config/ 目錄下：
- wavtokenizer_config.yaml - 主要配置文件
- 其他模型相關的配置文件