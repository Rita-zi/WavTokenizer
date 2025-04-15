import torch
from torch.utils.data import Dataset
import torchaudio
import os
import re
from pathlib import Path

class AudioDataset(Dataset):
    """音頻數據集類，處理輸入（不同材質）和目標（乾淨）音檔的配對"""
    
    def __init__(self, input_dirs, target_dir):
        """
        初始化數據集
        
        Args:
            input_dirs (list): 輸入音頻目錄列表，包含不同材質的音頻 (box/, plastic/, papercup/)
            target_dir (str): 目標音頻目錄，包含乾淨的音頻文件 (box2/)
        """
        self.input_dirs = input_dirs
        self.target_dir = target_dir
        self.paired_files = []
        
        # 建立輸入和目標音頻的配對
        self._build_pairs()
    
    def _build_pairs(self):
        """建立輸入和目標音頻的配對關係"""
        # 先獲取所有目標音頻文件
        target_files = {f.stem: f for f in Path(self.target_dir).glob('*.wav')}
        
        # 對每個輸入目錄
        for input_dir in self.input_dirs:
            material = os.path.basename(input_dir)  # 獲取材質名稱 (box/plastic/papercup)
            
            # 遍歷輸入目錄中的所有 wav 文件
            for input_file in Path(input_dir).glob('*.wav'):
                # 解析文件名以獲取說話者和序號信息
                match = re.match(r'nor_(\w+)_\w+_LDV_(\d+)\.wav', input_file.name)
                if not match:
                    continue
                    
                speaker, number = match.groups()
                
                # 構建對應的目標文件名（乾淨音頻）
                target_stem = f'nor_{speaker}_clean_{number.zfill(3)}'
                
                # 如果找到對應的目標文件
                if target_stem in target_files:
                    self.paired_files.append({
                        'input': str(input_file),
                        'target': str(target_files[target_stem]),
                        'speaker': speaker,
                        'material': material,
                        'number': number
                    })
    
    def __getitem__(self, idx):
        """獲取一對音頻數據"""
        pair = self.paired_files[idx]
        
        # 讀取音頻文件
        input_wav, sr_in = torchaudio.load(pair['input'])
        target_wav, sr_target = torchaudio.load(pair['target'])
        
        # 確保採樣率一致
        if sr_in != 24000:
            input_wav = torchaudio.transforms.Resample(sr_in, 24000)(input_wav)
        if sr_target != 24000:
            target_wav = torchaudio.transforms.Resample(sr_target, 24000)(target_wav)
        
        # 正規化
        input_wav = input_wav / (torch.max(torch.abs(input_wav)) + 1e-8)
        target_wav = target_wav / (torch.max(torch.abs(target_wav)) + 1e-8)
        
        return input_wav, target_wav
    
    def __len__(self):
        """返回數據集大小"""
        return len(self.paired_files)
    
    def get_metadata(self, idx):
        """獲取指定索引的元數據"""
        return self.paired_files[idx]