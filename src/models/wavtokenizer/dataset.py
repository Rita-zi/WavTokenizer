"""
Dataset handling for WavTokenizer
Original implementation from try3.py, reorganized for better structure
"""
import torch
from torch.utils.data import Dataset
import torchaudio
import os
from pathlib import Path
import re

class AudioDataset(Dataset):
    """音頻數據集類，處理輸入（不同材質）和目標（乾淨）音檔的配對"""
    
    def __init__(self, input_dirs, target_dir, sampling_rate=24000):
        """
        初始化數據集
        
        Args:
            input_dirs (list): 輸入音頻目錄列表，包含不同材質的音頻
            target_dir (str): 目標音頻目錄，包含乾淨的音頻文件
            sampling_rate (int): 目標採樣率
        """
        self.input_dirs = input_dirs
        self.target_dir = target_dir
        self.sampling_rate = sampling_rate
        self.paired_files = []
        
        # 建立輸入和目標音頻的配對
        self._build_pairs()
        
        # 打印數據集信息
        self._print_dataset_info()
    
    def _build_pairs(self):
        """建立輸入和目標音頻的配對關係"""
        # 先獲取所有目標音頻文件
        target_files = {
            Path(f).stem: f 
            for f in Path(self.target_dir).glob('*.wav')
        }
        
        # 對每個輸入目錄
        for input_dir in self.input_dirs:
            material = Path(input_dir).name  # 獲取材質名稱
            
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
    
    def _print_dataset_info(self):
        """打印數據集信息"""
        print(f"\nDataset Information:")
        print(f"Total pairs: {len(self.paired_files)}")
        
        # 統計材質分布
        material_counts = {}
        speaker_counts = {}
        
        for pair in self.paired_files:
            material = pair['material']
            speaker = pair['speaker']
            
            material_counts[material] = material_counts.get(material, 0) + 1
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        print("\nMaterial distribution:")
        for material, count in material_counts.items():
            print(f"  - {material}: {count}")
            
        print("\nSpeaker distribution:")
        for speaker, count in speaker_counts.items():
            print(f"  - {speaker}: {count}")
    
    def __getitem__(self, idx):
        """獲取一對音頻數據"""
        pair = self.paired_files[idx]
        
        # 讀取音頻文件
        input_wav, sr_in = torchaudio.load(pair['input'])
        target_wav, sr_target = torchaudio.load(pair['target'])
        
        # 確保採樣率一致
        if sr_in != self.sampling_rate:
            input_wav = torchaudio.transforms.Resample(sr_in, self.sampling_rate)(input_wav)
        if sr_target != self.sampling_rate:
            target_wav = torchaudio.transforms.Resample(sr_target, self.sampling_rate)(target_wav)
        
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

def create_data_loaders(dataset, batch_size, num_workers=4, pin_memory=True, validation_split=0.2):
    """創建訓練和驗證數據加載器"""
    from torch.utils.data import DataLoader, random_split
    
    # 計算訓練集和驗證集的大小
    total_size = len(dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    
    # 分割數據集
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 創建數據加載器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def collate_fn(batch):
    """處理批次數據"""
    input_wavs = [item[0] for item in batch]
    target_wavs = [item[1] for item in batch]
    
    # 找出最短的音訊長度
    min_len = min(
        min(wav.size(-1) for wav in input_wavs),
        min(wav.size(-1) for wav in target_wavs)
    )
    
    # 對齊長度
    input_wavs = [wav[..., :min_len] for wav in input_wavs]
    target_wavs = [wav[..., :min_len] for wav in target_wavs]
    
    return torch.stack(input_wavs), torch.stack(target_wavs)