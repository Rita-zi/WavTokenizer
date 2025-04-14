import torch
import torchaudio
import yaml
import os
import numpy as np
from tqdm import tqdm
from decoder.pretrained import WavTokenizer
from pathlib import Path
import argparse
import json

def load_config(config_path):
    """載入配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def init_model(config_path, model_path, device):
    """初始化模型"""
    model = WavTokenizer.from_pretrained0802(config_path, model_path)
    model.eval()
    model.to(device)
    return model

def process_audio(audio_path, device):
    """處理音頻文件"""
    wav, sr = torchaudio.load(audio_path)
    if sr != 24000:
        resampler = torchaudio.transforms.Resample(sr, 24000)
        wav = resampler(wav)
    wav = wav / (torch.max(torch.abs(wav)) + 1e-8)
    return wav.to(device)

def extract_features(model, wav):
    """提取特徵向量"""
    with torch.no_grad():
        features = model.feature_extractor.encodec.encoder(wav)
    return features

def save_features(features, metadata, save_path):
    """保存特徵向量和元數據"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if save_path.endswith('.pt'):
        torch.save({
            'features': features,
            'metadata': metadata
        }, save_path)
    else:  # .npy
        np.save(save_path, features.cpu().numpy())
        metadata_path = save_path.replace('.npy', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Extract feature vectors from audio files using WavTokenizer')
    parser.add_argument('--config', type=str, default='config/extract_config.yaml',
                      help='Path to configuration file')
    args = parser.parse_args()

    # 載入配置
    config = load_config(args.config)
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化模型
    model = init_model(
        config['model']['config_path'],
        config['model']['model_path'],
        device
    )
    
    # 創建輸出目錄
    output_dir = Path(config['output']['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有音頻文件
    audio_files = []
    for input_dir in config['data']['input_dirs']:
        audio_files.extend(list(Path(input_dir).glob('*.wav')))

    # 批次處理音頻文件
    batch_size = config['data']['batch_size']
    save_format = config['output']['save_format']
    
    for i in tqdm(range(0, len(audio_files), batch_size)):
        batch_files = audio_files[i:i + batch_size]
        batch_wavs = []
        batch_metadata = []
        
        # 處理每個文件
        for audio_file in batch_files:
            try:
                wav = process_audio(str(audio_file), device)
                batch_wavs.append(wav)
                
                # 收集元數據
                metadata = {
                    'file_path': str(audio_file),
                    'original_dir': str(audio_file.parent.name),
                    'file_name': audio_file.name
                }
                batch_metadata.append(metadata)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                continue
        
        if not batch_wavs:
            continue
            
        # 合併批次
        batch_tensor = torch.cat(batch_wavs, dim=0)
        
        # 提取特徵
        try:
            features = extract_features(model, batch_tensor)
            
            # 單獨保存每個文件的特徵
            for j, (feature, metadata) in enumerate(zip(features, batch_metadata)):
                file_name = Path(metadata['file_name']).stem
                save_path = output_dir / f"{file_name}.{save_format}"
                save_features(
                    feature.unsqueeze(0),  # 添加批次維度
                    metadata,
                    str(save_path)
                )
                
        except Exception as e:
            print(f"Error extracting features for batch {i}: {str(e)}")
            continue

    print(f"\nFeature extraction completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()