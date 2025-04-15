"""
Visualization utilities for WavTokenizer training monitoring
Original implementation from add.py
"""
import matplotlib.pyplot as plt
import torch
import torchaudio
import os
import numpy as np
from pathlib import Path

def plot_spectrograms(audio, save_path, device, title="Spectrogram"):
    """繪製並保存頻譜圖"""
    try:
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=4096,
            hop_length=512,
            win_length=4096,
            n_mels=128
        ).to(device)
        
        amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)
        
        with torch.no_grad():
            spec = transform(audio)
            spec_db = amplitude_to_db(spec)
            spec_db = spec_db.cpu()
        
        plt.figure(figsize=(10, 4))
        plt.imshow(spec_db.squeeze().numpy(), cmap='viridis', origin='lower', aspect='auto')
        plt.title(title)
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    except Exception as e:
        print(f"Error in plot_spectrograms: {str(e)}")
        print(f"Audio device: {audio.device}")
        print(f"Audio shape: {audio.shape}")

def plot_training_curves(training_history, save_path):
    """繪製訓練過程的各種指標曲線"""
    plt.figure(figsize=(12, 8))
    
    # 創建兩個Y軸
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # 繪製各種損失曲線
    lines = []
    if 'train_losses' in training_history:
        line1 = ax1.plot(training_history['epochs'], 
                        training_history['train_losses'], 
                        'b-', label='Train Loss')
        lines.extend(line1)
        
    if 'val_losses' in training_history:
        line2 = ax1.plot(training_history['epochs'], 
                        training_history['val_losses'], 
                        'g-', label='Val Loss')
        lines.extend(line2)
        
    if 'feature_losses' in training_history:
        line3 = ax1.plot(training_history['epochs'], 
                        training_history['feature_losses'], 
                        'c-', label='Feature Loss')
        lines.extend(line3)
        
    if 'voice_losses' in training_history:
        line4 = ax1.plot(training_history['epochs'], 
                        training_history['voice_losses'], 
                        'm-', label='Voice Loss')
        lines.extend(line4)
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 繪製學習率曲線
    if 'lr_values' in training_history:
        line5 = ax2.plot(training_history['epochs'], 
                        training_history['lr_values'], 
                        'r-', label='Learning Rate')
        lines.extend(line5)
        ax2.set_ylabel('Learning Rate', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
    
    # 合併圖例
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels)
    
    plt.title('Training Metrics Over Time')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_audio_samples(input_wav, output_wav, target_wav, save_dir, prefix="sample"):
    """保存音頻樣本和對應的頻譜圖"""
    device = input_wav.device
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (input_audio, output_audio, target_audio) in enumerate(
        zip(input_wav, output_wav, target_wav)):
            
        # 正規化音頻
        input_audio = input_audio / (torch.max(torch.abs(input_audio)) + 1e-8)
        output_audio = output_audio / (torch.max(torch.abs(output_audio)) + 1e-8)
        target_audio = target_audio / (torch.max(torch.abs(target_audio)) + 1e-8)
        
        # 重塑形狀
        input_audio = input_audio.reshape(1, -1)
        output_audio = output_audio.reshape(1, -1)
        target_audio = target_audio.reshape(1, -1)
        
        # 基礎檔名
        base_name = f"{prefix}_{i+1}"
        
        # 保存音頻和頻譜圖
        for audio, name in [
            (input_audio, 'input'),
            (output_audio, 'output'),
            (target_audio, 'target')
        ]:
            # 保存音頻
            audio_path = save_dir / f"{base_name}_{name}.wav"
            torchaudio.save(audio_path, audio.cpu(), 24000)
            
            # 保存頻譜圖
            spec_path = save_dir / f"{base_name}_{name}_spec.png"
            plot_spectrograms(
                audio.to(device),
                spec_path,
                device,
                title=f'{name.capitalize()} Spectrogram'
            )

def monitor_gpu_memory():
    """監控 GPU 記憶體使用情況"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()/1024**2
        cached = torch.cuda.memory_reserved()/1024**2
        return allocated, cached
    return 0, 0