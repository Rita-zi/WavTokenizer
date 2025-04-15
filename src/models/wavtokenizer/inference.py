"""
Inference script for WavTokenizer model
"""
import torch
import torchaudio
import os
from pathlib import Path
import yaml
from tqdm import tqdm

from .model import EnhancedWavTokenizer
from .utils.visualization import plot_spectrograms, save_audio_samples

def load_model(config_path, checkpoint_path, device):
    """載入模型"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 初始化模型
    model = EnhancedWavTokenizer(
        config['model']['config_path'],
        config['model']['model_path']
    ).to(device)
    
    # 載入訓練好的權重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def process_audio(model, input_path, output_dir, device):
    """處理單個音頻文件"""
    # 確保輸出目錄存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 讀取音頻
    wav, sr = torchaudio.load(input_path)
    
    # 重採樣到 24kHz（如果需要）
    if sr != 24000:
        wav = torchaudio.transforms.Resample(sr, 24000)(wav)
    
    # 正規化
    wav = wav / (torch.max(torch.abs(wav)) + 1e-8)
    
    # 移動到設備
    wav = wav.to(device)
    
    # 產生輸出文件名
    input_name = Path(input_path).stem
    output_path = output_dir / f"{input_name}_enhanced.wav"
    spec_path = output_dir / f"{input_name}_spectrograms.png"
    
    # 使用模型處理音頻
    with torch.no_grad():
        output, input_features, enhanced_features = model(wav)
        
        # 正規化輸出
        output = output / (torch.max(torch.abs(output)) + 1e-8)
        
        # 保存增強後的音頻
        torchaudio.save(output_path, output.cpu(), 24000)
        
        # 生成並保存頻譜圖
        plot_spectrograms(wav, spec_path, device, title="Input Spectrogram")
        plot_spectrograms(output, str(spec_path).replace('.png', '_enhanced.png'),
                         device, title="Enhanced Spectrogram")
    
    return output_path, spec_path

def batch_process(model, input_dir, output_dir, device):
    """批次處理目錄中的所有音頻文件"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 獲取所有 wav 文件
    wav_files = list(input_dir.glob('*.wav'))
    print(f"\nFound {len(wav_files)} wav files in {input_dir}")
    
    # 批次處理
    results = []
    for wav_file in tqdm(wav_files, desc="Processing audio files"):
        try:
            output_path, spec_path = process_audio(model, wav_file, output_dir, device)
            results.append({
                'input': str(wav_file),
                'output': str(output_path),
                'spectrogram': str(spec_path),
                'status': 'success'
            })
        except Exception as e:
            print(f"Error processing {wav_file}: {str(e)}")
            results.append({
                'input': str(wav_file),
                'status': 'failed',
                'error': str(e)
            })
    
    return results

def main():
    # 加載配置
    config_path = "config/model_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 載入模型
    checkpoint_path = os.path.join(config['output']['save_dir'], 'best_model.pth')
    model = load_model(config_path, checkpoint_path, device)
    print(f"Model loaded from: {checkpoint_path}")
    
    # 批次處理音頻
    results = batch_process(
        model,
        input_dir=config['data']['input_dirs'][0],  # 使用第一個輸入目錄
        output_dir=config['output']['enhanced_dir'],
        device=device
    )
    
    # 打印結果
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {success_count}/{len(results)} files")
    print(f"Enhanced audio saved to: {config['output']['enhanced_dir']}")

if __name__ == "__main__":
    main()