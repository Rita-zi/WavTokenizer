#更新try3
import os
import torch
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from decoder.pretrained import WavTokenizer
from encoder.utils import convert_audio
import numpy as np
from tqdm import tqdm
import random
import torch.nn.functional as F
import torch.nn as nn
import traceback
import datetime

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class EnhancedFeatureExtractor(nn.Module):
    def __init__(self, config_path, model_path, num_residual_blocks=2):
        super(EnhancedFeatureExtractor, self).__init__()
        # Load base model
        base_model = WavTokenizer.from_pretrained0802(config_path, model_path)
        self.encodec = base_model.feature_extractor.encodec

        # Get encoder output dimension from the model configuration
        encoder_dim = 512  # This is the default dimension for the model

        # Only freeze encoder parameters, keep decoder trainable
        for param in self.encodec.encoder.parameters():
            param.requires_grad = False
            
        # Ensure decoder parameters are trainable
        for param in self.encodec.decoder.parameters():
            param.requires_grad = True
        
        self.adapter_conv = nn.Conv1d(encoder_dim, 256, kernel_size=1)  
        self.adapter_bn = nn.BatchNorm1d(256)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_residual_blocks)])
        self.out_conv = nn.Conv1d(256, encoder_dim, kernel_size=1)  
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Ensure input requires grad
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
            
        # Get encoder features
        with torch.no_grad():
            features = self.encodec.encoder(x)
            
        # Process features through our trainable layers
        features = self.relu(self.adapter_bn(self.adapter_conv(features)))
        features = self.residual_blocks(features)
        features = self.out_conv(features)
        
        return features

class EnhancedModel(nn.Module):
    def __init__(self, config_path, model_path):
        super(EnhancedModel, self).__init__()
        self.feature_extractor = EnhancedFeatureExtractor(config_path, model_path)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.feature_extractor.encodec.decoder(features)
        return output

def pad_or_trim(wav, target_length):
    """將音頻填充或裁剪到指定長度"""
    current_length = wav.size(-1)
    
    if (current_length == target_length):
        return wav
    
    if (current_length > target_length):
        # 如果太長，從中間截取所需長度
        start = (current_length - target_length) // 2
        return wav[..., start:start + target_length]
    
    # 如果太短，進行填充
    padding_length = target_length - current_length
    left_pad = padding_length // 2
    right_pad = padding_length - left_pad
    return torch.nn.functional.pad(wav, (left_pad, right_pad))

class AudioAugment:
    def __init__(self):
        self.augments = [
            self.add_noise,
            self.time_stretch,
            self.pitch_shift
        ]
        self.pitch_range = (-2, 2)
    
    def add_noise(self, wav, noise_level=0.005):
        noise = torch.randn_like(wav) * noise_level
        return wav + noise
    
    def time_stretch(self, wav, factor_range=(0.95, 1.05)):
        """Time stretch using librosa"""
        try:
            factor = random.uniform(*factor_range)
            # Convert to numpy for librosa processing
            wav_np = wav.squeeze().numpy()
            
            # Apply time stretching
            stretched = librosa.effects.time_stretch(y=wav_np, rate=factor)
            
            # Convert back to torch and ensure correct shape
            return torch.from_numpy(stretched).float().unsqueeze(0)
        except Exception as e:
            print(f"Error in time_stretch: {str(e)}")
            return wav

    def pitch_shift(self, wav, sr=24000):
        """Pitch shift using librosa"""
        try:
            n_steps = random.uniform(*self.pitch_range)
            # Convert to numpy for librosa processing
            wav_np = wav.squeeze().numpy()
            
            # Apply pitch shifting
            shifted = librosa.effects.pitch_shift(
                y=wav_np,
                sr=sr,
                n_steps=n_steps
            )
            
            # Convert back to torch and ensure correct shape
            return torch.from_numpy(shifted).float().unsqueeze(0)
        except Exception as e:
            print(f"Error in pitch_shift: {str(e)}")
            return wav

    def __call__(self, wav):
        """Apply random augmentation with error handling"""
        try:
            aug = random.choice(self.augments)
            augmented = aug(wav)
            
            # Ensure output has correct shape and type
            if augmented.dim() == 1:
                augmented = augmented.unsqueeze(0)
            if augmented.dtype != torch.float32:
                augmented = augmented.float()
                
            return augmented
        except Exception as e:
            print(f"Error in augmentation: {str(e)}")
            return wav

def process_audio(audio_path, target_sr=24000, normalize=True):
    """音頻處理函數，負責讀取、轉換採樣率和正規化音頻"""
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, target_sr, 1)  # [1, T]

    
    if normalize:
        wav = wav / (wav.abs().max() + 1e-8)
    return wav  # 保持原始長度 [1, T]

class AudioDataset(Dataset):
    """
    自定義音頻資料集類別，負責載入輸入和目標音頻文件。
    """
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.paired_files = []
        
        # 檢查目錄是否存在
        if (not os.path.exists(input_dir) or not os.path.exists(target_dir)):
            print(f"Error: Directory not found!")
            print(f"Input directory exists: {os.path.exists(input_dir)}")
            print(f"Target directory exists: {os.path.exists(target_dir)}")
            return

        print(f"\nSearching for files in:")
        print(f"Input directory: {input_dir}")
        print(f"Target directory: {target_dir}")
        
        # 獲取並打印所有文件列表
        input_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
        target_files = [f for f in os.listdir(target_dir) if f.endswith('.wav')]
        
        print("\nInput files found:")
        for f in input_files:
            print(f"  {f}")
        
        print("\nTarget files found:")
        for f in target_files:
            print(f"  {f}")
        
        # 遍歷輸入目錄
        for input_file in input_files:
            try:
                # 提取語者ID和檔案編號
                parts = input_file.split('_')
                if len(parts) >= 2:  # 放寬條件，只要有兩個部分就行
                    speaker_id = parts[1]    # 例如: boy1
                    file_num = parts[-1]     # 取最後一部分作為檔案編號
                    
                    print(f"\nTrying to match: {input_file}")
                    print(f"Looking for files with speaker '{speaker_id}' and number '{file_num}'")
                    
                    # 尋找對應的target檔案
                    matched_targets = [
                        t for t in target_files 
                        if speaker_id in t and file_num in t
                    ]
                    
                    if matched_targets:
                        target_file = matched_targets[0]
                        print(f"Found match: {input_file} -> {target_file}")
                        self.paired_files.append({
                            'input': input_file,
                            'target': target_file
                        })
                    else:
                        print(f"No match found for {input_file}")
            
            except Exception as e:
                print(f"Error processing {input_file}: {str(e)}")
                continue
        
        # 打印最終結果
        print(f"\nTotal paired files: {len(self.paired_files)}")
        if self.paired_files:
            print("\nSuccessfully paired files:")
            for pair in self.paired_files:
                print(f"Input:  {pair['input']}")
                print(f"Target: {pair['target']}")
        else:
            print("\nWARNING: No valid file pairs found!")
            print("1. Check that your files exist in both directories")
            print("2. Check file naming format")
            print("3. Make sure speaker IDs match exactly")
            print("4. Make sure file numbers match exactly")
        
    def __len__(self):
        return len(self.paired_files)
        
    def __getitem__(self, idx):
        pair = self.paired_files[idx]
        input_path = os.path.join(self.input_dir, pair['input'])
        target_path = os.path.join(self.target_dir, pair['target'])
        
        # 讀取音頻，不指定固定長度
        input_wav = process_audio(input_path, normalize=True)
        target_wav = process_audio(target_path, normalize=False)
        
        return input_wav, target_wav


def normalize_audio(audio, target_range=(-1, 1)):
    """音頻正規化函數"""
    if audio.abs().max() > 0:
        # 保持音頻的正負特性
        max_val = audio.abs().max()
        min_val = -max_val
        # 將範圍映射到目標範圍
        audio_norm = (audio - min_val) / (max_val - min_val)
        audio_norm = audio_norm * (target_range[1] - target_range[0]) + target_range[0]
        return audio_norm
    return audio

def plot_spectrograms(wav, save_path, device, title=None):
    """繪製頻譜圖"""
    try:
        wav_numpy = wav.cpu().numpy().squeeze()
        D = librosa.stft(wav_numpy, 
            n_fft=4096,        
            hop_length=512,     
            win_length=4096    
        )
        D = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(
            D,
            y_axis='log',       # Y軸: 頻率 (Hz), 對數刻度
            x_axis='time',      # X軸: 時間 (秒)
            sr=24000,           # 採樣率
            hop_length=512,
            fmin=20,            # 最小頻率: 20Hz
            fmax=20000          # 最大頻率: 20kHz
        )
        
        # 添加軸標籤
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        
        # 添加標題和顏色條
        if title:
            plt.title(title)
        plt.colorbar(format='%+2.0f dB')
        
        # 保存並關閉
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error in plot_spectrograms: {str(e)}")

def save_sample(input_wav, output, target_wav, epoch, batch_idx, save_dir, device):
    """保存音頻樣本和頻譜圖"""
    try:
        audio_dir = os.path.join(save_dir, f'epoch_{epoch+1}_samples')
        os.makedirs(audio_dir, exist_ok=True)
        
        for j in range(output.size(0)):
            try:
                # 確保所有張量都先detach並移至CPU，並正規化到[-1,1]
                with torch.no_grad():
                    output_audio = normalize_audio(output[j].detach().cpu()).reshape(1, -1)
                    input_audio = normalize_audio(input_wav[j].detach().cpu()).reshape(1, -1)
                    target_audio = normalize_audio(target_wav[j].detach().cpu()).reshape(1, -1)
                
                # 創建基礎檔名
                base_name = f"batch_{batch_idx}_sample_{j+1}"
                
                # 保存音頻
                for audio, prefix in [
                    (output_audio, 'output'),
                    (input_audio, 'input'),
                    (target_audio, 'target')
                ]:
                    # 音頻文件路徑
                    audio_path = os.path.join(audio_dir, f'{base_name}_{prefix}.wav')
                    spec_path = os.path.join(audio_dir, f'{base_name}_{prefix}_spec.png')
                    
                    # 保存音頻
                    torchaudio.save(audio_path, audio, 24000)
                    
                    # 保存頻譜圖
                    plot_spectrograms(
                        audio,
                        spec_path,
                        device,
                        title=f'Epoch {epoch+1} {prefix.capitalize()} Spectrogram'
                    )
                    
                print(f"Saved sample {j+1} from batch {batch_idx}")
                
            except Exception as e:
                print(f"Error saving sample {j+1} from batch {batch_idx}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error in save_sample function: {str(e)}")

def info_nce_loss(features_1, features_2, temperature=0.07):
    """改進的對比損失函數"""
    device = features_1.device
    
    # 確保特徵啟用梯度
    features_1 = features_1.detach().requires_grad_(True)
    features_2 = features_2.detach().requires_grad_(True)
    
    # 統一特徵維度
    if features_1.dim() == 3:  # [B, C, T]
        features_1 = features_1.permute(0, 2, 1)  # [B, T, C]
        features_1 = features_1.reshape(-1, features_1.size(-1))  # [B*T, C]
    
    if features_2.dim() == 3:  # [B, C, T]
        features_2 = features_2.permute(0, 2, 1)  # [B, T, C]
        features_2 = features_2.reshape(-1, features_2.size(-1))  # [B*T, C]
    
    # 正規化特徵
    features_1 = torch.nn.functional.normalize(features_1, dim=-1)
    features_2 = torch.nn.functional.normalize(features_2, dim=-1)
    
    batch_size = features_1.size(0)
    
    # 合併特徵並計算相似度矩陣
    features = torch.cat([features_1, features_2], dim=0)  # [2*B, C]
    similarity = torch.matmul(features, features.t()) / temperature  # [2*B, 2*B]
    
    # 生成標籤
    labels = torch.arange(batch_size, device=device)
    labels = torch.cat([labels + batch_size, labels])  # [2*B]
    
    # 計算交叉熵損失
    loss = torch.nn.CrossEntropyLoss()(similarity, labels)
    
    return loss

def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    """Save model checkpoint with error handling and best model tracking"""
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # Save current epoch checkpoint
        epoch_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)
        
        # Read current best loss if exists
        best_path = os.path.join(save_dir, 'best_model.pth')
        current_best_loss = float('inf')
        if os.path.exists(best_path):
            try:
                current_best_loss = torch.load(best_path)['loss']
            except:
                pass
        
        # Update best model if current loss is better
        if loss < current_best_loss:
            torch.save(checkpoint, best_path)
            print(f"\n=== New Best Model ===")
            print(f"Previous best loss: {current_best_loss:.6f}")
            print(f"New best loss: {loss:.6f}")
            print(f"Improvement: {current_best_loss - loss:.6f}")
        
        # Log the saves
        print(f"\n=== Checkpoint Saved ===")
        print(f"Epoch: {epoch}")
        print(f"Loss: {loss:.6f}")
        print(f"Epoch checkpoint: {epoch_path}")
        print(f"Best model path: {best_path}")
        
        return True
        
    except Exception as e:
        print(f"\n=== Error Saving Checkpoint ===")
        print(f"Error: {str(e)}")
        print(f"Stack trace:\n{traceback.format_exc()}")
        return False

def load_pretrained_weights(model, encodec_path=None, byol_path=None):
    """加載預訓練權重"""
    # 加載原始WavTokenizer權重
    if encodec_path and os.path.exists(encodec_path):
        encodec_state = torch.load(encodec_path, map_location='cpu')
        model.feature_extractor.encodec.load_state_dict(encodec_state, strict=False)
        
    # 加載BYOL權重
    if byol_path and os.path.exists(byol_path):
        byol_state = torch.load(byol_path, map_location='cpu')
        if 'state_dict' in byol_state:
            byol_state = byol_state['state_dict']
        # 加載到特徵提取器
        model.feature_extractor.load_state_dict(byol_state, strict=False)
    
    return model

def collate_fn(batch):
    """自定義 collate 函數"""
    input_wavs = [item[0] for item in batch]  # Keep channel dimension
    target_wavs = [item[1] for item in batch]
    
    # 找出最長的音訊長度
    max_len = max(max(wav.size(-1) for wav in input_wavs),
                 max(wav.size(-1) for wav in target_wavs))
    
    # 將所有音訊補齊到相同長度
    padded_inputs = []
    padded_targets = []
    
    for input_wav, target_wav in zip(input_wavs, target_wavs):
        if input_wav.size(-1) < max_len:
            input_wav = torch.nn.functional.pad(input_wav, (0, max_len - input_wav.size(-1)))
        if target_wav.size(-1) < max_len:
            target_wav = torch.nn.functional.pad(target_wav, (0, max_len - target_wav.size(-1)))
        padded_inputs.append(input_wav)
        padded_targets.append(target_wav)
    
    # Stack到batch
    return (torch.stack(padded_inputs),
            torch.stack(padded_targets))

def perceptual_loss(output, target, device, sample_rate=24000):
    """Enhanced perceptual loss using MelSpectrogram"""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=80,
        f_min=20,
        f_max=8000
    ).to(device)
    
    # Compute mel spectrograms
    mel_output = mel_transform(output)
    mel_target = mel_transform(target)
    
    # Log-scale mel spectrograms
    mel_output = torch.log(mel_output + 1e-5)
    mel_target = torch.log(mel_target + 1e-5)
    
    return torch.nn.functional.mse_loss(mel_output, mel_target)
    
def compute_hybrid_loss(output, target, device):
    """結合多種損失函數的混合損失"""
    
    # 確保輸入張量維度正確
    if output.dim() == 3:
        output = output.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
        
    # 1. 基礎時域損失
    l1_loss = torch.nn.L1Loss()(output, target)
    
    # 2. 頻段分離損失
    def frequency_selective_loss(x, y, n_fft=4096, hop_length=512):
        # 確保輸入是2D張量 [B, T]
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        
        x_stft = torch.stft(x, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        y_stft = torch.stft(y, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        
        # 分離頻段
        freq_bins = x_stft.shape[1]
        low_freq = int(freq_bins * 0.2)
        mid_freq = int(freq_bins * 0.4)
        
        # 計算各頻段損失
        low_loss = torch.mean(torch.abs(x_stft[:, :low_freq] - y_stft[:, :low_freq]))
        mid_loss = torch.mean(torch.abs(x_stft[:, low_freq:mid_freq] - y_stft[:, low_freq:mid_freq]))
        high_loss = torch.mean(torch.abs(x_stft[:, mid_freq:] - y_stft[:, mid_freq:]))
        
        return 0.5 * low_loss + 0.3 * mid_loss + 0.2 * high_loss

    # 3. 相位敏感損失
    def phase_loss(x, y, n_fft=4096):
        # 確保輸入是2D張量 [B, T]
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        
        x_stft = torch.stft(x, n_fft=n_fft, hop_length=512, return_complex=True)
        y_stft = torch.stft(y, n_fft=n_fft, hop_length=512, return_complex=True)
        return torch.mean(1 - torch.cos(torch.angle(x_stft) - torch.angle(y_stft)))

    # 計算各個損失分量
    freq_loss = frequency_selective_loss(output, target)
    phase = phase_loss(output, target)
    
    # 組合損失
    total_loss = (
        0.4 * l1_loss + 
        0.4 * freq_loss + 
        0.2 * phase
    )
    
    return total_loss, {
        'l1': l1_loss.item(),
        'freq': freq_loss.item(),
        'phase': phase.item()
    }

def compute_mel_contrast_loss(output, target, device):
    """計算梅爾頻譜對比度損失"""
    def get_mel_spectrogram(audio):
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=2048,
            hop_length=512,
            n_mels=80,
            power=2
        ).to(device)
        return torch.log10(torch.clamp(mel_spec(audio), min=1e-5))

    # 計算梅爾頻譜
    output_mel = get_mel_spectrogram(output)
    target_mel = get_mel_spectrogram(target)

    # 計算能量遮罩
    energy_mask = (target_mel < target_mel.mean()).float()
    
    # 計算低能量區域的損失
    low_energy_loss = torch.mean(torch.abs(output_mel * energy_mask))
    
    # 計算對比度損失
    contrast_loss = torch.mean(torch.abs(
        torch.sigmoid(output_mel) - torch.sigmoid(target_mel)
    ))

    return low_energy_loss + contrast_loss, {
        'low_energy': low_energy_loss.item(),
        'contrast': contrast_loss.item()
    }

def compute_enhanced_loss(output, target, device):

    
    def stft_loss(x, y, n_fft, weights=None):
        x_stft = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
        y_stft = torch.stft(y.squeeze(1), n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
        
        # 計算幅度和相位損失
        mag_loss = torch.abs(torch.abs(x_stft) - torch.abs(y_stft))
        phase_loss = 1 - torch.cos(torch.angle(x_stft) - torch.angle(y_stft))
        
        if weights is not None:
            mag_loss = mag_loss * weights.unsqueeze(-1)
            phase_loss = phase_loss * weights.unsqueeze(-1)
            
        return torch.mean(mag_loss + 0.5 * phase_loss)

    # 高頻增強權重
    freq_weights = torch.linspace(1.0, 3.0, 2049).to(device)  # 高頻權重更大
    
    # 多尺度STFT損失
    loss_1 = stft_loss(output, target, n_fft=4096, weights=freq_weights)
    loss_2 = stft_loss(output, target, n_fft=2048, weights=freq_weights[:1025])
    loss_3 = stft_loss(output, target, n_fft=1024, weights=freq_weights[:513])
    
    # 時域L1損失
    time_loss = torch.nn.L1Loss()(output, target)
    
    return 0.4 * time_loss + 0.6 * (loss_1 + loss_2 + loss_3)

def compute_optimized_loss(output, target, device):
    """優化的損失函數，結合效率和質量"""
    
    # 1. 快速時域損失
    time_loss = F.l1_loss(output, target)
    
    # 2. 高效頻譜損失
    def quick_stft_loss(x, y, n_fft=2048):
        x_stft = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=n_fft//4, 
                           return_complex=True)
        y_stft = torch.stft(y.squeeze(1), n_fft=n_fft, hop_length=n_fft//4, 
                           return_complex=True)
        
        # 計算頻率權重 (著重高頻)
        freq_weights = torch.linspace(1.0, 2.0, x_stft.shape[1], device=device)
        
        # 計算magnitude和phase損失
        mag_loss = torch.abs(torch.abs(x_stft) - torch.abs(y_stft))
        phase_loss = 1 - torch.cos(torch.angle(x_stft) - torch.angle(y_stft))
        
        # 應用頻率權重
        weighted_loss = (mag_loss + 0.5 * phase_loss) * freq_weights.unsqueeze(-1)
        return torch.mean(weighted_loss)

    # 3. 計算頻譜損失
    spec_loss = quick_stft_loss(output, target)
    
    # 4. 組合損失 (加重高頻部分)
    total_loss = 0.3 * time_loss + 0.7 * spec_loss
    
    return total_loss

def compute_optimized_loss_v2(output, target, device, lambda_perceptual=0.1):
    """優化的損失函數"""
    # 確保輸入維度正確
    if output.dim() == 3:
        output = output.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
        
    # 1. 時域損失
    time_loss = F.l1_loss(output, target)
    
    # 2. STFT損失（高頻加權）
    def spectral_detail_loss(x, y, n_fft=2048):
        x_stft = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
        y_stft = torch.stft(y.squeeze(1), n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
        
        # 計算頻率權重
        freq_weights = torch.linspace(1.0, 2.0, x_stft.shape[1], device=device)
        
        # 細節損失
        detail_loss = torch.abs(torch.abs(x_stft) - torch.abs(y_stft))
        weighted_detail = detail_loss * freq_weights.unsqueeze(-1)
        return torch.mean(weighted_detail)
    
    # 3. 對比度損失 (try3的特點)
    def contrast_loss(x, y, n_fft=4096):
        x_stft = torch.stft(x, n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
        y_stft = torch.stft(y, n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
        
        # 能量對比
        x_energy = torch.abs(x_stft)
        y_energy = torch.abs(y_stft)
        
        # 強化高低能量差異
        energy_diff = torch.abs(x_energy - y_energy)
        energy_mask = (y_energy > y_energy.mean()).float()
        contrast = energy_diff * energy_mask
        
        return torch.mean(contrast)
    
    detail = spectral_detail_loss(output, target)
    contrast = contrast_loss(output, target)
    percep = perceptual_loss(output, target, device)
    
    # 組合損失
    return (0.2 * time_loss + 
            0.4 * detail +     # 保持細節
            0.2 * contrast +   # 增強對比
            0.2 * percep)      # 保持感知質量

def stft_loss_fn(x, y, n_fft, weights=None):
    """改進的STFT損失"""
    x_stft = torch.stft(x, n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
    y_stft = torch.stft(y, n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
    
    mag_x = torch.abs(x_stft)
    mag_y = torch.abs(y_stft)
    
    phase_x = torch.angle(x_stft)
    phase_y = torch.angle(y_stft)
    
    mag_loss = torch.abs(mag_x - mag_y)
    phase_loss = 1 - torch.cos(phase_x - phase_y)
    
    if weights is not None:
        mag_loss = mag_loss * weights.unsqueeze(-1)
        phase_loss = phase_loss * weights.unsqueeze(-1)
    
    return torch.mean(mag_loss + 0.5 * phase_loss)

def evaluate_audio_quality(output, target, device):
    """評估音訊質量"""
    metrics = {}
    
    # 1. SNR (Signal-to-Noise Ratio)
    signal_power = torch.mean(target ** 2)
    noise_power = torch.mean((target - output) ** 2)
    metrics['snr'] = 10 * torch.log10(signal_power / (noise_power + 1e-6))
    
    # 2. PESQ-like metric (simplified)
    with torch.no_grad():
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=2048,
            hop_length=512,
            n_mels=80
        ).to(device)
        
        mel_output = torch.log(mel_transform(output) + 1e-5)
        mel_target = torch.log(mel_transform(target) + 1e-5)
        metrics['spectral_distance'] = F.mse_loss(mel_output, mel_target)
    
    # 3. Envelope similarity
    output_env = torch.abs(output)
    target_env = torch.abs(target)
    metrics['env_similarity'] = F.cosine_similarity(output_env, target_env, dim=-1).mean()
    
    return metrics

def compute_hybrid_enhanced_loss(output, target, device):
    """增強型混合損失函數"""
    
    # 1. 時域損失
    time_loss = F.l1_loss(output, target)
    
    # 2. 頻譜細節損失
    def spectral_detail_loss(x, y, n_fft=2048):
        x_stft = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
        y_stft = torch.stft(y.squeeze(1), n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
        
        # 頻率權重
        freq_weights = torch.linspace(1.0, 2.0, x_stft.shape[1], device=device)
        
        # 細節損失
        detail_loss = torch.abs(torch.abs(x_stft) - torch.abs(y_stft))
        weighted_detail = detail_loss * freq_weights.unsqueeze(-1)
        return torch.mean(weighted_detail)
    
    # 3. 能量對比損失
    def energy_contrast_loss(x, y, n_fft=4096):
        x_stft = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
        y_stft = torch.stft(y.squeeze(1), n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
        
        # 計算能量
        x_power = torch.abs(x_stft) ** 2
        y_power = torch.abs(y_stft) ** 2
        
        # 轉換為分貝
        x_db = 10 * torch.log10(x_power + 1e-8)
        y_db = 10 * torch.log10(y_power + 1e-8)
        
        # 計算高低能量區域遮罩
        high_energy_mask = (y_db > y_db.mean()).float()
        low_energy_mask = (y_db <= y_db.mean()).float()
        
        # 分別計算高低能量區域的損失
        high_energy_loss = torch.mean(torch.abs(x_db - y_db) * high_energy_mask) * 1.5
        low_energy_loss = torch.mean(torch.abs(x_db - y_db) * low_energy_mask) * 0.5
        
        return high_energy_loss + low_energy_loss
    
    # 4. 感知損失
    percep_loss = perceptual_loss(output, target, device)
    
    # 計算各個損失
    detail = spectral_detail_loss(output, target)
    energy = energy_contrast_loss(output, target)
    
    # 組合損失
    total_loss = (
        0.2 * time_loss +      # 基礎重建
        0.3 * detail +         # 頻譜細節
        0.3 * energy +         # 能量對比
        0.2 * percep_loss      # 感知質量
    )
    
    return total_loss

def compute_voice_focused_loss(output, target, device):
    """專注於人聲特徵的損失函數"""
    
    # 1. 基礎時域損失 (from try2)
    time_loss = F.l1_loss(output, target)
    
    # 2. 人聲頻段加權的STFT損失
    def voice_weighted_stft_loss(x, y, n_fft=2048):
        x_stft = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
        y_stft = torch.stft(y.squeeze(1), n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
        
        # 計算頻率範圍 (關注人聲頻段 80Hz-3400Hz)
        freqs = torch.linspace(0, 12000, x_stft.shape[1], device=device)  # 24000/2 = 12000Hz
        voice_weights = torch.ones_like(freqs, device=device)
        voice_mask = ((freqs >= 80) & (freqs <= 3400)).float()  #3400
        voice_weights = voice_weights + voice_mask * 2.0  # 人聲頻段權重加倍
        
        # 計算幅度和相位損失
        mag_loss = torch.abs(torch.abs(x_stft) - torch.abs(y_stft))
        phase_loss = 1 - torch.cos(torch.angle(x_stft) - torch.angle(y_stft))
        
        # 應用權重
        weighted_loss = (mag_loss + 0.3 * phase_loss) * voice_weights.unsqueeze(-1)
        return torch.mean(weighted_loss)
    
    # 3. 多尺度損失，著重人聲頻段
    stft_loss = (
        voice_weighted_stft_loss(output, target, n_fft=2048) + 
        voice_weighted_stft_loss(output, target, n_fft=1024) + 
        voice_weighted_stft_loss(output, target, n_fft=512)
    )
    
    # 4. 組合損失
    total_loss = 0.3 * time_loss + 0.7 * stft_loss
    
    return total_loss

def train_model(model, train_loader, optimizer, scheduler, device, save_dir, config):
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        valid_batches = 0
        
        # 使用tqdm來顯示進度
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                          desc=f'Epoch {epoch+1}/{config["epochs"]}')
        
        for batch_idx, (input_wav, target_wav) in progress_bar:
            try:
                # Move to device and ensure shapes
                input_wav = input_wav.to(device)
                target_wav = target_wav.to(device)
                
                # Normalize inputs
                input_wav = (input_wav - input_wav.mean()) / (input_wav.std() + 1e-6)
                target_wav = (target_wav - target_wav.mean()) / (target_wav.std() + 1e-6)
                
                optimizer.zero_grad()
                
                try:
                    # Forward pass
                    features = model.feature_extractor(input_wav)
                    
                    # 特徵增強，參考try2的方法
                    features = features * 1.5  # 增加特徵增強強度
                    features = torch.tanh(features)  # 限制範圍
                    
                    output = model.feature_extractor.encodec.decoder(features)
                    
                    # Ensure shapes match
                    min_length = min(output.size(-1), target_wav.size(-1))
                    output = output[..., :min_length]
                    target_wav = target_wav[..., :min_length]
                    
                    # Remove channel dimension for loss computation
                    output = output.squeeze(1)
                    target_wav = target_wav.squeeze(1)
                    
                    # Compute loss
                    loss = compute_voice_focused_loss(output, target_wav, device)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                    optimizer.step()
                    
                    # Update progress
                    total_loss += loss.item()
                    valid_batches += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.6f}',
                        'avg_loss': f'{(total_loss/valid_batches):.6f}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
                    
                    # Save samples periodically
                    if batch_idx % 50 == 0:
                        save_sample(
                            input_wav.unsqueeze(1),
                            output.unsqueeze(1),
                            target_wav.unsqueeze(1),
                            epoch, batch_idx, save_dir, device
                        )
                        
                except Exception as e:
                    print(f"Error in forward pass: {str(e)}")
                    continue
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        # Epoch summary
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Average Loss: {avg_loss:.6f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(model, optimizer, epoch + 1, best_loss, save_dir)
                print(f"New best model saved! Loss: {best_loss:.6f}")

# 在main中更新配置
def main():
    config = {
        'epochs': 1000,
        'batch_size': 4,
        'learning_rate': 2e-4,    # 提高學習率
        'weight_decay': 0.001,    # 降低權重衰減
        'scheduler_patience': 3,
        'scheduler_factor': 0.7,
        'grad_clip': 0.5,
        'min_lr': 1e-6,
        'feature_scale': 1.5      # 增加特徵增強強度
    }
    # ...existing code...

def verify_wavtokenizer_structure(model, device):
    """驗證WavTokenizer架構的完整性"""
    print("\n=== Verifying WavTokenizer Structure ===")
    
    # 1. 檢查基本組件
    checks = {
        'encodec': hasattr(model.feature_extractor, 'encodec'),
        'encoder': hasattr(model.feature_extractor.encodec, 'encoder'),
        'decoder': hasattr(model.feature_extractor.encodec, 'decoder')
    }
    
    for name, exists in checks.items():
        print(f"✓ {name}: {'Present' if exists else 'Missing'}")
    
    # 2. 驗證前向傳播
    try:
        test_input = torch.randn(1, 1, 24000).to(device)  # 1秒音訊
        with torch.no_grad():
            features = model.feature_extractor(test_input)
            output = model.feature_extractor.encodec.decoder(features)
            print(f"✓ Feature extraction shape: {features.shape}")
            print(f"✓ Output shape: {output.shape}")
            print(f"✓ Feature range: [{features.min():.3f}, {features.max():.3f}]")
            print(f"✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
    except Exception as e:
        print(f"✗ Forward pass failed: {str(e)}")
        return False
    
    # 3. 檢查參數是否可訓練
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # 4. 驗證模型模式
    print(f"✓ Training mode: {model.training}")
    
    return all(checks.values())

def verify_checkpoint_loading(model, checkpoint_path):
    """驗證預訓練權重載入"""
    try:
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict['model_state_dict'])
            print(f"✓ Successfully loaded checkpoint: {checkpoint_path}")
            return True
    except Exception as e:
        print(f"✗ Checkpoint loading failed: {str(e)}")
    return False

def main():
    config = {
        'epochs': 1000,
        'batch_size': 8,         # 增加批次大小
        'learning_rate': 1e-3,    # 提高學習率
        'weight_decay': 0.001,   # 降低權重衰減
        'scheduler_patience': 2,   # 降低容忍度
        'scheduler_factor': 0.7,   # 更激進的lr衰減
        'grad_clip': 0.5,
        'min_lr': 1e-6,
        'lambda_perceptual': 0.1,
        'encoder_dim': 512,
        'accumulation_steps': 2    # 添加梯度累積
    }
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    config_path = "./wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    model_path = "./wavtokenizer_large_speech_320_24k.ckpt"
    
    # Initialize model with verification
    print(f"\nInitializing model from config: {config_path}")
    try:
        model = EnhancedModel(config_path, model_path).to(device)
        
        # Verify model structure before training
        if not verify_wavtokenizer_structure(model, device):
            print("ERROR: Model structure verification failed!")
            return
        
        model.train()
        print("\nModel initialization successful!")
        
    except Exception as e:
        print(f"ERROR: Model initialization failed: {str(e)}")
        print(f"Stack trace:", traceback.format_exc())
        return
    
    # Dataset & DataLoader
    dataset = AudioDataset("./box", "./box2")
    train_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Create scheduler with correct parameters
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',              # 監視損失下降
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience'],
        min_lr=config['min_lr'],
        verbose=True
    )
    
    # Training
    save_dir = './self_supervised_output'
    train_model(model, train_loader, optimizer, scheduler, device, save_dir, config)

if __name__ == "__main__":
    main()