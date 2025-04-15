"""
WavTokenizer model implementation
Original core model code from tsne.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder.pretrained import WavTokenizer

class ResidualBlock(nn.Module):
    """殘差模塊，用於特徵增強"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class EnhancedFeatureExtractor(nn.Module):
    """特徵提取和增強模塊"""
    def __init__(self, config_path, model_path, num_residual_blocks=2):
        super().__init__()
        # 載入預訓練的WavTokenizer模型
        base_model = WavTokenizer.from_pretrained0802(config_path, model_path)
        self.encodec = base_model.feature_extractor.encodec
        encoder_dim = 512

        # 凍結encoder和decoder
        for param in self.encodec.encoder.parameters():
            param.requires_grad = False
        for param in self.encodec.decoder.parameters():
            param.requires_grad = False
            
        # 特徵增強層
        self.adapter_conv = nn.Conv1d(encoder_dim, 256, kernel_size=1)
        self.adapter_bn = nn.BatchNorm1d(256)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_residual_blocks)])
        self.out_conv = nn.Conv1d(256, encoder_dim, kernel_size=1)
        self.relu = nn.ReLU()
        
        # 確保特徵增強層為可訓練狀態
        for module in [self.adapter_conv, self.adapter_bn, self.residual_blocks, self.out_conv]:
            for param in module.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
            
        with torch.no_grad():
            features = self.encodec.encoder(x)
        
        features = self.adapter_conv(features)
        features = self.adapter_bn(features)
        features = self.relu(features)
        features = self.residual_blocks(features)
        features = self.out_conv(features)
        features = self.relu(features)
        
        return features

class EnhancedWavTokenizer(nn.Module):
    """主要模型類，組合特徵提取和增強"""
    def __init__(self, config_path, model_path):
        super().__init__()
        self.feature_extractor = EnhancedFeatureExtractor(config_path, model_path)
    
    def forward(self, x):
        enhanced_features = self.feature_extractor(x)
        output = self.feature_extractor.encodec.decoder(enhanced_features)
        input_features = self.feature_extractor.encodec.encoder(x)
        return output, input_features, enhanced_features

    @staticmethod
    def compute_feature_loss(enhanced_features, target_features):
        """計算特徵空間的損失"""
        # 正規化特徵
        enhanced_norm = F.normalize(enhanced_features, dim=1)
        target_norm = F.normalize(target_features, dim=1)
        
        # 計算餘弦相似度
        cos_sim = torch.bmm(enhanced_norm.transpose(1, 2), target_norm)
        
        # 計算 L2 距離
        l2_dist = torch.norm(enhanced_features - target_features, dim=1)
        
        return l2_dist.mean()  # 使用 L2 距離作為主要損失

    @staticmethod
    def compute_voice_focused_loss(output, target, device):
        """計算語音重建損失"""
        min_length = min(output.size(-1), target.size(-1))
        output = output[..., :min_length]
        target = target[..., :min_length]
        
        if output.dim() == 3:
            output = output.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        
        time_loss = F.l1_loss(output, target)
        
        def stft_loss(x, y, n_fft=2048):
            x_stft = torch.stft(x, n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
            y_stft = torch.stft(y, n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
            
            freqs = torch.linspace(0, 12000, x_stft.shape[1], device=device)
            voice_weights = torch.ones_like(freqs, device=device)
            voice_mask = ((freqs >= 80) & (freqs <= 3400)).float()
            voice_weights = voice_weights + voice_mask * 2.0
            
            mag_loss = torch.abs(torch.abs(x_stft) - torch.abs(y_stft))
            phase_loss = 1 - torch.cos(torch.angle(x_stft) - torch.angle(y_stft))
            weighted_loss = (mag_loss + 0.3 * phase_loss) * voice_weights.unsqueeze(-1)
            return torch.mean(weighted_loss)
        
        stft_loss_total = (
            stft_loss(output, target, n_fft=2048) +
            stft_loss(output, target, n_fft=1024) +
            stft_loss(output, target, n_fft=512)
        )
        
        return 0.3 * time_loss + 0.7 * stft_loss_total

    def compute_hybrid_loss(self, output, target_wav, enhanced_features, target_features, device):
        """計算混合損失"""
        feature_loss = self.compute_feature_loss(enhanced_features, target_features)
        total_loss = feature_loss
        
        return total_loss, {
            'feature_loss': feature_loss.item(),
            'voice_loss': 0.0
        }