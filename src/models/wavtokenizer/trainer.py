"""
Training logic for WavTokenizer model
"""
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from .utils.visualization import plot_spectrograms, save_audio_samples, plot_learning_curves

class WavTokenizerTrainer:
    def __init__(self, model, optimizer, scheduler=None, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_loss = float('inf')
        
    def train(self, train_loader, val_loader, save_dir, num_epochs=100):
        """訓練模型並保存檢查點"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = save_dir / 'best_model.pth'
        
        # 記錄訓練指標
        history = {
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
            'feature_losses': [],
            'voice_losses': [],
            'lr_values': []
        }
        
        # 載入已有的最佳模型（如果存在）
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.best_loss = checkpoint['loss']
            print(f"\nLoaded previous best loss: {self.best_loss}")
        
        # 訓練循環
        for epoch in range(num_epochs):
            # 訓練階段
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # 驗證階段
            if val_loader:
                val_metrics = self._validate(val_loader)
            else:
                val_metrics = {'loss': None}
            
            # 更新學習率
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(train_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # 更新歷史記錄
            current_lr = self.optimizer.param_groups[0]['lr']
            history['epochs'].append(epoch + 1)
            history['train_losses'].append(train_metrics['loss'])
            history['val_losses'].append(val_metrics['loss'])
            history['feature_losses'].append(train_metrics['feature_loss'])
            history['voice_losses'].append(train_metrics['voice_loss'])
            history['lr_values'].append(current_lr)
            
            # 保存最佳模型
            if train_metrics['loss'] < self.best_loss:
                self.best_loss = train_metrics['loss']
                self._save_checkpoint(best_model_path, epoch, history)
                
            # 定期保存檢查點和可視化結果
            if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
                # 保存檢查點
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pt'
                self._save_checkpoint(checkpoint_path, epoch, history)
                
                # 繪製學習曲線
                curve_path = save_dir / f'learning_curve_epoch_{epoch+1}.png'
                plot_learning_curves(
                    history['epochs'],
                    history['train_losses'],
                    history['val_losses'],
                    history['feature_losses'],
                    history['voice_losses'],
                    history['lr_values'],
                    curve_path
                )
                
                # 保存音頻樣本
                if (epoch + 1) % 300 == 0 or epoch == num_epochs - 1:
                    self._save_audio_samples(train_loader, epoch, save_dir)
        
        return history
    
    def _train_epoch(self, train_loader, epoch):
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0.0
        total_feature_loss = 0.0
        total_voice_loss = 0.0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f'Epoch {epoch+1}')
        
        for batch_idx, (input_wav, target_wav) in progress_bar:
            self.optimizer.zero_grad()
            
            # 移動數據到設備並正規化
            input_wav = input_wav.to(self.device)
            target_wav = target_wav.to(self.device)
            input_wav = input_wav / (torch.max(torch.abs(input_wav)) + 1e-8)
            target_wav = target_wav / (torch.max(torch.abs(target_wav)) + 1e-8)
            
            # 前向傳播
            output, input_features, enhanced_features = self.model(input_wav)
            with torch.no_grad():
                target_features = self.model.feature_extractor.encodec.encoder(target_wav)
            
            # 計算損失
            loss, loss_details = self.model.compute_hybrid_loss(
                output, target_wav, enhanced_features, target_features, self.device)
            
            # 反向傳播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_feature_loss += loss_details["feature_loss"]
            total_voice_loss += loss_details["voice_loss"]
            
            # 更新進度條
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'feat': f'{loss_details["feature_loss"]:.4f}',
                'voice': f'{loss_details["voice_loss"]:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # 計算平均損失
        avg_loss = total_loss / len(train_loader)
        avg_feature_loss = total_feature_loss / len(train_loader)
        avg_voice_loss = total_voice_loss / len(train_loader)
        
        return {
            'loss': avg_loss,
            'feature_loss': avg_feature_loss,
            'voice_loss': avg_voice_loss
        }
    
    def _validate(self, val_loader):
        """驗證模型"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for input_wav, target_wav in val_loader:
                input_wav = input_wav.to(self.device)
                target_wav = target_wav.to(self.device)
                input_wav = input_wav / (torch.max(torch.abs(input_wav)) + 1e-8)
                target_wav = target_wav / (torch.max(torch.abs(target_wav)) + 1e-8)
                
                output, input_features, enhanced_features = self.model(input_wav)
                target_features = self.model.feature_extractor.encodec.encoder(target_wav)
                
                loss, _ = self.model.compute_hybrid_loss(
                    output, target_wav, enhanced_features, target_features, self.device)
                
                total_loss += loss.item()
        
        return {'loss': total_loss / len(val_loader)}
    
    def _save_checkpoint(self, path, epoch, history):
        """保存檢查點"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_loss,
            'history': history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, path)
        print(f"\nCheckpoint saved to: {path}")
    
    def _save_audio_samples(self, loader, epoch, save_dir):
        """保存音頻樣本"""
        self.model.eval()
        audio_dir = save_dir / f'epoch_{epoch+1}_samples'
        audio_dir.mkdir(exist_ok=True)
        
        with torch.no_grad():
            for batch_idx, (input_wav, target_wav) in enumerate(loader):
                if batch_idx >= 2:  # 只保存前兩個批次
                    break
                    
                input_wav = input_wav.to(self.device)
                target_wav = target_wav.to(self.device)
                output, _, _ = self.model(input_wav)
                
                save_audio_samples(
                    input_wav, output, target_wav,
                    audio_dir, f"batch_{batch_idx}"
                )