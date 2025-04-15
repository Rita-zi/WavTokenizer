"""
Main training script for WavTokenizer
"""
import os
import torch
from pathlib import Path
import yaml
import random
import numpy as np
from torch.utils.data import DataLoader

from .model import EnhancedWavTokenizer
from .trainer import WavTokenizerTrainer
from .dataset import AudioDataset, collate_fn

def set_seed(seed=42):
    """固定隨機種子以確保可重現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")

def worker_init_fn(worker_id):
    """初始化數據加載器工作進程的隨機種子"""
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_gpu():
    """設置 GPU 環境"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        
        print("\nGPU Information:")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name()}")
        print(f"Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    else:
        device = torch.device('cpu')
        print("\nUsing CPU")
    
    return device

def main():
    # 加載配置
    config_path = Path("config/model_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 設置環境
    set_seed(42)
    device = setup_gpu()
    
    # 初始化模型
    model = EnhancedWavTokenizer(
        config['model']['config_path'],
        config['model']['model_path']
    ).to(device)
    
    # 載入數據集
    dataset = AudioDataset(
        input_dirs=config['data']['input_dirs'],
        target_dir=config['data']['clean_dir']
    )
    
    # 分割訓練集和驗證集
    from torch.utils.data import random_split
    val_size = int(len(dataset) * config['training']['validation']['split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 創建數據加載器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        prefetch_factor=config['data']['prefetch_factor'],
        collate_fn=collate_fn,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        prefetch_factor=config['data']['prefetch_factor'],
        collate_fn=collate_fn,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )
    
    # 初始化優化器
    optimizer = torch.optim.AdamW(
        model.feature_extractor.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 初始化學習率調度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['training']['scheduler']['T_0'],
        T_mult=config['training']['scheduler']['T_mult'],
        eta_min=config['training']['scheduler']['eta_min']
    )
    
    # 初始化訓練器
    trainer = WavTokenizerTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    # 開始訓練
    print(f"\nStarting training for {config['training']['epochs']} epochs")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Initial learning rate: {config['training']['learning_rate']}")
    print(f"Saving outputs to: {config['output']['save_dir']}")
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=config['output']['save_dir'],
        num_epochs=config['training']['epochs']
    )
    
    print("\nTraining completed!")
    print(f"Best loss: {trainer.best_loss:.6f}")
    print(f"Model saved to: {config['output']['save_dir']}")

if __name__ == "__main__":
    main()