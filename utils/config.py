import torch

class TrainingConfig:
    def __init__(self):
        # Основные настройки
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = 256
        self.batch_size = 2
        self.epochs = 200
        
        # Гиперпараметры
        self.lr = 0.0002
        self.lambda_cycle = 10
        self.lambda_identity = 5.0
        
        # Архитектура
        self.n_residual_blocks = 9
        self.use_dropout = True
        self.gradient_clip = 1.0
        
        # Планировщик
        self.lr_decay_start = 50
        self.lr_decay_end = 150
        self.final_lr_ratio = 0.1
        
        # Дополнительные
        self.use_early_stopping = True
        self.early_stopping_patience = 20
        self.save_interval = 10

class InferenceConfig:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = 256