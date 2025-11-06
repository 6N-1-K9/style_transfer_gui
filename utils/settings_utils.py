import os
import json
import torch
from datetime import datetime

class TrainingSettings:
    """Класс для сохранения и загрузки настроек обучения"""
    
    def __init__(self):
        self.timestamp = None
        self.training_config = {}
        self.paths = {}
        self.model_info = {}
    
    def save(self, file_path, trainer):
        """Сохраняет настройки обучения в файл"""
        settings = {
            'timestamp': datetime.now().isoformat(),
            'training_config': {
                'image_size': trainer.image_size,
                'batch_size': trainer.batch_size,
                'epochs': trainer.epochs,
                'lr': trainer.lr,
                'lambda_cycle': trainer.lambda_cycle,
                'lambda_identity': trainer.lambda_identity,
                'n_residual_blocks': trainer.n_residual_blocks,
                'use_dropout': trainer.use_dropout,
                'gradient_clip': trainer.gradient_clip,
                'lr_decay_start': trainer.lr_decay_start,
                'lr_decay_end': trainer.lr_decay_end,
                'final_lr_ratio': trainer.final_lr_ratio,
                'use_early_stopping': trainer.use_early_stopping,
                'early_stopping_patience': trainer.early_stopping_patience,
                'save_interval': trainer.save_interval,
            },
            'paths': {
                'dataset_a': trainer.dataset_a_path,
                'dataset_b': trainer.dataset_b_path,
                'models_dir': trainer.models_dir,
                'stats_dir': trainer.stats_dir,
            },
            'model_info': {
                'current_epoch': getattr(trainer, 'current_epoch', 0),
                'total_epochs': trainer.epochs,
                'last_saved_epoch': getattr(trainer, 'last_saved_epoch', 0),
            }
        }
        
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения настроек: {e}")
            return False
    
    def load(self, file_path):
        """Загружает настройки из файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            self.timestamp = settings['timestamp']
            self.training_config = settings['training_config']
            self.paths = settings['paths']
            self.model_info = settings.get('model_info', {})
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки настроек: {e}")
            return False

def find_latest_checkpoint(stats_dir):
    """Находит последний чекпоинт в папке статистики"""
    checkpoints_dir = os.path.join(stats_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoints_dir, "checkpoint_latest.pth")
    
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    
    return None

def find_latest_model(models_dir):
    """Находит последнюю сохраненную модель"""
    g_a2b_dir = os.path.join(models_dir, "G_A2B")
    if os.path.exists(g_a2b_dir):
        model_files = [f for f in os.listdir(g_a2b_dir) if f.endswith('.pth')]
        if model_files:
            # Сортируем по номеру эпохи
            model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_model = model_files[-1]
            return os.path.join(g_a2b_dir, latest_model)
    
    return None

def validate_resume_files(settings_path, checkpoint_path):
    """Проверяет валидность файлов для продолжения обучения"""
    errors = []
    
    # Проверяем файл настроек
    if not os.path.exists(settings_path):
        errors.append("Файл настроек не найден")
    else:
        # Проверяем валидность настроек
        settings = TrainingSettings()
        if not settings.load(settings_path):
            errors.append("Невалидный файл настроек")
    
    # Проверяем чекпоинт или модель
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        errors.append("Чекпоинт или модель не найдены")
    
    return len(errors) == 0, errors

def get_settings_path(models_dir):
    """Возвращает путь для сохранения настроек"""
    return os.path.join(models_dir, "training_settings.json")

def find_latest_checkpoint(stats_dir):
    """Находит последний чекпоинт в папке статистики"""
    checkpoints_dir = os.path.join(stats_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoints_dir, "checkpoint_latest.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"✅ Найден чекпоинт: {checkpoint_path}")
        return checkpoint_path
    else:
        print(f"❌ Чекпоинт не найден: {checkpoint_path}")
        return None

def find_latest_model(models_dir):
    """Находит последнюю сохраненную модель"""
    g_a2b_dir = os.path.join(models_dir, "G_A2B")
    if os.path.exists(g_a2b_dir):
        model_files = [f for f in os.listdir(g_a2b_dir) if f.endswith('.pth')]
        if model_files:
            # Сортируем по номеру эпохи
            model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_model = model_files[-1]
            latest_path = os.path.join(g_a2b_dir, latest_model)
            print(f"✅ Найдена модель: {latest_path}")
            return latest_path
        else:
            print(f"❌ Модели не найдены в: {g_a2b_dir}")
    else:
        print(f"❌ Папка моделей не существует: {g_a2b_dir}")
    
    return None