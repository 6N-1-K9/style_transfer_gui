import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
import numpy as np
from tqdm import tqdm
import json
import csv
from datetime import datetime
from .models import StyleGenerator, Discriminator

class FlatImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = self._get_image_paths()
        
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {folder_path}")
    
    def _get_image_paths(self):
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(self.folder_path, ext)))
            image_paths.extend(glob.glob(os.path.join(self.folder_path, ext.upper())))
        return image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class StyleTransferTrainer:
    def __init__(self, dataset_a_path, dataset_b_path, models_dir, stats_dir,
                 image_size=256, batch_size=2, epochs=200, lr=0.0002,
                 lambda_cycle=10, lambda_identity=5.0, n_residual_blocks=9,
                 use_dropout=True, gradient_clip=1.0, lr_decay_start=50,
                 lr_decay_end=150, final_lr_ratio=0.1, use_early_stopping=True,
                 early_stopping_patience=20, save_interval=10,
                 log_callback=None, progress_callback=None):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_a_path = dataset_a_path
        self.dataset_b_path = dataset_b_path
        self.models_dir = models_dir
        self.stats_dir = stats_dir
        
        # Сохраняем ВСЕ настройки
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.n_residual_blocks = n_residual_blocks
        self.use_dropout = use_dropout
        self.gradient_clip = gradient_clip
        self.lr_decay_start = lr_decay_start
        self.lr_decay_end = lr_decay_end
        self.final_lr_ratio = final_lr_ratio
        self.use_early_stopping = use_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.save_interval = save_interval
        
        self.log_callback = log_callback or print
        self.progress_callback = progress_callback
        self.stop_training = False
        
        # Логируем реальные настройки
        self._log("⚙️  ПРИМЕНЕННЫЕ НАСТРОЙКИ:")
        self._log(f"   Размер изображения: {self.image_size}px")
        self._log(f"   Batch size: {self.batch_size}")
        self._log(f"   Эпохи: {self.epochs}")
        self._log(f"   Learning Rate: {self.lr}")
        self._log(f"   Lambda Cycle: {self.lambda_cycle}")
        self._log(f"   Lambda Identity: {self.lambda_identity}")
        self._log(f"   Residual Blocks: {self.n_residual_blocks}")
        self._log(f"   Устройство: {self.device}")
        
        self._setup_directories()
        self._setup_models()
        self._setup_data()
        self._setup_training()
        self._setup_statistics()
    
    def _setup_directories(self):
        """Создает необходимые директории"""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, "G_A2B"), exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, "G_B2A"), exist_ok=True)
    
    def _setup_statistics(self):
        """Настраивает систему сбора статистики"""
        # Создаем подпапки для статистики
        self.losses_dir = os.path.join(self.stats_dir, "losses")
        self.metrics_dir = os.path.join(self.stats_dir, "metrics")
        self.checkpoints_dir = os.path.join(self.stats_dir, "checkpoints")
        self.visual_dir = os.path.join(self.stats_dir, "visual_progress")
        
        os.makedirs(self.losses_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.visual_dir, exist_ok=True)
        
        # Инициализируем историю статистики
        self.statistics = {
            'epochs': [],
            'losses': {
                'G': [], 'D_A': [], 'D_B': [],
                'G_GAN_A2B': [], 'G_GAN_B2A': [],
                'G_cycle_ABA': [], 'G_cycle_BAB': [],
                'G_identity_A': [], 'G_identity_B': []
            },
            'timing': {
                'epoch_times': [],
                'batch_times': [],
                'total_time': 0
            },
            'data_info': {
                'dataset_A_size': len(self.dataset_A),
                'dataset_B_size': len(self.dataset_B),
                'image_size': self.image_size,
                'batch_size': self.batch_size
            }
        }
        
        # Сохраняем конфигурацию обучения
        self._save_training_config()
    
    def _save_training_config(self):
        """Сохраняет конфигурацию обучения"""
        config = {
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.lr,
            'lambda_cycle': self.lambda_cycle,
            'lambda_identity': self.lambda_identity,
            'n_residual_blocks': self.n_residual_blocks,
            'use_dropout': self.use_dropout,
            'gradient_clip': self.gradient_clip,
            'device': str(self.device),
            'dataset_A_path': self.dataset_a_path,
            'dataset_B_path': self.dataset_b_path,
            'dataset_A_size': len(self.dataset_A),
            'dataset_B_size': len(self.dataset_B),
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = os.path.join(self.stats_dir, "training_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self._log(f"💾 Конфигурация обучения сохранена: {config_path}")
    
    def _log(self, message):
        """Логирует сообщение"""
        self.log_callback(message)
    
    def _setup_models(self):
        """Инициализирует модели"""
        self._log("🔄 Инициализация моделей...")
        
        self.G_A2B = StyleGenerator(n_residual_blocks=self.n_residual_blocks).to(self.device)
        self.G_B2A = StyleGenerator(n_residual_blocks=self.n_residual_blocks).to(self.device)
        self.D_A = Discriminator().to(self.device)
        self.D_B = Discriminator().to(self.device)
        
        # Инициализация весов
        def weights_init_normal(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        
        self.G_A2B.apply(weights_init_normal)
        self.G_B2A.apply(weights_init_normal)
        self.D_A.apply(weights_init_normal)
        self.D_B.apply(weights_init_normal)
        
        self._log("✅ Модели инициализированы")
    
    def _setup_data(self):
        """Настраивает данные и трансформации"""
        self._log("🔄 Загрузка данных...")
        
        # Трансформации с ПРАВИЛЬНЫМ размером изображения
        style_transform = transforms.Compose([
            transforms.Resize(self.image_size + 50),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        content_transform = transforms.Compose([
            transforms.Resize(self.image_size + 30),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Датасеты
        self.dataset_A = FlatImageDataset(self.dataset_a_path, transform=content_transform)
        self.dataset_B = FlatImageDataset(self.dataset_b_path, transform=style_transform)
        
        self.dataloader_A = DataLoader(self.dataset_A, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.dataloader_B = DataLoader(self.dataset_B, batch_size=self.batch_size, shuffle=True, num_workers=0)
        
        self._log(f"✅ Загружено {len(self.dataset_A)} изображений A и {len(self.dataset_B)} изображений B")
        self._log(f"✅ Batch size: {self.batch_size}, Размер: {self.image_size}px")
    
    def _setup_training(self):
        """Настраивает обучение"""
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        self.optimizer_G = optim.Adam(
            list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )
        self.optimizer_D_A = optim.Adam(self.D_A.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D_B = optim.Adam(self.D_B.parameters(), lr=self.lr, betas=(0.5, 0.999))
    
    def train(self):
        """Запускает обучение"""
        self._log("🚀 Начало обучения...")
        self._log(f"📊 Всего батчей за эпоху: {min(len(self.dataloader_A), len(self.dataloader_B))}")
        
        start_time = datetime.now()
        
        for epoch in range(1, self.epochs + 1):
            if self.stop_training:
                self._log("⏹️ Обучение остановлено пользователем")
                break
            
            epoch_start = datetime.now()
            epoch_losses = self._train_epoch(epoch)
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            
            self._update_statistics(epoch, epoch_losses, epoch_time)
            self._save_epoch_statistics(epoch)
            
            # Сохранение моделей
            if epoch % self.save_interval == 0 or epoch == self.epochs:
                self._save_models(epoch)
            
            # Колбэк прогресса (ВЫЗЫВАЕМ ПОСЛЕ КАЖДОЙ ЭПОХИ)
            if self.progress_callback:
                total_loss = epoch_losses['G']
                self.progress_callback(epoch, self.epochs, total_loss)
        
        total_time = (datetime.now() - start_time).total_seconds()
        self.statistics['timing']['total_time'] = total_time
        self._save_final_statistics()
        
        self._log("✅ Обучение завершено!")
        self._log(f"⏱️  Общее время обучения: {total_time:.2f} секунд")
    
    def _train_epoch(self, epoch):
        """Одна эпоха обучения"""
        epoch_losses = {
            'G': 0, 'D_A': 0, 'D_B': 0,
            'G_GAN_A2B': 0, 'G_GAN_B2A': 0,
            'G_cycle_ABA': 0, 'G_cycle_BAB': 0,
            'G_identity_A': 0, 'G_identity_B': 0
        }
        
        min_length = min(len(self.dataloader_A), len(self.dataloader_B))
        
        # Используем tqdm для прогресс-бара в консоли
        progress_bar = tqdm(zip(self.dataloader_A, self.dataloader_B), 
                           total=min_length, desc=f"Epoch {epoch}/{self.epochs}", 
                           leave=False)
        
        for i, (real_A, real_B) in enumerate(progress_bar):
            if self.stop_training:
                break
            
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
            
            # Обучение генераторов
            batch_losses = self._train_generators(real_A, real_B)
            
            # Обучение дискриминаторов
            batch_losses.update(self._train_discriminators(real_A, real_B))
            
            # Обновляем статистику эпохи
            for key in epoch_losses:
                epoch_losses[key] += batch_losses[key]
            
            # Обновляем прогресс-бар
            progress_bar.set_postfix({
                'G_loss': f"{batch_losses['G']:.4f}",
                'D_loss': f"{(batch_losses['D_A'] + batch_losses['D_B'])/2:.4f}"
            })
        
        progress_bar.close()
        
        # Усредняем по батчам
        for key in epoch_losses:
            epoch_losses[key] /= min_length
        
        self._log(f"📈 Эпоха {epoch}: G_loss={epoch_losses['G']:.4f}, "
                 f"D_A_loss={epoch_losses['D_A']:.4f}, D_B_loss={epoch_losses['D_B']:.4f}")
        
        return epoch_losses
    
    def _train_generators(self, real_A, real_B):
        """Обучение генераторов"""
        self.optimizer_G.zero_grad()
        
        # Identity loss
        same_B = self.G_A2B(real_B)
        loss_identity_B = self.criterion_identity(same_B, real_B) * self.lambda_identity
        
        same_A = self.G_B2A(real_A)
        loss_identity_A = self.criterion_identity(same_A, real_A) * self.lambda_identity
        
        # GAN loss
        fake_B = self.G_A2B(real_A)
        pred_fake = self.D_B(fake_B)
        loss_GAN_A2B = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        
        fake_A = self.G_B2A(real_B)
        pred_fake = self.D_A(fake_A)
        loss_GAN_B2A = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        
        # Cycle loss
        recovered_A = self.G_B2A(fake_B)
        loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A) * self.lambda_cycle
        
        recovered_B = self.G_A2B(fake_A)
        loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B) * self.lambda_cycle
        
        # Total loss
        loss_G = (loss_GAN_A2B + loss_GAN_B2A + 
                 loss_cycle_ABA + loss_cycle_BAB + 
                 loss_identity_A + loss_identity_B)
        
        loss_G.backward()
        
        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()), self.gradient_clip)
        
        self.optimizer_G.step()
        
        return {
            'G': loss_G.item(),
            'G_GAN_A2B': loss_GAN_A2B.item(),
            'G_GAN_B2A': loss_GAN_B2A.item(),
            'G_cycle_ABA': loss_cycle_ABA.item(),
            'G_cycle_BAB': loss_cycle_BAB.item(),
            'G_identity_A': loss_identity_A.item(),
            'G_identity_B': loss_identity_B.item()
        }
    
    def _train_discriminators(self, real_A, real_B):
        """Обучение дискриминаторов"""
        losses = {}
        
        # Discriminator A
        self.optimizer_D_A.zero_grad()
        
        pred_real = self.D_A(real_A)
        loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
        
        fake_A = self.G_B2A(real_B)
        pred_fake = self.D_A(fake_A.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()
        self.optimizer_D_A.step()
        losses['D_A'] = loss_D_A.item()
        
        # Discriminator B
        self.optimizer_D_B.zero_grad()
        
        pred_real = self.D_B(real_B)
        loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
        
        fake_B = self.G_A2B(real_A)
        pred_fake = self.D_B(fake_B.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()
        self.optimizer_D_B.step()
        losses['D_B'] = loss_D_B.item()
        
        return losses
    
    def _update_statistics(self, epoch, epoch_losses, epoch_time):
        """Обновляет статистику обучения"""
        self.statistics['epochs'].append(epoch)
        self.statistics['timing']['epoch_times'].append(epoch_time)
        
        for key, value in epoch_losses.items():
            self.statistics['losses'][key].append(value)
    
    def _save_epoch_statistics(self, epoch):
        """Сохраняет статистику эпохи"""
        # Сохраняем losses в CSV
        losses_csv_path = os.path.join(self.losses_dir, "losses_history.csv")
        file_exists = os.path.isfile(losses_csv_path)
        
        with open(losses_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'G_loss', 'D_A_loss', 'D_B_loss', 
                               'G_GAN_A2B', 'G_GAN_B2A', 'G_cycle_ABA', 'G_cycle_BAB',
                               'G_identity_A', 'G_identity_B', 'epoch_time'])
            
            writer.writerow([
                epoch,
                self.statistics['losses']['G'][-1],
                self.statistics['losses']['D_A'][-1],
                self.statistics['losses']['D_B'][-1],
                self.statistics['losses']['G_GAN_A2B'][-1],
                self.statistics['losses']['G_GAN_B2A'][-1],
                self.statistics['losses']['G_cycle_ABA'][-1],
                self.statistics['losses']['G_cycle_BAB'][-1],
                self.statistics['losses']['G_identity_A'][-1],
                self.statistics['losses']['G_identity_B'][-1],
                self.statistics['timing']['epoch_times'][-1]
            ])
        
        # Сохраняем полную статистику каждые 10 эпох
        if epoch % 10 == 0:
            stats_path = os.path.join(self.metrics_dir, f"statistics_epoch_{epoch:03d}.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.statistics, f, indent=2, ensure_ascii=False)
    
    def _save_final_statistics(self):
        """Сохраняет финальную статистику"""
        # Финальная статистика
        final_stats_path = os.path.join(self.stats_dir, "final_statistics.json")
        with open(final_stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.statistics, f, indent=2, ensure_ascii=False)
        
        # Сводный отчет
        summary_path = os.path.join(self.stats_dir, "training_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("ОБУЧЕНИЕ CYCLEGAN - СВОДНЫЙ ОТЧЕТ\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Дата обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Общее время: {self.statistics['timing']['total_time']:.2f} секунд\n")
            f.write(f"Количество эпох: {len(self.statistics['epochs'])}\n")
            f.write(f"Размер датасета A: {self.statistics['data_info']['dataset_A_size']}\n")
            f.write(f"Размер датасета B: {self.statistics['data_info']['dataset_B_size']}\n\n")
            
            if self.statistics['losses']['G']:
                f.write("ФИНАЛЬНЫЕ ПОТЕРИ:\n")
                f.write(f"  Generator Loss: {self.statistics['losses']['G'][-1]:.4f}\n")
                f.write(f"  Discriminator A Loss: {self.statistics['losses']['D_A'][-1]:.4f}\n")
                f.write(f"  Discriminator B Loss: {self.statistics['losses']['D_B'][-1]:.4f}\n")
        
        self._log(f"💾 Финальная статистика сохранена в: {self.stats_dir}")
    
    def _save_models(self, epoch):
        """Сохраняет модели"""
        g_a2b_path = os.path.join(self.models_dir, "G_A2B", f'G_A2B_epoch_{epoch}.pth')
        g_b2a_path = os.path.join(self.models_dir, "G_B2A", f'G_B2A_epoch_{epoch}.pth')
        
        torch.save(self.G_A2B.state_dict(), g_a2b_path)
        torch.save(self.G_B2A.state_dict(), g_b2a_path)
        
        self._log(f"💾 Модели сохранены (эпоха {epoch})")