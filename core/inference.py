import os
import torch
from .models import StyleGenerator
from utils.image_utils import preprocess_image, postprocess_image

class StyleTransferInference:
    def __init__(self, model_path, device='auto', log_callback=None):
        """
        Инициализация для применения стиля
        
        Args:
            model_path: путь к файлу модели
            device: 'auto', 'cuda', или 'cpu'
            log_callback: функция для логирования
        """
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model_path = model_path
        self.log_callback = log_callback or print
        self.model = None
        
        self._load_model()
    
    def _log(self, message):
        """Логирует сообщение"""
        self.log_callback(message)
    
    def _load_model(self):
        """Загружает модель"""
        try:
            self._log(f"🔄 Загрузка модели из {os.path.basename(self.model_path)}...")
            
            # Определяем архитектуру по имени файла
            n_residual_blocks = 9  # Можно добавить логику определения из имени файла
            
            self.model = StyleGenerator(n_residual_blocks=n_residual_blocks).to(self.device)
            
            # Загружаем веса
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            
            self._log("✅ Модель успешно загружена")
            
        except Exception as e:
            self._log(f"❌ Ошибка загрузки модели: {e}")
            raise
    
    def transfer_style(self, image_path, image_size=256):
        """
        Применяет стиль к изображению
        
        Args:
            image_path: путь к входному изображению
            image_size: размер для обработки
            
        Returns:
            PIL.Image или None в случае ошибки
        """
        try:
            # Предобработка
            image_tensor, original_size, original_image = preprocess_image(
                image_path, image_size, self.device
            )
            
            if image_tensor is None:
                return None
            
            # Применяем стиль
            with torch.no_grad():
                styled_tensor = self.model(image_tensor)
            
            # Постобработка
            styled_image = postprocess_image(styled_tensor, original_size)
            
            return styled_image
            
        except Exception as e:
            self._log(f"❌ Ошибка применения стиля к {os.path.basename(image_path)}: {e}")
            return None
    
    def transfer_style_batch(self, image_paths, image_size=256):
        """
        Применяет стиль к нескольким изображениям
        
        Args:
            image_paths: список путей к изображениям
            image_size: размер для обработки
            
        Returns:
            dict: {путь_к_файлу: PIL.Image}
        """
        results = {}
        
        for image_path in image_paths:
            styled_image = self.transfer_style(image_path, image_size)
            if styled_image:
                results[image_path] = styled_image
        
        return results