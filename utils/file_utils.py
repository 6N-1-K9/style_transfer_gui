import os
import glob
from PIL import Image

def get_image_files(folder_path):
    """Получает все изображения из папки"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    return image_files

def get_model_files(models_dir):
    """Получает все файлы моделей из папки"""
    model_files = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.pth') or file.endswith('.pt'):
                model_files.append(os.path.join(root, file))
    return model_files

def create_directory(path):
    """Создает директорию если не существует"""
    os.makedirs(path, exist_ok=True)

def safe_save_image(image, path):
    """Безопасно сохраняет изображение"""
    try:
        create_directory(os.path.dirname(path))
        image.save(path, quality=95)
        return True
    except Exception as e:
        print(f"Ошибка сохранения изображения: {e}")
        return False