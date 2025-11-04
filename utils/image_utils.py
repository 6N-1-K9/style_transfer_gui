import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def preprocess_image(image_path, image_size=256, device='cpu'):
    """Предобработка изображения для модели"""
    try:
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        return image_tensor, original_size, image
        
    except Exception as e:
        print(f"Ошибка загрузки изображения: {e}")
        return None, None, None

def postprocess_image(styled_tensor, original_size):
    """Постобработка результата модели"""
    styled_tensor = styled_tensor.squeeze(0)
    styled_tensor = (styled_tensor * 0.5 + 0.5).clamp(0, 1)
    
    styled_np = styled_tensor.cpu().numpy().transpose(1, 2, 0)
    styled_np = (styled_np * 255).astype(np.uint8)
    
    styled_image = Image.fromarray(styled_np)
    styled_image = styled_image.resize(original_size, Image.LANCZOS)
    
    return styled_image

def resize_image_for_display(image, max_size=(300, 300)):
    """Изменяет размер изображения для отображения в интерфейсе"""
    image.thumbnail(max_size, Image.LANCZOS)
    return image