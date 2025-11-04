#!/usr/bin/env python3
"""
Главный файл запуска Style Transfer Application
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox

# Добавляем пути для импортов
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'gui'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def check_dependencies():
    """Проверяет наличие необходимых зависимостей"""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import torchvision
    except ImportError:
        missing_deps.append("torchvision")
    
    try:
        from PIL import Image
    except ImportError:
        missing_deps.append("PIL (pillow)")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    if missing_deps:
        return False, ", ".join(missing_deps)
    else:
        return True, None

def setup_directories():
    """Создает необходимые директории"""
    directories = [
        'data/dataset_A',
        'data/dataset_B', 
        'data/models',
        'data/generated',
        'data/statistics'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Директории созданы")

def main():
    """Главная функция"""
    print("🚀 Запуск Style Transfer Application...")
    
    # Проверяем зависимости
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        print(f"❌ Отсутствуют зависимости: {missing_deps}")
        messagebox.showerror(
            "Ошибка зависимостей", 
            f"Не удалось импортировать необходимые библиотеки:\n{missing_deps}\n\n"
            "Убедитесь, что установлены:\n"
            "• torch\n• torchvision\n• Pillow\n• numpy\n\n"
            "Установите командой:\n"
            "pip install torch torchvision pillow numpy"
        )
        return
    
    # Создаем директории
    try:
        setup_directories()
    except Exception as e:
        print(f"❌ Ошибка создания директорий: {e}")
        messagebox.showerror("Ошибка", f"Не удалось создать директории:\n{e}")
        return
    
    # Запускаем GUI
    try:
        from gui.main_window import MainWindow
        
        print("✅ Зависимости проверены")
        print("✅ Директории созданы") 
        print("🖥️  Запуск графического интерфейса...")
        
        app = MainWindow()
        app.run()
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        messagebox.showerror("Ошибка импорта", 
                           f"Не удалось загрузить модули приложения:\n{e}")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        messagebox.showerror("Критическая ошибка", 
                           f"Не удалось запустить приложение:\n{e}")

if __name__ == "__main__":
    main()