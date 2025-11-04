import tkinter as tk
from tkinter import ttk
from .training_tab import TrainingTab
from .inference_tab import InferenceTab

class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.setup_tabs()
    
    def setup_window(self):
        """Настраивает главное окно"""
        self.root.title("Style Transfer Application")
        self.root.geometry("1200x700")
        self.root.minsize(1000, 600)
        
        # Центрирование окна
        self.center_window()
        
        # Стиль
        self.setup_styles()
    
    def center_window(self):
        """Центрирует окно на экране"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def setup_styles(self):
        """Настраивает стили интерфейса"""
        style = ttk.Style()
        
        # Современный стиль
        style.theme_use('clam')
        
        # Кастомные стили
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
    
    def setup_tabs(self):
        """Настраивает вкладки приложения"""
        # Создаем notebook для вкладок
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Вкладка обучения
        self.training_tab = TrainingTab(notebook)
        notebook.add(self.training_tab, text='🎓 Обучение модели')
        
        # Вкладка применения
        self.inference_tab = InferenceTab(notebook)
        notebook.add(self.inference_tab, text='🎨 Применение модели')
        
        # Вкладка о программе
        about_tab = ttk.Frame(notebook)
        notebook.add(about_tab, text='ℹ️ О программе')
        self.setup_about_tab(about_tab)
    
    def setup_about_tab(self, parent):
        """Настраивает вкладку 'О программе'"""
        about_text = """
Style Transfer Application

Программа для переноса художественных стилей 
с использованием нейросетей CycleGAN.

Возможности:
• Обучение моделей на собственных датасетах
• Применение обученных моделей к изображениям
• Визуализация процесса обучения
• Пакетная обработка изображений

Технологии:
• PyTorch для нейросетей
• CycleGAN архитектура
• Tkinter для интерфейса

Автор: Style Transfer Team
Версия: 1.0.0
        """
        
        text_widget = tk.Text(parent, wrap='word', font=('Arial', 10), 
                             padx=10, pady=10, relief='flat')
        text_widget.insert('1.0', about_text)
        text_widget.config(state='disabled')
        text_widget.pack(fill='both', expand=True)
    
    def run(self):
        """Запускает приложение"""
        self.root.mainloop()