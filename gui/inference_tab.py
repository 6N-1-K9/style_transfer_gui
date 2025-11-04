import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
from PIL import Image, ImageTk
from .widgets import PathSelector, LogWidget
from utils.file_utils import get_image_files, safe_save_image
from utils.image_utils import preprocess_image, postprocess_image, resize_image_for_display
from core.inference import StyleTransferInference
from utils.config import InferenceConfig

class InferenceTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.inference = None
        self.current_images = []
        self.config = InferenceConfig()
        self.MAX_PREVIEW_IMAGES = 20  # Ограничиваем количество превью
        
        self.setup_ui()
    
    def setup_ui(self):
        # Левая панель - управление
        left_frame = ttk.Frame(self)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        # ВЫБОР МОДЕЛИ (через PathSelector)
        self.model_selector = PathSelector(
            left_frame, 
            "Модель:", 
            "Выбрать", 
            "file",
            file_types=[
                ("Модели PyTorch", "*.pth *.pt"),
                ("Все файлы", "*.*")
            ],
            initial_dir=os.path.abspath("data/models")
        )
        self.model_selector.pack(fill="x", pady=5)
        
        # Выбор изображения/папки
        self.input_selector = PathSelector(
            left_frame, 
            "Входное изображение/папка:", 
            "Выбрать", 
            "file",
            file_types=[
                ("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Все файлы", "*.*")
            ],
            initial_dir=os.path.abspath("data/dataset_A")
        )
        self.input_selector.pack(fill="x", pady=5)
        
        # Переключатель файл/папка
        self.input_mode = tk.StringVar(value="file")
        file_radio = ttk.Radiobutton(left_frame, text="Один файл", 
                                   variable=self.input_mode, value="file",
                                   command=self.on_input_mode_change)
        file_radio.pack(anchor="w")
        
        folder_radio = ttk.Radiobutton(left_frame, text="Папка с файлами", 
                                     variable=self.input_mode, value="folder",
                                     command=self.on_input_mode_change)
        folder_radio.pack(anchor="w")
        
        # Информация о количестве изображений
        self.image_count_label = ttk.Label(left_frame, text="", foreground="gray")
        self.image_count_label.pack(anchor="w", pady=(0, 10))
        
        # НАСТРОЙКИ ОБРАБОТКИ
        settings_frame = ttk.LabelFrame(left_frame, text="Настройки обработки")
        settings_frame.pack(fill="x", pady=10)
        
        # Размер изображения (ручной ввод)
        ttk.Label(settings_frame, text="Размер обработки:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.image_size_var = tk.StringVar(value=str(self.config.image_size))
        image_size_entry = ttk.Entry(settings_frame, textvariable=self.image_size_var, width=10)
        image_size_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        # Устройство
        ttk.Label(settings_frame, text="Устройство:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.device_var = tk.StringVar(value=self.config.device)
        device_combo = ttk.Combobox(settings_frame, textvariable=self.device_var, 
                                   values=["auto", "cuda", "cpu"], state="readonly", width=10)
        device_combo.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        # Выбор папки для сохранения
        self.output_selector = PathSelector(
            left_frame, 
            "Папка для результатов:", 
            "Выбрать", 
            "directory",
            initial_dir=os.path.abspath("data/generated")
        )
        self.output_selector.pack(fill="x", pady=5)
        
        # Кнопки управления
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill="x", pady=10)
        
        self.apply_btn = ttk.Button(button_frame, text="Применить стиль", command=self.apply_style)
        self.apply_btn.pack(side="left", padx=(0, 5))
        
        # Центральная панель - логи
        center_frame = ttk.Frame(self)
        center_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        self.log_widget = LogWidget(center_frame)
        self.log_widget.pack(fill="both", expand=True)
        
        # Правая панель - предпросмотр
        right_frame = ttk.Frame(self)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Вкладки для исходных и результатных изображений
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill="both", expand=True)
        
        # Вкладка исходных изображений
        self.original_frame = ttk.Frame(notebook)
        notebook.add(self.original_frame, text="Исходные")
        
        self.original_canvas = tk.Canvas(self.original_frame, bg="white")
        scrollbar_orig = ttk.Scrollbar(self.original_frame, orient="vertical", command=self.original_canvas.yview)
        self.original_scrollable = ttk.Frame(self.original_canvas)
        
        self.original_scrollable.bind(
            "<Configure>",
            lambda e: self.original_canvas.configure(scrollregion=self.original_canvas.bbox("all"))
        )
        
        self.original_canvas.create_window((0, 0), window=self.original_scrollable, anchor="nw")
        self.original_canvas.configure(yscrollcommand=scrollbar_orig.set)
        
        self.original_canvas.pack(side="left", fill="both", expand=True)
        scrollbar_orig.pack(side="right", fill="y")
        
        # Вкладка результатов
        self.result_frame = ttk.Frame(notebook)
        notebook.add(self.result_frame, text="Результаты")
        
        self.result_canvas = tk.Canvas(self.result_frame, bg="white")
        scrollbar_res = ttk.Scrollbar(self.result_frame, orient="vertical", command=self.result_canvas.yview)
        self.result_scrollable = ttk.Frame(self.result_canvas)
        
        self.result_scrollable.bind(
            "<Configure>",
            lambda e: self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))
        )
        
        self.result_canvas.create_window((0, 0), window=self.result_scrollable, anchor="nw")
        self.result_canvas.configure(yscrollcommand=scrollbar_res.set)
        
        self.result_canvas.pack(side="left", fill="both", expand=True)
        scrollbar_res.pack(side="right", fill="y")
    
    def on_input_mode_change(self):
        """Обновляет тип выбора ввода при изменении режима"""
        if self.input_mode.get() == "file":
            self.input_selector.mode = "file"
            self.image_count_label.config(text="")
        else:
            self.input_selector.mode = "directory"
            # Показываем количество изображений при выборе папки
            self.update_image_count()
    
    def update_image_count(self):
        """Обновляет информацию о количестве изображений в выбранной папке"""
        input_path = self.input_selector.get_path()
        if input_path and os.path.exists(input_path) and self.input_mode.get() == "folder":
            try:
                image_files = get_image_files(input_path)
                count = len(image_files)
                preview_count = min(count, self.MAX_PREVIEW_IMAGES)
                self.image_count_label.config(
                    text=f"Найдено изображений: {count}" + 
                         (f" (показано первых {preview_count})" if count > self.MAX_PREVIEW_IMAGES else "")
                )
            except:
                self.image_count_label.config(text="")
        else:
            self.image_count_label.config(text="")
    
    def _get_inference_config(self):
        """Получает настройки из UI"""
        try:
            return {
                'image_size': int(self.image_size_var.get()),
                'device': self.device_var.get()
            }
        except ValueError as e:
            raise ValueError(f"Некорректное значение в настройках: {e}")
    
    def apply_style(self):
        """Применяет стиль к выбранным изображениям"""
        input_path = self.input_selector.get_path()
        model_path = self.model_selector.get_path()
        output_dir = self.output_selector.get_path() or "data/generated"
        
        if not input_path:
            messagebox.showerror("Ошибка", "Выберите входное изображение или папку")
            return
        
        if not model_path:
            messagebox.showerror("Ошибка", "Выберите файл модели")
            return
        
        if not os.path.exists(input_path):
            messagebox.showerror("Ошибка", "Выбранный путь не существует")
            return
        
        if not os.path.exists(model_path):
            messagebox.showerror("Ошибка", "Выбранная модель не найдена")
            return
        
        # Проверяем расширение файла модели
        if not (model_path.endswith('.pth') or model_path.endswith('.pt')):
            messagebox.showerror("Ошибка", "Файл модели должен иметь расширение .pth или .pt")
            return
        
        # Получаем настройки
        try:
            inference_config = self._get_inference_config()
        except ValueError as e:
            messagebox.showerror("Ошибка настроек", str(e))
            return
        
        # Получаем список изображений для обработки
        if self.input_mode.get() == "file":
            image_files = [input_path]
        else:
            image_files = get_image_files(input_path)
            if not image_files:
                messagebox.showerror("Ошибка", "В выбранной папке нет изображений")
                return
        
        # Настраиваем UI
        self.apply_btn.config(state="disabled")
        self.log_widget.clear()
        self.log_widget.log(f"🎨 Начинаем обработку {len(image_files)} изображений...")
        self.log_widget.log(f"📁 Модель: {os.path.basename(model_path)}")
        self.log_widget.log(f"⚙️ Настройки: размер={inference_config['image_size']}, устройство={inference_config['device']}")
        
        # Очищаем предпросмотр
        self.clear_preview(self.original_scrollable)
        self.clear_preview(self.result_scrollable)
        
        # Показываем исходные изображения (ограниченное количество)
        self.show_original_images(image_files)
        
        # Запускаем обработку в отдельном потоке
        thread = threading.Thread(
            target=self._inference_worker, 
            args=(image_files, model_path, output_dir, inference_config),
            daemon=True
        )
        thread.start()
    
    def show_original_images(self, image_files):
        """Показывает исходные изображения в предпросмотре (ограниченное количество)"""
        # Ограничиваем количество отображаемых изображений
        preview_files = image_files[:self.MAX_PREVIEW_IMAGES]
        
        if len(image_files) > self.MAX_PREVIEW_IMAGES:
            self.log_widget.log(f"ℹ️  Показаны первые {self.MAX_PREVIEW_IMAGES} из {len(image_files)} изображений")
        
        for i, image_file in enumerate(preview_files):
            try:
                image = Image.open(image_file)
                image = resize_image_for_display(image, (150, 150))
                photo = ImageTk.PhotoImage(image)
                
                # Сохраняем ссылку на изображение
                self.current_images.append(photo)
                
                # Создаем виджет для изображения
                frame = ttk.Frame(self.original_scrollable)
                frame.pack(pady=5, padx=5, fill="x")
                
                label = ttk.Label(frame, image=photo)
                label.pack()
                
                name_label = ttk.Label(frame, text=os.path.basename(image_file), 
                                     font=("Arial", 8))
                name_label.pack()
                
            except Exception as e:
                self.log_widget.log(f"❌ Ошибка загрузки {os.path.basename(image_file)}: {e}")
    
    def clear_preview(self, parent):
        """Очищает область предпросмотра"""
        for widget in parent.winfo_children():
            widget.destroy()
        self.current_images.clear()
    
    def _inference_worker(self, image_files, model_path, output_dir, inference_config):
        """Рабочая функция для применения стиля"""
        try:
            self.inference = StyleTransferInference(
                model_path=model_path,
                device=inference_config['device'],
                log_callback=self.log_widget.log
            )
            
            processed_count = 0
            for i, image_file in enumerate(image_files):
                self.log_widget.log(f"🔄 Обрабатываем: {os.path.basename(image_file)} ({i+1}/{len(image_files)})")
                
                # Применяем стиль
                result_image = self.inference.transfer_style(image_file, inference_config['image_size'])
                
                if result_image:
                    # Сохраняем результат
                    original_name = os.path.splitext(os.path.basename(image_file))[0]
                    model_name = os.path.splitext(os.path.basename(model_path))[0]
                    output_filename = f"{original_name}_{model_name}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    if safe_save_image(result_image, output_path):
                        self.log_widget.log(f"✅ Сохранено: {output_filename}")
                        processed_count += 1
                        
                        # Показываем результат в предпросмотре (только первые N)
                        if processed_count <= self.MAX_PREVIEW_IMAGES:
                            self.after(0, self._show_result_image, result_image, output_filename)
                    else:
                        self.log_widget.log(f"❌ Ошибка сохранения: {output_filename}")
                else:
                    self.log_widget.log(f"❌ Ошибка обработки: {os.path.basename(image_file)}")
            
            self.log_widget.log(f"🎉 Обработка завершена! Успешно обработано: {processed_count}/{len(image_files)}")
            
        except Exception as e:
            self.log_widget.log(f"❌ Критическая ошибка: {str(e)}")
        
        finally:
            self.after(0, self._inference_finished)
    
    def _show_result_image(self, image, filename):
        """Показывает результат в предпросмотре (вызывается из главного потока)"""
        try:
            image = resize_image_for_display(image, (150, 150))
            photo = ImageTk.PhotoImage(image)
            
            # Сохраняем ссылку на изображение
            self.current_images.append(photo)
            
            # Создаем виджет для изображения
            frame = ttk.Frame(self.result_scrollable)
            frame.pack(pady=5, padx=5, fill="x")
            
            label = ttk.Label(frame, image=photo)
            label.pack()
            
            name_label = ttk.Label(frame, text=filename, font=("Arial", 8))
            name_label.pack()
            
        except Exception as e:
            self.log_widget.log(f"❌ Ошибка отображения результата: {e}")
    
    def _inference_finished(self):
        """Вызывается когда обработка завершена"""
        self.apply_btn.config(state="normal")