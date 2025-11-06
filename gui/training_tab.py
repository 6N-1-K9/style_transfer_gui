import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import torch
from .widgets import PathSelector, LogWidget
from utils.file_utils import get_model_files, create_directory
from core.trainer import StyleTransferTrainer
from utils.config import TrainingConfig
from utils.settings_utils import TrainingSettings, find_latest_checkpoint, find_latest_model, validate_resume_files, get_settings_path

class TrainingTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.trainer = None
        self.training_thread = None
        self.is_training = False
        self.config = TrainingConfig()
        
        self.setup_ui()
    
    def setup_ui(self):
        # Основной контейнер с фиксированным разделением
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - настройки (40% ширины)
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Правая панель - логи (60% ширины)  
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        # Настраиваем пропорции панелей
        main_paned.sashpos(0, int(self.winfo_screenwidth() * 0.4))
        
        # Создаем прокручиваемую рамку для левой панели
        left_canvas = tk.Canvas(left_frame, borderwidth=0, highlightthickness=0)
        scrollbar_left = ttk.Scrollbar(left_frame, orient="vertical", command=left_canvas.yview)
        scrollable_left_frame = ttk.Frame(left_canvas)
        
        scrollable_left_frame.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )
        
        left_canvas.create_window((0, 0), window=scrollable_left_frame, anchor="nw")
        left_canvas.configure(yscrollcommand=scrollbar_left.set)
        
        left_canvas.pack(side="left", fill="both", expand=True)
        scrollbar_left.pack(side="right", fill="y")
        
        # Левая панель содержимое
        # Выбор датасетов
        self.dataset_a_selector = PathSelector(
            scrollable_left_frame, "Датасет A (Предметы):", "Выбрать", "directory"
        )
        self.dataset_a_selector.pack(fill="x", pady=5)
        
        self.dataset_b_selector = PathSelector(
            scrollable_left_frame, "Датасет B (Стили):", "Выбрать", "directory"
        )
        self.dataset_b_selector.pack(fill="x", pady=5)
        
        # Выбор папок для сохранения (ОБЯЗАТЕЛЬНЫЕ)
        self.models_selector = PathSelector(
            scrollable_left_frame, "Папка для моделей*:", "Выбрать", "directory"
        )
        self.models_selector.pack(fill="x", pady=5)
        
        self.stats_selector = PathSelector(
            scrollable_left_frame, "Папка для статистики*:", "Выбрать", "directory"
        )
        self.stats_selector.pack(fill="x", pady=5)
        
        # Информация об обязательных полях
        required_label = ttk.Label(scrollable_left_frame, text="* - обязательные поля", 
                                 foreground="red", font=("Arial", 8))
        required_label.pack(anchor="w", pady=(0, 10))
        
        # НАСТРОЙКИ ОБУЧЕНИЯ
        settings_frame = ttk.LabelFrame(scrollable_left_frame, text="Основные настройки обучения")
        settings_frame.pack(fill="x", pady=10)
        
        # Сетка для настроек
        row = 0
        
        # Размер изображения (ручной ввод)
        ttk.Label(settings_frame, text="Размер изображения:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.image_size_var = tk.StringVar(value=str(self.config.image_size))
        image_size_entry = ttk.Entry(settings_frame, textvariable=self.image_size_var, width=10)
        image_size_entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(settings_frame, text="px", font=("Arial", 8)).grid(row=row, column=2, sticky="w", padx=(0, 5), pady=2)
        row += 1
        
        # Batch size
        ttk.Label(settings_frame, text="Batch size:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.batch_size_var = tk.StringVar(value=str(self.config.batch_size))
        batch_size_entry = ttk.Entry(settings_frame, textvariable=self.batch_size_var, width=10)
        batch_size_entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1
        
        # Количество эпох
        ttk.Label(settings_frame, text="Эпохи:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.epochs_var = tk.StringVar(value=str(self.config.epochs))
        epochs_entry = ttk.Entry(settings_frame, textvariable=self.epochs_var, width=10)
        epochs_entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1
        
        # Learning rate
        ttk.Label(settings_frame, text="Learning Rate:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.lr_var = tk.StringVar(value=str(self.config.lr))
        lr_entry = ttk.Entry(settings_frame, textvariable=self.lr_var, width=10)
        lr_entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1
        
        # Lambda Cycle
        ttk.Label(settings_frame, text="Lambda Cycle:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.lambda_cycle_var = tk.StringVar(value=str(self.config.lambda_cycle))
        lambda_cycle_entry = ttk.Entry(settings_frame, textvariable=self.lambda_cycle_var, width=10)
        lambda_cycle_entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1
        
        # Lambda Identity
        ttk.Label(settings_frame, text="Lambda Identity:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.lambda_identity_var = tk.StringVar(value=str(self.config.lambda_identity))
        lambda_identity_entry = ttk.Entry(settings_frame, textvariable=self.lambda_identity_var, width=10)
        lambda_identity_entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1
        
        # ДОПОЛНИТЕЛЬНЫЕ НАСТРОЙКИ
        advanced_frame = ttk.LabelFrame(scrollable_left_frame, text="Архитектурные настройки")
        advanced_frame.pack(fill="x", pady=10)
        
        row_adv = 0
        
        # Residual Blocks (ТЕПЕРЬ ЧИСЛОВОЙ ВВОД)
        ttk.Label(advanced_frame, text="Residual Blocks:").grid(row=row_adv, column=0, sticky="w", padx=5, pady=2)
        self.n_residual_blocks_var = tk.StringVar(value=str(self.config.n_residual_blocks))
        n_residual_blocks_entry = ttk.Entry(advanced_frame, textvariable=self.n_residual_blocks_var, width=10)
        n_residual_blocks_entry.grid(row=row_adv, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(advanced_frame, text="(6, 9, 12...)", font=("Arial", 8)).grid(row=row_adv, column=2, sticky="w", padx=(0, 5), pady=2)
        row_adv += 1
        
        # Use Dropout
        ttk.Label(advanced_frame, text="Use Dropout:").grid(row=row_adv, column=0, sticky="w", padx=5, pady=2)
        self.use_dropout_var = tk.BooleanVar(value=self.config.use_dropout)
        use_dropout_check = ttk.Checkbutton(advanced_frame, variable=self.use_dropout_var)
        use_dropout_check.grid(row=row_adv, column=1, sticky="w", padx=5, pady=2)
        row_adv += 1
        
        # Gradient Clip
        ttk.Label(advanced_frame, text="Gradient Clip:").grid(row=row_adv, column=0, sticky="w", padx=5, pady=2)
        self.gradient_clip_var = tk.StringVar(value=str(self.config.gradient_clip))
        gradient_clip_entry = ttk.Entry(advanced_frame, textvariable=self.gradient_clip_var, width=10)
        gradient_clip_entry.grid(row=row_adv, column=1, sticky="w", padx=5, pady=2)
        row_adv += 1
        
        # Настройки планировщика
        scheduler_frame = ttk.LabelFrame(scrollable_left_frame, text="Настройки планировщика Learning Rate")
        scheduler_frame.pack(fill="x", pady=10)
        
        row_sched = 0
        
        # LR Decay Start
        ttk.Label(scheduler_frame, text="LR Decay Start:").grid(row=row_sched, column=0, sticky="w", padx=5, pady=2)
        self.lr_decay_start_var = tk.StringVar(value=str(self.config.lr_decay_start))
        lr_decay_start_entry = ttk.Entry(scheduler_frame, textvariable=self.lr_decay_start_var, width=10)
        lr_decay_start_entry.grid(row=row_sched, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(scheduler_frame, text="эпоха", font=("Arial", 8)).grid(row=row_sched, column=2, sticky="w", padx=(0, 5), pady=2)
        row_sched += 1
        
        # LR Decay End
        ttk.Label(scheduler_frame, text="LR Decay End:").grid(row=row_sched, column=0, sticky="w", padx=5, pady=2)
        self.lr_decay_end_var = tk.StringVar(value=str(self.config.lr_decay_end))
        lr_decay_end_entry = ttk.Entry(scheduler_frame, textvariable=self.lr_decay_end_var, width=10)
        lr_decay_end_entry.grid(row=row_sched, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(scheduler_frame, text="эпоха", font=("Arial", 8)).grid(row=row_sched, column=2, sticky="w", padx=(0, 5), pady=2)
        row_sched += 1
        
        # Final LR Ratio
        ttk.Label(scheduler_frame, text="Final LR Ratio:").grid(row=row_sched, column=0, sticky="w", padx=5, pady=2)
        self.final_lr_ratio_var = tk.StringVar(value=str(self.config.final_lr_ratio))
        final_lr_ratio_entry = ttk.Entry(scheduler_frame, textvariable=self.final_lr_ratio_var, width=10)
        final_lr_ratio_entry.grid(row=row_sched, column=1, sticky="w", padx=5, pady=2)
        row_sched += 1
        
        # ДОПОЛНИТЕЛЬНЫЕ ОПЦИИ
        options_frame = ttk.LabelFrame(scrollable_left_frame, text="Дополнительные опции")
        options_frame.pack(fill="x", pady=10)
        
        row_opt = 0
        
        # Use Early Stopping
        ttk.Label(options_frame, text="Early Stopping:").grid(row=row_opt, column=0, sticky="w", padx=5, pady=2)
        self.use_early_stopping_var = tk.BooleanVar(value=self.config.use_early_stopping)
        use_early_stopping_check = ttk.Checkbutton(options_frame, variable=self.use_early_stopping_var)
        use_early_stopping_check.grid(row=row_opt, column=1, sticky="w", padx=5, pady=2)
        row_opt += 1
        
        # Early Stopping Patience
        ttk.Label(options_frame, text="Early Stopping Patience:").grid(row=row_opt, column=0, sticky="w", padx=5, pady=2)
        self.early_stopping_patience_var = tk.StringVar(value=str(self.config.early_stopping_patience))
        early_stopping_patience_entry = ttk.Entry(options_frame, textvariable=self.early_stopping_patience_var, width=10)
        early_stopping_patience_entry.grid(row=row_opt, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(options_frame, text="эпох", font=("Arial", 8)).grid(row=row_opt, column=2, sticky="w", padx=(0, 5), pady=2)
        row_opt += 1
        
        # Save Interval
        ttk.Label(options_frame, text="Интервал сохранения:").grid(row=row_opt, column=0, sticky="w", padx=5, pady=2)
        self.save_interval_var = tk.StringVar(value=str(self.config.save_interval))
        save_interval_entry = ttk.Entry(options_frame, textvariable=self.save_interval_var, width=10)
        save_interval_entry.grid(row=row_opt, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(options_frame, text="эпох", font=("Arial", 8)).grid(row=row_opt, column=2, sticky="w", padx=(0, 5), pady=2)
        row_opt += 1
        
        # Кнопки управления
        button_frame = ttk.Frame(scrollable_left_frame)
        button_frame.pack(fill="x", pady=15)
        
        self.start_btn = ttk.Button(button_frame, text="Начать обучение", command=self.start_training)
        self.start_btn.pack(side="left", padx=(0, 5))
        
        self.resume_btn = ttk.Button(button_frame, text="Продолжить обучение", command=self.resume_training)
        self.resume_btn.pack(side="left", padx=(0, 5))
        
        self.stop_btn = ttk.Button(button_frame, text="Остановить", command=self.stop_training, state="disabled")
        self.stop_btn.pack(side="left")
        
        # Прогресс бар
        progress_frame = ttk.Frame(scrollable_left_frame)
        progress_frame.pack(fill="x", pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Готов к обучению")
        self.progress_label.pack()
        
        # Список моделей
        models_frame = ttk.LabelFrame(scrollable_left_frame, text="Обученные модели")
        models_frame.pack(fill="both", expand=True, pady=10)
        
        # Фрейм для списка моделей с прокруткой
        models_list_frame = ttk.Frame(models_frame)
        models_list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.models_listbox = tk.Listbox(models_list_frame)
        models_scrollbar = ttk.Scrollbar(models_list_frame, orient="vertical", command=self.models_listbox.yview)
        self.models_listbox.configure(yscrollcommand=models_scrollbar.set)
        
        self.models_listbox.pack(side="left", fill="both", expand=True)
        models_scrollbar.pack(side="right", fill="y")
        
        refresh_btn = ttk.Button(models_frame, text="Обновить список моделей", command=self.refresh_models_list)
        refresh_btn.pack(pady=5)
        
        # Правая панель - логи (фиксированная)
        self.log_widget = LogWidget(right_frame)
        self.log_widget.pack(fill="both", expand=True)
    
    def refresh_models_list(self):
        """Обновляет список обученных моделей"""
        models_dir = self.models_selector.get_path()
        if not models_dir:
            self.log_widget.log("❌ Сначала выберите папку для моделей")
            return
            
        self.models_listbox.delete(0, tk.END)
        
        model_files = get_model_files(models_dir)
        for model_file in model_files:
            self.models_listbox.insert(tk.END, os.path.basename(model_file))
        
        if model_files:
            self.log_widget.log(f"✅ Загружено {len(model_files)} моделей")
        else:
            self.log_widget.log("ℹ️ Модели не найдены")
    
    def _get_training_config(self):
        """Получает настройки из UI и возвращает конфиг"""
        try:
            config = {
                'image_size': int(self.image_size_var.get()),
                'batch_size': int(self.batch_size_var.get()),
                'epochs': int(self.epochs_var.get()),
                'lr': float(self.lr_var.get()),
                'lambda_cycle': float(self.lambda_cycle_var.get()),
                'lambda_identity': float(self.lambda_identity_var.get()),
                'n_residual_blocks': int(self.n_residual_blocks_var.get()),
                'use_dropout': self.use_dropout_var.get(),
                'gradient_clip': float(self.gradient_clip_var.get()),
                'lr_decay_start': int(self.lr_decay_start_var.get()),
                'lr_decay_end': int(self.lr_decay_end_var.get()),
                'final_lr_ratio': float(self.final_lr_ratio_var.get()),
                'use_early_stopping': self.use_early_stopping_var.get(),
                'early_stopping_patience': int(self.early_stopping_patience_var.get()),
                'save_interval': int(self.save_interval_var.get()),
            }
            return config
        except ValueError as e:
            raise ValueError(f"Некорректное значение в настройках: {e}")
    
    def _update_ui_from_settings(self, settings):
        """Обновляет UI с настройками из файла"""
        config = settings.training_config
        paths = settings.paths
        
        # Обновляем пути
        self.dataset_a_selector.set_path(paths['dataset_a'])
        self.dataset_b_selector.set_path(paths['dataset_b'])
        self.models_selector.set_path(paths['models_dir'])
        self.stats_selector.set_path(paths['stats_dir'])
        
        # Обновляем настройки
        self.image_size_var.set(str(config['image_size']))
        self.batch_size_var.set(str(config['batch_size']))
        self.epochs_var.set(str(config['epochs']))
        self.lr_var.set(str(config['lr']))
        self.lambda_cycle_var.set(str(config['lambda_cycle']))
        self.lambda_identity_var.set(str(config['lambda_identity']))
        self.n_residual_blocks_var.set(str(config['n_residual_blocks']))
        self.use_dropout_var.set(config['use_dropout'])
        self.gradient_clip_var.set(str(config['gradient_clip']))
        self.lr_decay_start_var.set(str(config['lr_decay_start']))
        self.lr_decay_end_var.set(str(config['lr_decay_end']))
        self.final_lr_ratio_var.set(str(config['final_lr_ratio']))
        self.use_early_stopping_var.set(config['use_early_stopping'])
        self.early_stopping_patience_var.set(str(config['early_stopping_patience']))
        self.save_interval_var.set(str(config['save_interval']))
    
    def resume_training(self):
        """Продолжает обучение с последней точки"""
        if self.is_training:
            return
        
        # Запрашиваем файл настроек с ПРАВИЛЬНЫМ фильтром
        settings_path = filedialog.askopenfilename(
            title="Выберите файл настроек обучения (training_settings.json)",
            filetypes=[
                ("JSON files", "*.json"),
                ("Training settings", "training_settings.json"),
                ("All files", "*.*")
            ],
            initialdir=os.path.abspath("data/models")
        )
        
        if not settings_path:
            return
        
        # Проверяем, что файл существует и имеет правильное расширение
        if not os.path.exists(settings_path):
            messagebox.showerror("Ошибка", f"Файл не существует: {settings_path}")
            return
        
        if not settings_path.lower().endswith('.json'):
            messagebox.showerror("Ошибка", "Выберите файл с расширением .json")
            return
        
        # Загружаем настройки
        settings = TrainingSettings()
        if not settings.load(settings_path):
            messagebox.showerror("Ошибка", "Не удалось загрузить файл настроек. Файл поврежден или имеет неверный формат.")
            return
        
        # Находим последний чекпоинт или модель
        models_dir = settings.paths['models_dir']
        stats_dir = settings.paths['stats_dir']
        
        # Логируем информацию для отладки
        self.log_widget.log(f"🔍 Поиск чекпоинтов в: {stats_dir}")
        self.log_widget.log(f"🔍 Поиск моделей в: {models_dir}")
        
        checkpoint_path = find_latest_checkpoint(stats_dir)
        if not checkpoint_path:
            self.log_widget.log("ℹ️ Чекпоинт не найден, ищем последнюю модель...")
            checkpoint_path = find_latest_model(models_dir)
        
        if checkpoint_path:
            self.log_widget.log(f"✅ Найден файл для продолжения: {os.path.basename(checkpoint_path)}")
        else:
            self.log_widget.log("❌ Не найден ни чекпоинт, ни модель для продолжения")
        
        # Проверяем валидность файлов
        is_valid, errors = validate_resume_files(settings_path, checkpoint_path)
        if not is_valid:
            error_message = "\n".join(errors)
            messagebox.showerror("Ошибка", f"Не удалось начать продолжение обучения:\n{error_message}")
            return
        
        # Обновляем UI с настройками из файла
        self._update_ui_from_settings(settings)
        
        # Настраиваем UI
        self.is_training = True
        self.start_btn.config(state="disabled")
        self.resume_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress_var.set(0)
        self.progress_label.config(text="Подготовка к продолжению обучения...")
        self.log_widget.clear()
        
        self.log_widget.log("🔄 Продолжение обучения...")
        self.log_widget.log(f"📁 Настройки: {os.path.basename(settings_path)}")
        self.log_widget.log(f"📁 Чекпоинт: {os.path.basename(checkpoint_path)}")
        self.log_widget.log(f"📊 Продолжаем с эпохи: {settings.model_info.get('current_epoch', 1)}")
        
        # Запускаем обучение в отдельном потоке с чекпоинтом
        self.training_thread = threading.Thread(
            target=self._resume_training_worker, 
            args=(settings, checkpoint_path),
            daemon=True
        )
        self.training_thread.start()
    
    def _resume_training_worker(self, settings, checkpoint_path):
        """Рабочая функция для продолжения обучения в отдельном потоке"""
        try:
            # Создаем тренер с загрузкой из чекпоинта
            self.trainer = StyleTransferTrainer(
                dataset_a_path=settings.paths['dataset_a'],
                dataset_b_path=settings.paths['dataset_b'],
                models_dir=settings.paths['models_dir'],
                stats_dir=settings.paths['stats_dir'],
                **settings.training_config,
                resume_from=checkpoint_path,
                log_callback=self.log_widget.log,
                progress_callback=self.update_progress
            )
            
            # Продолжаем обучение
            self.trainer.train()
            
        except Exception as e:
            self.log_widget.log(f"❌ Ошибка продолжения обучения: {str(e)}")
        
        finally:
            self.is_training = False
            self.after(0, self._training_finished)
    
    def start_training(self):
        """Запускает обучение в отдельном потоке"""
        if self.is_training:
            return
        
        # Получаем пути
        dataset_a = self.dataset_a_selector.get_path()
        dataset_b = self.dataset_b_selector.get_path()
        models_dir = self.models_selector.get_path()
        stats_dir = self.stats_selector.get_path()
        
        # ПРОВЕРКА ОБЯЗАТЕЛЬНЫХ ПОЛЕЙ
        if not models_dir:
            messagebox.showerror("Ошибка", "Выберите папку для сохранения моделей")
            return
            
        if not stats_dir:
            messagebox.showerror("Ошибка", "Выберите папку для статистики")
            return
        
        # Проверяем существование датасетов
        if not dataset_a or not os.path.exists(dataset_a):
            messagebox.showerror("Ошибка", "Выберите корректный датасет A")
            return
        
        if not dataset_b or not os.path.exists(dataset_b):
            messagebox.showerror("Ошибка", "Выберите корректный датасет B")
            return
        
        # Получаем настройки
        try:
            training_config = self._get_training_config()
        except ValueError as e:
            messagebox.showerror("Ошибка настроек", str(e))
            return
        
        # Создаем папки если не существуют
        create_directory(models_dir)
        create_directory(stats_dir)
        
        # Настраиваем UI
        self.is_training = True
        self.start_btn.config(state="disabled")
        self.resume_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress_var.set(0)
        self.progress_label.config(text="Подготовка к обучению...")
        self.log_widget.clear()
        
        # Логируем настройки
        self.log_widget.log("⚙️  Настройки обучения:")
        self.log_widget.log(f"   Размер изображения: {training_config['image_size']}px")
        self.log_widget.log(f"   Batch size: {training_config['batch_size']}")
        self.log_widget.log(f"   Эпохи: {training_config['epochs']}")
        self.log_widget.log(f"   Learning Rate: {training_config['lr']}")
        self.log_widget.log(f"   Lambda Cycle: {training_config['lambda_cycle']}")
        self.log_widget.log(f"   Lambda Identity: {training_config['lambda_identity']}")
        self.log_widget.log(f"   Residual Blocks: {training_config['n_residual_blocks']}")
        self.log_widget.log(f"   Use Dropout: {training_config['use_dropout']}")
        self.log_widget.log(f"   Gradient Clip: {training_config['gradient_clip']}")
        self.log_widget.log("")
        
        # Запускаем обучение в отдельном потоке
        self.training_thread = threading.Thread(
            target=self._training_worker, 
            args=(dataset_a, dataset_b, models_dir, stats_dir, training_config),
            daemon=True
        )
        self.training_thread.start()
    
    def _training_worker(self, dataset_a, dataset_b, models_dir, stats_dir, training_config):
        """Рабочая функция для обучения в отдельном потоке"""
        try:
            # Создаем тренер
            self.trainer = StyleTransferTrainer(
                dataset_a_path=dataset_a,
                dataset_b_path=dataset_b,
                models_dir=models_dir,
                stats_dir=stats_dir,
                **training_config,
                log_callback=self.log_widget.log,
                progress_callback=self.update_progress
            )
            
            # Запускаем обучение
            self.trainer.train()
            
        except Exception as e:
            self.log_widget.log(f"❌ Ошибка обучения: {str(e)}")
        
        finally:
            self.is_training = False
            self.after(0, self._training_finished)
    
    def update_progress(self, epoch, total_epochs, loss):
        """Обновляет прогресс обучения"""
        progress = (epoch / total_epochs) * 100
        self.progress_var.set(progress)
        self.progress_label.config(text=f"Эпоха {epoch}/{total_epochs}, Loss: {loss:.4f}")
    
    def _training_finished(self):
        """Вызывается когда обучение завершено"""
        self.start_btn.config(state="normal")
        self.resume_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.progress_label.config(text="Обучение завершено")
        self.refresh_models_list()
        self.log_widget.log("✅ Обучение завершено!")
    
    def stop_training(self):
        """Останавливает обучение"""
        if self.trainer and self.is_training:
            self.trainer.stop_training = True
            self.log_widget.log("⏹️ Остановка обучения...")