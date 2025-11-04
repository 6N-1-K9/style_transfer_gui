import tkinter as tk
from tkinter import ttk, filedialog
import os

class PathSelector(tk.Frame):
    """Виджет для выбора пути"""
    def __init__(self, parent, label_text, button_text="Обзор", 
                 mode="file", file_types=None, initial_dir=None):
        super().__init__(parent)
        
        self.mode = mode  # "file" или "directory"
        self.file_types = file_types
        self.initial_dir = initial_dir
        
        self.label = tk.Label(self, text=label_text)
        self.label.pack(anchor="w")
        
        frame = tk.Frame(self)
        frame.pack(fill="x", pady=5)
        
        self.path_var = tk.StringVar()
        self.entry = tk.Entry(frame, textvariable=self.path_var, width=50)
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.browse_btn = tk.Button(frame, text=button_text, command=self.browse)
        self.browse_btn.pack(side="right")
    
    def browse(self):
        if self.mode == "file":
            path = filedialog.askopenfilename(
                title="Выберите файл",
                filetypes=self.file_types,
                initialdir=self.initial_dir
            )
        else:  # directory
            path = filedialog.askdirectory(
                title="Выберите папку",
                initialdir=self.initial_dir
            )
        
        if path:
            self.path_var.set(path)
    
    def get_path(self):
        return self.path_var.get()
    
    def set_path(self, path):
        self.path_var.set(path)

class LogWidget(tk.Frame):
    """Виджет для отображения логов"""
    def __init__(self, parent, height=20):
        super().__init__(parent)
        
        self.text_widget = tk.Text(self, height=height, wrap="word", state="disabled")
        scrollbar = tk.Scrollbar(self, orient="vertical", command=self.text_widget.yview)
        self.text_widget.config(yscrollcommand=scrollbar.set)
        
        self.text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def log(self, message):
        self.text_widget.config(state="normal")
        self.text_widget.insert("end", message + "\n")
        self.text_widget.see("end")
        self.text_widget.config(state="disabled")
    
    def clear(self):
        self.text_widget.config(state="normal")
        self.text_widget.delete(1.0, "end")
        self.text_widget.config(state="disabled")