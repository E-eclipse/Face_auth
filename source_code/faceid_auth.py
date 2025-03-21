import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import time
import numpy as np
import re
from cv2.face import LBPHFaceRecognizer_create
from PIL import Image, ImageTk
import threading
import json
from datetime import datetime

class FaceAuthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Система Авторизации по Лицу")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Variables
        self.face_id = tk.StringVar()
        self.is_capturing = False
        self.count = 0
        self.last_print_time = time.time()
        self.recognizer = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.successful_recognitions = 0
        self.is_authenticated = False
        
        # Load users data
        self.users_file = "users.json"
        self.load_users()
        
        # Create main container with style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('Main.TFrame', background='#2c3e50')
        style.configure('Title.TLabel', 
                       background='#2c3e50',
                       foreground='white',
                       font=('Arial', 28, 'bold'))
        style.configure('Subtitle.TLabel',
                       background='#2c3e50',
                       foreground='#ecf0f1',
                       font=('Arial', 14))
        style.configure('Status.TLabel',
                       background='#2c3e50',
                       foreground='#ecf0f1',
                       font=('Arial', 12))
        style.configure('Info.TLabel',
                       background='#2c3e50',
                       foreground='#ecf0f1',
                       font=('Arial', 12))
        style.configure('Custom.TLabelframe',
                       background='#34495e',
                       foreground='white')
        style.configure('Custom.TLabelframe.Label',
                       background='#34495e',
                       foreground='white',
                       font=('Arial', 12, 'bold'))
        style.configure('Custom.TButton',
                       background='#3498db',
                       foreground='white',
                       font=('Arial', 12),
                       padding=10)
        style.configure('Custom.TEntry',
                       fieldbackground='#ecf0f1',
                       foreground='#2c3e50',
                       font=('Arial', 12))
        
        self.main_frame = ttk.Frame(self.root, style='Main.TFrame', padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(self.main_frame, 
                              text="Система Авторизации по Лицу",
                              style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 30))
        
        # Create video frame with border
        self.video_frame = ttk.Label(self.main_frame, borderwidth=2, relief="solid")
        self.video_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="nsew")
        
        # Create control buttons
        self.create_control_buttons()
        
        # Create status label
        self.status_label = ttk.Label(self.main_frame, 
                                    text="Система готова к работе",
                                    style='Status.TLabel')
        self.status_label.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Initialize camera
        self.cam = None
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        
    def load_users(self):
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        else:
            self.users = {}
            
    def save_users(self):
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f)
            
    def create_control_buttons(self):
        registration_frame = ttk.LabelFrame(self.main_frame, 
                                         text="Регистрация нового пользователя",
                                         style='Custom.TLabelframe',
                                         padding="20")
        registration_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        
        ttk.Label(registration_frame, 
                 text="Имя пользователя:",
                 style='Subtitle.TLabel').grid(row=0, column=0, padx=10, pady=10)
        ttk.Entry(registration_frame, 
                 textvariable=self.face_id,
                 style='Custom.TEntry').grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Button(registration_frame, 
                  text="Начать регистрацию",
                  command=self.start_registration,
                  style='Custom.TButton').grid(row=1, column=0, columnspan=2, pady=20)
        
        guide_frame = ttk.LabelFrame(registration_frame,
                                   text="Инструкция по регистрации",
                                   style='Custom.TLabelframe',
                                   padding="15")
        guide_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky="nsew")
        
        guide_text = """
1. Введите имя пользователя (только английские буквы)
2. Нажмите "Начать регистрацию"
3. Расположите лицо в центре кадра
4. Вращайте голову в разные стороны
5. Дождитесь завершения регистрации (200 снимков)
        """
        
        ttk.Label(guide_frame,
                 text=guide_text,
                 style='Info.TLabel').grid(row=0, column=0, pady=10, sticky="w")
        
        auth_frame = ttk.LabelFrame(self.main_frame, 
                                  text="Авторизация",
                                  style='Custom.TLabelframe',
                                  padding="20")
        auth_frame.grid(row=2, column=1, padx=20, pady=10, sticky="nsew")
        
        ttk.Label(auth_frame, 
                 text="Имя пользователя:",
                 style='Subtitle.TLabel').grid(row=0, column=0, padx=10, pady=10)
        self.auth_username = ttk.Entry(auth_frame, style='Custom.TEntry')
        self.auth_username.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Button(auth_frame, 
                  text="Войти в систему",
                  command=self.start_authentication,
                  style='Custom.TButton').grid(row=1, column=0, columnspan=2, pady=20)
        
        auth_guide_frame = ttk.LabelFrame(auth_frame,
                                        text="Инструкция по авторизации",
                                        style='Custom.TLabelframe',
                                        padding="15")
        auth_guide_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky="nsew")
        
        auth_guide_text = """
1. Введите имя пользователя
2. Нажмите "Войти в систему"
3. Расположите лицо в центре кадра
4. Дождитесь подтверждения (10 успешных распознаваний)
        """
        
        ttk.Label(auth_guide_frame,
                 text=auth_guide_text,
                 style='Info.TLabel').grid(row=0, column=0, pady=10, sticky="w")
        
    def start_registration(self):
        if not self.face_id.get():
            messagebox.showerror("Ошибка", "Пожалуйста, введите имя пользователя!")
            return
            
        if not re.match("^[a-zA-Z]+$", self.face_id.get()):
            messagebox.showerror("Ошибка", "Используйте только английские буквы!")
            return
            
        if self.face_id.get() in self.users:
            messagebox.showerror("Ошибка", "Пользователь уже существует!")
            return
            
        self.is_capturing = True
        self.count = 0
        self.status_label.config(text="Регистрация в процессе...")
        
        # Create directory for face images
        face_dir = os.path.join(os.getcwd(), f"dataset_{self.face_id.get()}")
        if not os.path.exists(face_dir):
            os.makedirs(face_dir)
            
        # Start camera thread
        threading.Thread(target=self.capture_training_images, args=(face_dir,), daemon=True).start()
        
    def capture_training_images(self, face_dir):
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        while self.is_capturing and self.count < 200:
            ret, img = self.cam.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.count += 1
                
                # Save face image
                img_path = f"{face_dir}/User.{self.face_id.get()}.{self.count}.jpg"
                cv2.imwrite(img_path, gray[y:y+h, x:x+w])
                
                # Update status
                self.status_label.config(text=f"{self.count}/200")
                
            # Display position instructions
            cv2.putText(img, f"{self.count}/200", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Update video frame
            self.update_video_frame(img)
            
        self.cam.release()
        self.train_model(face_dir)
            
    def train_model(self, face_dir):
        recognizer = LBPHFaceRecognizer_create()
        faces, ids = self.get_images_and_labels(face_dir)
        recognizer.train(faces, np.array(ids))
        
        # Save model
        model_dir = os.path.join(os.getcwd(), f"{self.face_id.get()}_model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        recognizer.write(f'{model_dir}/face_{self.face_id.get()}.yml')
        
        # Save user data
        self.users[self.face_id.get()] = {
            'model_path': f'{model_dir}/face_{self.face_id.get()}.yml',
            'registration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'last_login': None,
            'total_logins': 0
        }
        self.save_users()
        
        self.status_label.config(text="Регистрация завершена!")
        messagebox.showinfo("Успех", "Регистрация успешно завершена!")
        
        if self.cam is not None:
            self.cam.release()
            self.cam = None
        
        self.video_frame.configure(image='')
        
    def get_images_and_labels(self, path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        ids = []
        
        for image_path in image_paths:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces.append(img)
            id = int(os.path.split(image_path)[-1].split(".")[2])
            ids.append(id)
            
        return faces, ids
        
    def start_authentication(self):
        username = self.auth_username.get()
        if not username:
            messagebox.showerror("Ошибка", "Пожалуйста, введите имя пользователя!")
            return
            
        if username not in self.users:
            messagebox.showerror("Ошибка", "Пользователь не найден!")
            return
            
        self.is_capturing = True
        self.successful_recognitions = 0
        self.status_label.config(text="Авторизация в процессе...")
        
        # Load the trained model
        model_path = self.users[username]['model_path']
        if not os.path.exists(model_path):
            messagebox.showerror("Ошибка", "Модель не найдена!")
            return
            
        self.recognizer = LBPHFaceRecognizer_create()
        self.recognizer.read(model_path)
        
        # Start recognition thread
        threading.Thread(target=self.authenticate_face, daemon=True).start()
        
    def authenticate_face(self):
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        while self.is_capturing and not self.is_authenticated:
            ret, img = self.cam.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(10, 10))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
                
                confidence_value = 100 - confidence
                if confidence_value >= 70:
                    self.successful_recognitions += 1
                    confidence_str = f"{confidence_value:.2f}%"
                    
                    # Update status
                    self.status_label.config(text=f"{self.successful_recognitions}/10")
                    
                    if self.successful_recognitions >= 10:
                        self.is_authenticated = True
                        messagebox.showinfo("Успех", "Авторизация успешно завершена!")
                        self.show_main_interface()
                else:
                    confidence_str = f"{confidence_value:.2f}%"
                
                cv2.putText(img, f"{self.successful_recognitions}/10", (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, confidence_str, (x + 5, y + h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
            
            self.update_video_frame(img)
            
        self.cam.release()
        
    def show_main_interface(self):
        self.main_frame = ttk.Frame(self.root, style='Main.TFrame', padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
        username = self.auth_username.get()
        self.users[username]['last_login'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.users[username]['total_logins'] += 1
        self.save_users()
        
        welcome_label = ttk.Label(self.main_frame, 
                                text=f"Добро пожаловать, {username}!",
                                style='Title.TLabel')
        welcome_label.grid(row=0, column=0, pady=20)
        
        project_info_frame = ttk.LabelFrame(self.main_frame,
                                          text="Информация о проекте",
                                          style='Custom.TLabelframe',
                                          padding="15")
        project_info_frame.grid(row=1, column=0, pady=10, sticky="nsew")
        
        project_info = """
Система авторизации по лицу - это современное решение для безопасной аутентификации пользователей.

Основные возможности:
- Регистрация новых пользователей с помощью распознавания лиц
- Авторизация зарегистрированных пользователей
- Хранение истории входов и статистики использования
- Высокая точность распознавания лиц
- Удобный и интуитивно понятный интерфейс

Технологии:
- OpenCV для обработки изображений
- LBPH (Local Binary Pattern Histogram) для распознавания лиц
- Tkinter для создания графического интерфейса
- JSON для хранения данных пользователей
        """
        
        ttk.Label(project_info_frame,
                 text=project_info,
                 style='Info.TLabel').grid(row=0, column=0, pady=5, sticky="w")
        
        stats_frame = ttk.LabelFrame(self.main_frame,
                                   text="Статистика системы",
                                   style='Custom.TLabelframe',
                                   padding="15")
        stats_frame.grid(row=2, column=0, pady=10, sticky="nsew")
        
        total_users = len(self.users)
        total_models = sum(1 for user in self.users.values() if os.path.exists(user['model_path']))
        total_datasets = sum(1 for user in self.users if os.path.exists(f"dataset_{user}"))
        
        user_reg_date = self.users[username]['registration_date']
        user_last_login = self.users[username]['last_login']
        user_total_logins = self.users[username]['total_logins']
        
        stats = [
            f"Всего зарегистрированных пользователей: {total_users}",
            f"Всего обученных моделей: {total_models}",
            f"Всего наборов данных: {total_datasets}",
            "",
            f"Ваша регистрация: {user_reg_date}",
            f"Последний вход: {user_last_login}",
            f"Всего успешных входов: {user_total_logins}"
        ]
        
        for i, stat in enumerate(stats):
            ttk.Label(stats_frame,
                     text=stat,
                     style='Info.TLabel').grid(row=i, column=0, pady=5, sticky="w")
        
        logout_button = ttk.Button(self.main_frame, 
                                 text="Выйти из системы",
                                 command=self.logout,
                                 style='Custom.TButton')
        logout_button.grid(row=3, column=0, pady=20)
        
    def logout(self):
        self.is_authenticated = False
        self.successful_recognitions = 0
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        title_label = ttk.Label(self.main_frame, 
                              text="Система Авторизации по Лицу",
                              style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        self.video_frame = ttk.Label(self.main_frame, borderwidth=2, relief="solid")
        self.video_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="nsew")
        
        self.create_control_buttons()
        
        self.status_label = ttk.Label(self.main_frame, 
                                    text="Система готова к работе",
                                    style='Status.TLabel')
        self.status_label.grid(row=3, column=0, columnspan=2, pady=10)
        
    def update_video_frame(self, img):
        # Convert OpenCV image to PIL format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        
        # Update label
        self.video_frame.imgtk = img
        self.video_frame.configure(image=img)
        
    def __del__(self):
        if self.cam is not None:
            self.cam.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAuthApp(root)
    root.mainloop() 