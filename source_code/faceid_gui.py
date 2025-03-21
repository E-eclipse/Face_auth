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

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("800x600")
        
        # Variables
        self.face_id = tk.StringVar()
        self.is_capturing = False
        self.count = 0
        self.last_print_time = time.time()
        self.recognizer = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        # Create main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create video frame
        self.video_frame = ttk.Label(self.main_frame)
        self.video_frame.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Create control buttons
        self.create_control_buttons()
        
        # Create status label
        self.status_label = ttk.Label(self.main_frame, text="Ready", font=('Arial', 12))
        self.status_label.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Initialize camera
        self.cam = None
        
    def create_control_buttons(self):
        # Training section
        training_frame = ttk.LabelFrame(self.main_frame, text="Training", padding="5")
        training_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        ttk.Label(training_frame, text="Enter name:").grid(row=0, column=0, padx=5)
        ttk.Entry(training_frame, textvariable=self.face_id).grid(row=0, column=1, padx=5)
        ttk.Button(training_frame, text="Start Training", command=self.start_training).grid(row=0, column=2, padx=5)
        
        # Recognition section
        recognition_frame = ttk.LabelFrame(self.main_frame, text="Recognition", padding="5")
        recognition_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        ttk.Button(recognition_frame, text="Start Recognition", command=self.start_recognition).grid(row=0, column=0, padx=5)
        ttk.Button(recognition_frame, text="Stop Recognition", command=self.stop_recognition).grid(row=0, column=1, padx=5)
        
    def start_training(self):
        if not self.face_id.get():
            messagebox.showerror("Error", "Please enter a name first!")
            return
            
        if not re.match("^[a-zA-Z]+$", self.face_id.get()):
            messagebox.showerror("Error", "Please use only English letters!")
            return
            
        self.is_capturing = True
        self.count = 0
        self.status_label.config(text="Training in progress...")
        
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
                self.status_label.config(text=f"{self.count}/200 images")
                
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
        
        self.status_label.config(text="Training completed!")
        messagebox.showinfo("Success", "Face model trained successfully!")
        
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
        
    def start_recognition(self):
        if not self.face_id.get():
            messagebox.showerror("Error", "Please train the model first!")
            return
            
        self.is_capturing = True
        self.status_label.config(text="Recognition in progress...")
        
        # Load the trained model
        model_path = os.path.join(os.getcwd(), f"{self.face_id.get()}_model/face_{self.face_id.get()}.yml")
        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Model not found! Please train first.")
            return
            
        self.recognizer = LBPHFaceRecognizer_create()
        self.recognizer.read(model_path)
        
        # Start recognition thread
        threading.Thread(target=self.recognize_face, daemon=True).start()
        
    def recognize_face(self):
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        while self.is_capturing:
            ret, img = self.cam.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(10, 10))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
                
                confidence_value = 100 - confidence
                if confidence_value >= 60:
                    name = self.face_id.get()
                    confidence_str = f"{confidence_value:.2f}%"
                    
                    # Update status every 10 seconds
                    current_time = time.time()
                    if current_time - self.last_print_time >= 10:
                        self.status_label.config(text=f"Recognized: {name} ({confidence_str})")
                        self.last_print_time = current_time
                else:
                    name = "Unknown"
                    confidence_str = f"{confidence_value:.2f}%"
                
                cv2.putText(img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, confidence_str, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
            
            self.update_video_frame(img)
            
        self.cam.release()
        
    def stop_recognition(self):
        self.is_capturing = False
        self.status_label.config(text="Recognition stopped")
        
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
    app = FaceRecognitionApp(root)
    root.mainloop() 