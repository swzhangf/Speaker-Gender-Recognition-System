import os
import threading
import time
import numpy as np
import librosa
import librosa.display
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# UI Libraries
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

# ==========================================
# Logic Class (Data Processing & Model)
# ==========================================
class GenderModel:
    def __init__(self):
        self.model = GaussianNB()
        self.is_trained = False
        self.accuracy = 0.0
        self.report = ""
        self.classes = ["Male", "Female"]

    def pre_emphasis(self, signal, alpha=0.97):
        """
        Apply a pre-emphasis filter to the signal to amplify high frequencies.
        H(z) = 1 - alpha * z^-1
        """
        if len(signal) < 2: return signal
        return np.append(signal[0], signal[1:] - alpha * signal[:-1])

    def extract_features(self, file_path):
        """
        Extract MFCC features from an audio file.
        Returns: Average MFCC vector (for classification), Raw Signal, Sample Rate, and Full MFCC Matrix (for visualization).
        """
        try:
            # Load audio (Resample to 16kHz)
            y, sr = librosa.load(file_path, sr=16000)
            
            # Apply Pre-emphasis
            y_pre = self.pre_emphasis(y)
            
            # Extract MFCCs (13 coefficients)
            mfccs = librosa.feature.mfcc(y=y_pre, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
            
            # Calculate mean across time (for Naive Bayes input)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            
            return mfccs_mean, y, sr, mfccs
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None, None, None, None

    def train(self, male_dir, female_dir, progress_callback=None):
        """
        Load all Audio from directories, extract features, and train the Naive Bayes classifier.
        """
        features = []
        labels = []
        
        # Scan directories for Audio files
        if not os.path.exists(male_dir) or not os.path.exists(female_dir):
            raise FileNotFoundError("One or both sample directories do not exist.")

        male_files = [os.path.join(male_dir, f) for f in os.listdir(male_dir) if f.endswith('.wav')]
        female_files = [os.path.join(female_dir, f) for f in os.listdir(female_dir) if f.endswith('.wav')]
        
        total_files = len(male_files) + len(female_files)
        if total_files == 0: 
            raise ValueError("No Audio files found in the specified folders.")
        
        count = 0
        
        # Process Male Samples (Label 0)
        for f in male_files:
            feat, _, _, _ = self.extract_features(f)
            if feat is not None:
                features.append(feat)
                labels.append(0)
            count += 1
            if progress_callback: progress_callback(count, total_files)

        # Process Female Samples (Label 1)
        for f in female_files:
            feat, _, _, _ = self.extract_features(f)
            if feat is not None:
                features.append(feat)
                labels.append(1)
            count += 1
            if progress_callback: progress_callback(count, total_files)

        X = np.array(features)
        y = np.array(labels)

        # Split Data (80% Train, 20% Test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Model
        self.model.fit(X_train, y_train)
        
        # Evaluate Performance
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.report = classification_report(y_test, y_pred, target_names=self.classes)
        self.is_trained = True
        
        return self.accuracy, self.report

    def predict(self, file_path):
        """
        Predict gender for a single file.
        """
        if not self.is_trained: 
            raise Exception("Model is not trained yet.")
        
        feat, y, sr, mfccs = self.extract_features(file_path)
        if feat is None: 
            raise Exception("Failed to process audio file.")
        
        # Predict class
        pred_idx = self.model.predict([feat])[0]
        # Get confidence probabilities
        probs = self.model.predict_proba([feat])[0]
        
        return self.classes[pred_idx], probs, y, sr, mfccs

# ==========================================
# Modern UI Class (View)
# ==========================================
class GenderAppPro(ttk.Window):
    def __init__(self):
        # Initialize with a modern theme ('litera', 'cosmo', or 'flatly')
        super().__init__(themename="litera")
        self.title("Advanced Speaker Gender Recognition")
        self.geometry("1100x750")
        
        self.model = GenderModel()
        self.current_audio_path = None
        
        # Initialize Audio Player
        pygame.mixer.init()
        
        # Default Paths (Pre-filled for convenience)
        self.path_male = tk.StringVar(value=r"")
        self.path_female = tk.StringVar(value=r"")

        self.setup_ui()

    def setup_ui(self):
        # --- Header Section ---
        header = ttk.Frame(self, padding=20, bootstyle="primary")
        header.pack(fill=X)
        ttk.Label(header, text="Gender Recognition by Voice", font=("Helvetica", 20, "bold"), bootstyle="inverse-primary").pack(side=LEFT)
        ttk.Label(header, text="MFCC + Naive Bayes Classifier", font=("Helvetica", 12), bootstyle="inverse-primary").pack(side=RIGHT, pady=5)

        # --- Main Layout (Left: Controls, Right: Visualization) ---
        main_pane = ttk.Panedwindow(self, orient=HORIZONTAL)
        main_pane.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # === Left Panel: Controls ===
        left_frame = ttk.Frame(main_pane, padding=10)
        main_pane.add(left_frame, weight=1)

        # 1. Training Section
        train_group = ttk.Labelframe(left_frame, text="1. Model Training", padding=15)
        train_group.pack(fill=X, pady=10)
        
        ttk.Label(train_group, text="Male Clips Folder:").pack(anchor=W)
        ttk.Entry(train_group, textvariable=self.path_male).pack(fill=X, pady=(0, 10))
        
        ttk.Label(train_group, text="Female Clips Folder:").pack(anchor=W)
        ttk.Entry(train_group, textvariable=self.path_female).pack(fill=X, pady=(0, 10))
        
        self.btn_train = ttk.Button(train_group, text="Train Model", command=self.start_training, bootstyle="success")
        self.btn_train.pack(fill=X, pady=5)
        
        self.progress = ttk.Progressbar(train_group, bootstyle="striped-success")
        self.progress.pack(fill=X, pady=5)
        
        self.lbl_status = ttk.Label(train_group, text="Status: Model not trained", font=("Arial", 9), bootstyle="secondary")
        self.lbl_status.pack(anchor=W)

        # 2. Prediction Section
        pred_group = ttk.Labelframe(left_frame, text="2. Recognition & Test", padding=15)
        pred_group.pack(fill=X, pady=10)
        
        btn_row = ttk.Frame(pred_group)
        btn_row.pack(fill=X)
        
        self.btn_load = ttk.Button(btn_row, text="Load Audio", command=self.load_audio, bootstyle="primary-outline", state=DISABLED)
        self.btn_load.pack(side=LEFT, fill=X, expand=True, padx=(0, 5))
        
        self.btn_play = ttk.Button(btn_row, text="▶ Play", command=self.play_audio, bootstyle="info-outline", state=DISABLED, width=8)
        self.btn_play.pack(side=RIGHT)

        # Result Display Area
        self.res_frame = ttk.Frame(pred_group, padding=10, bootstyle="secondary")
        self.res_frame.pack(fill=X, pady=15)
        
        ttk.Label(self.res_frame, text="Prediction:", font=("Arial", 10), bootstyle="inverse-secondary").pack(anchor=W)
        self.lbl_result = ttk.Label(self.res_frame, text="?", font=("Helvetica", 24, "bold"), anchor=CENTER, background="#e9ecef")
        self.lbl_result.pack(fill=X, pady=5)
        
        # Confidence Meters
        self.meter_male = ttk.Floodgauge(pred_group, bootstyle="info", text="Male Confidence", value=0, mask="{}%")
        self.meter_male.pack(fill=X, pady=5)
        
        self.meter_female = ttk.Floodgauge(pred_group, bootstyle="danger", text="Female Confidence", value=0, mask="{}%")
        self.meter_female.pack(fill=X, pady=5)

        # === Right Panel: Visualization ===
        right_frame = ttk.Frame(main_pane, padding=10)
        main_pane.add(right_frame, weight=3)
        
        ttk.Label(right_frame, text="Signal Analysis", font=("Arial", 14, "bold")).pack(anchor=W, pady=5)
        
        # Matplotlib Figure Setup
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [1, 2]})
        self.fig.tight_layout(pad=3.0)
        
        # Initial empty plots
        self.ax1.set_title("Waveform")
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])
        self.ax2.set_title("MFCC Spectrogram")
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)



    def start_training(self):
        m_path = self.path_male.get()
        f_path = self.path_female.get()
        
        if not os.path.exists(m_path) or not os.path.exists(f_path):
            messagebox.showerror("Path Error", "One or both sample paths do not exist.\nPlease check the folder paths.")
            return
        
        self.btn_train.config(state=DISABLED)
        self.lbl_status.config(text="Processing... Extracting features...", bootstyle="warning")
        
        # Run training in a separate thread to keep UI responsive
        threading.Thread(target=self.run_train_process, args=(m_path, f_path)).start()

    def run_train_process(self, m, f):
        try:
            acc, report = self.model.train(m, f, self.update_progress)
            # Schedule UI update on main thread
            self.after(0, lambda: self.finish_train(acc, report))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Training Error", str(e)))
            self.after(0, lambda: self.reset_ui_after_error())

    def update_progress(self, current, total):
        val = (current / total) * 100
        # Schedule progress bar update
        self.after(0, lambda: self.progress.configure(value=val))

    def finish_train(self, acc, report):
        self.lbl_status.config(text=f"Training Complete! Test Accuracy: {acc:.1%}", bootstyle="success")
        self.btn_train.config(state=NORMAL)
        self.btn_load.config(state=NORMAL)
        
        # Show detailed popup
        msg = f"Model Accuracy: {acc:.2%}\n\nClassification Report:\n{report}"
        messagebox.showinfo("Training Results", msg)

    def reset_ui_after_error(self):
        self.btn_train.config(state=NORMAL)
        self.lbl_status.config(text="Status: Error during training", bootstyle="danger")

    def load_audio(self):
        path = filedialog.askopenfilename(title="Select Audio File", filetypes=[("Audio Files", "*.mp3 *.wav")])
        if not path: return
        
        self.current_audio_path = path
        self.btn_play.config(state=NORMAL)
        self.lbl_result.config(text="Analyzing...", bootstyle="secondary")
        
        # Run prediction in thread
        threading.Thread(target=self.run_prediction, args=(path,)).start()

    def run_prediction(self, path):
        try:
            res_text, probs, y, sr, mfccs = self.model.predict(path)
            self.after(0, lambda: self.update_result_ui(res_text, probs, y, sr, mfccs))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Prediction Error", str(e)))
            self.after(0, lambda: self.lbl_result.config(text="Error"))

    def update_result_ui(self, label, probs, y, sr, mfccs):
        # 1. Update Result Text Color (Blue for Male, Red for Female)
        color = "info" if label == "Male" else "danger"
        self.lbl_result.config(text=label, bootstyle=f"{color}-inverse")

        # 2. Update Progress Bars
        male_p = probs[0] * 100
        female_p = probs[1] * 100
        
        self.meter_male.configure(value=male_p, text=f"Male: {male_p:.1f}%")
        self.meter_female.configure(value=female_p, text=f"Female: {female_p:.1f}%")

        # 3. Update Plots
        self.ax1.clear()
        self.ax2.clear()

        # Top: Waveform
        librosa.display.waveshow(y, sr=sr, ax=self.ax1, alpha=0.6, color='#007bff')
        self.ax1.set_title("Waveform (Time Domain)", fontsize=10)
        self.ax1.set_ylabel("Amplitude")
        self.ax1.set_xticks([]) 

        # Bottom: MFCC Spectrogram
        img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=self.ax2, cmap='coolwarm')
        self.ax2.set_title("MFCC Features (Frequency Domain)", fontsize=10)
        self.ax2.set_ylabel("MFCC Coefficients")
        self.ax2.set_xlabel("Time (s)")

        self.canvas.draw()

    def play_audio(self):
        if self.current_audio_path:
            try:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                
                pygame.mixer.music.load(self.current_audio_path)
                pygame.mixer.music.play()
            except Exception as e:
                messagebox.showerror("Playback Error", f"Could not play audio.\nError: {e}")

if __name__ == "__main__":
    app = GenderAppPro()
    app.mainloop()