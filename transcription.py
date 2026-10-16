import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pygame

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Transcription Interface")
        self.root.geometry("600x550")
        
        # Initialize audio mixer
        pygame.mixer.init()
        
        # State variables
        self.input_folder = ""
        self.output_folder = ""
        self.wav_files = []
        self.current_index = 0
        self.current_sequence = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # --- Folder Selection ---
        folder_frame = tk.Frame(self.root)
        folder_frame.pack(pady=10)
        
        tk.Button(folder_frame, text="Select Input Folder", command=self.select_input_folder).grid(row=0, column=0, padx=5)
        self.lbl_input = tk.Label(folder_frame, text="No input selected")
        self.lbl_input.grid(row=0, column=1, sticky="w")
        
        tk.Button(folder_frame, text="Select Output Folder", command=self.select_output_folder).grid(row=1, column=0, padx=5, pady=5)
        self.lbl_output = tk.Label(folder_frame, text="No output selected")
        self.lbl_output.grid(row=1, column=1, sticky="w")
        
        # --- File Status ---
        self.lbl_status = tk.Label(self.root, text="File: -- (0/0)", font=("Helvetica", 12, "bold"))
        self.lbl_status.pack(pady=10)
        
        # --- Playback & Navigation ---
        nav_frame = tk.Frame(self.root)
        nav_frame.pack(pady=10)
        
        tk.Button(nav_frame, text="Previous", command=self.prev_file, width=10).grid(row=0, column=0, padx=10)
        tk.Button(nav_frame, text="Play / Replay", command=self.play_audio, width=15, bg="lightgreen").grid(row=0, column=1, padx=10)
        tk.Button(nav_frame, text="Next", command=self.next_file, width=10).grid(row=0, column=2, padx=10)
        
        # --- Annotation Buttons ---
        tk.Label(self.root, text="Annotations", font=("Helvetica", 10, "italic")).pack(pady=(15, 5))
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        
        labels = ["roll", "tom", "bak", "beshcan"]
        for i, label in enumerate(labels):
            btn = tk.Button(btn_frame, text=label.upper(), width=10, height=2,
                            command=lambda l=label: self.add_annotation(l))
            btn.grid(row=0, column=i, padx=5)
            
        # --- Sequence Display ---
        tk.Label(self.root, text="Current Sequence:", font=("Helvetica", 10)).pack(pady=(20, 0))
        
        self.lbl_sequence = tk.Label(self.root, text="", font=("Helvetica", 12, "bold"), fg="blue", wraplength=550)
        self.lbl_sequence.pack(pady=10)
        
        # --- Backspace ---
        tk.Button(self.root, text="⌫ Backspace", command=self.backspace, width=15, bg="lightcoral").pack(pady=5)
        
        # --- Save Button ---
        tk.Button(self.root, text="Save Current Sequence", command=self.save_sequence, width=25).pack(pady=20)

    def select_input_folder(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_folder = folder
            self.lbl_input.config(text=self.input_folder)
            self.load_wav_files()

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.lbl_output.config(text=self.output_folder)

    def load_wav_files(self):
        if not self.input_folder:
            return
        
        self.wav_files = [f for f in os.listdir(self.input_folder) if f.lower().endswith(".wav")]
        self.wav_files.sort()
        
        if not self.wav_files:
            messagebox.showwarning("Warning", "No .wav files found in the selected input folder.")
            return
            
        self.current_index = 0
        self.load_current_file()

    def load_current_file(self):
        if not self.wav_files:
            return
            
        filename = self.wav_files[self.current_index]
        self.lbl_status.config(text=f"File: {filename} ({self.current_index + 1}/{len(self.wav_files)})")
        
        # Reset sequence for the new file (or load existing if you prefer, but here we reset)
        self.current_sequence = []
        self.update_sequence_display()
        
        # Load audio
        filepath = os.path.join(self.input_folder, filename)
        try:
            pygame.mixer.music.load(filepath)
            self.play_audio()
        except Exception as e:
            messagebox.showerror("Audio Error", f"Could not load audio: {e}")

    def play_audio(self):
        if self.wav_files:
            pygame.mixer.music.play()

    def add_annotation(self, label):
        self.current_sequence.append(label)
        self.update_sequence_display()

    def backspace(self):
        if self.current_sequence:
            self.current_sequence.pop()
            self.update_sequence_display()

    def update_sequence_display(self):
        display_text = " -> ".join(self.current_sequence)
        if not display_text:
            display_text = "(Empty)"
        self.lbl_sequence.config(text=display_text)

    def save_sequence(self):
        if not self.wav_files:
            return
        if not self.output_folder:
            messagebox.showwarning("Warning", "Please select an output folder first.")
            return
            
        filename = self.wav_files[self.current_index]
        lab_filename = os.path.splitext(filename)[0] + ".lab"
        lab_filepath = os.path.join(self.output_folder, lab_filename)
        
        try:
            with open(lab_filepath, "w") as f:
                for item in self.current_sequence:
                    f.write(f"-1 -1 {item}\n")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save file: {e}")

    def next_file(self):
        if self.current_index < len(self.wav_files) - 1:
            self.save_sequence() # Auto-save before moving
            self.current_index += 1
            self.load_current_file()
        else:
            self.save_sequence()
            messagebox.showinfo("Done", "You have reached the end of the folder.")

    def prev_file(self):
        if self.current_index > 0:
            self.save_sequence() # Auto-save before moving
            self.current_index -= 1
            self.load_current_file()

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()