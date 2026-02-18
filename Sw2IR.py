import sys
import os
import shutil
import time
import subprocess
import soundfile as sf
import traceback
import numpy as np
import scipy.signal as signal
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QLineEdit, QPushButton, QListWidget, QFileDialog, 
                               QTextEdit, QFrame, QGroupBox, QDoubleSpinBox, QCheckBox, QStyle, QAbstractItemView)
from PySide6.QtCore import Qt, QThread, Signal, QMimeData
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QLinearGradient, QColor, QPalette, QBrush, QFont

def add_scalefactor_to_wav(wav_file_path, scalefactor, output_file_path):
    """
    Adds a 'scalefactor=...' comment metadata tag to a WAV file using FFmpeg.
    """
    # ffmpeg command
    ffmpeg_command = [
        'ffmpeg', '-y', '-i', wav_file_path,
        '-metadata', f'comment=scalefactor={scalefactor}',
        '-codec', 'copy',
        output_file_path
    ]
    
    # Run silently
    try:
        subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        raise e

class ProcessingThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, ref_path, out_path, duration, align, do_norm, sweep_files):
        super().__init__()
        self.ref_path = ref_path
        self.out_path = out_path
        self.duration = duration
        self.align = align
        self.do_norm = do_norm
        self.sweep_files = sweep_files
        self.is_running = True

    def run(self):
        if not os.path.exists(self.ref_path):
            self.log_signal.emit(f"Error: Reference sweep not found: {self.ref_path}")
            return
        
        self.log_signal.emit(f"Starting Processing...")
        self.log_signal.emit(f"Processing {len(self.sweep_files)} files with {self.duration}s duration")
        self.log_signal.emit(f"Alignment: {'Enabled (-10ms)' if self.align else 'Disabled'}")
        self.log_signal.emit(f"Normalizing: {'Enabled' if self.do_norm else 'Disabled (Raw)'}")
        
        # Read Ref
        try:
            sig_or, fs_sweep = sf.read(self.ref_path)
            if sig_or.ndim > 1: sig_or = sig_or[:, 0]
            
            ref_len = len(sig_or)
            trim_seconds = (ref_len / fs_sweep) + 5.0
            trim_samples = int(trim_samples_calc := trim_seconds * fs_sweep)
            
            # Create a temp folder for intermediate files
            temp_dir = os.path.join(self.out_path, "sw2ir_temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
             
        except Exception as e:
            self.log_signal.emit(f"Error reading reference: {e}")
            return

        for idx, wav_file in enumerate(self.sweep_files):
            if not self.is_running: break
            try:
                self.log_signal.emit(f"Processing ({idx+1}/{len(self.sweep_files)}): {os.path.basename(wav_file)}")
                
                # 1. Read Measurement
                sig, fs = sf.read(wav_file)
                if fs != fs_sweep:
                    self.log_signal.emit(f"  Warning: Rate mismatch {fs} vs {fs_sweep}")
                
                if sig.ndim > 1: sig = sig[:, 0] # Mono

                if len(sig) > trim_samples:
                    sig = sig[:trim_samples]

                # 2. Deconvolution
                n_fft = 2 ** int(np.ceil(np.log2(len(sig_or) + len(sig))))
                
                fft_chirp = np.fft.fft(sig_or, n_fft)
                fft_resp = np.fft.fft(sig, n_fft)
                
                fft_chirp_safe = np.where(fft_chirp == 0, 1e-10, fft_chirp)
                ir_full = np.real(np.fft.ifft(fft_resp / fft_chirp_safe))

                # 3. Alignment
                if self.align:
                    peak_idx = np.argmax(np.abs(ir_full))
                    offset_samples = int(0.010 * fs)
                    start_idx = peak_idx - offset_samples
                    
                    if start_idx < 0:
                        start_idx = 0
                        self.log_signal.emit(f"  Warning: Peak too close to start")
                    
                    ir_full = np.roll(ir_full, -start_idx)
                
                # 4. Crop
                out_samples = int(self.duration * fs)
                if len(ir_full) > out_samples:
                    ir_full = ir_full[:out_samples]
                else:
                    ir_full = np.pad(ir_full, (0, out_samples - len(ir_full)))

                # 5. Normalize (Optional)
                peak_val = np.max(np.abs(ir_full))
                scalefactor = 1.0
                ir_final = ir_full

                if self.do_norm and peak_val > 0:
                    scalefactor = peak_val * 32768 / 32767 
                    ir_final = ir_full / scalefactor
                else:
                    scalefactor = 1.0 # Raw
                    ir_final = ir_full

                # 6. Save
                base_name = os.path.splitext(os.path.basename(wav_file))[0]
                align_tag = "_aligned" if self.align else ""
                norm_tag = "" if self.do_norm else "_raw" 
                out_filename = f"IR_{base_name}{align_tag}{norm_tag}.wav"
                
                # Paths
                final_path = os.path.join(self.out_path, out_filename)
                temp_path = os.path.join(temp_dir, f"temp_{out_filename}")
                
                # Write to temp first
                sf.write(temp_path, ir_final, fs, subtype='PCM_32')
                
                # Add Metadata using ffmpeg (temp -> final)
                try:
                    add_scalefactor_to_wav(temp_path, f'{scalefactor:.8f}', final_path)
                except Exception as e:
                    self.log_signal.emit(f"  Warning: FFmpeg failed (metadata missing).")
                    shutil.move(temp_path, final_path)

                self.log_signal.emit(f"  -> Saved: {out_filename}")
                
            except Exception as e:
                self.log_signal.emit(f"  Failed: {e}")
                traceback.print_exc()

        # Cleanup
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
            
        self.finished_signal.emit()

class DropLineEdit(QLineEdit):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setPlaceholderText("Drag Reference Sweep Here...")
        self.setStyleSheet("") # Use parent/global style

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if os.path.isfile(file_path): 
                self.setText(file_path)

class DropListWidget(QListWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # Style controlled by global stylesheet for consistency

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragEnterEvent):
        # Crucial for QListWidget to accept drops continuously during hover
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            for url in urls:
                file_path = url.toLocalFile()
                if os.path.isfile(file_path) and file_path.lower().endswith('.wav'):
                    items = [self.item(i).text() for i in range(self.count())]
                    if file_path not in items:
                        self.addItem(file_path)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

class Sw2IR(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sw2IR (Sweep to IR)")
        self.resize(600, 750)
        self.processing_thread = None

        # Styles
        self.STYLE_VIBRANT = """
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #232526, stop:1 #414345);
            }
            QLabel {
                color: #e0e0e0;
                font-weight: bold;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 14px;
            }
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 8px;
                margin-top: 20px;
                background-color: rgba(0, 0, 0, 0.2);
                font-weight: bold;
                color: #e0e0e0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                left: 10px;
                color: #cccccc;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #777777;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
            QLineEdit {
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 6px;
                background-color: #2b2b2b;
                color: #e0e0e0;
                selection-background-color: #555555;
            }
            QListWidget {
                border: 1px solid #555555;
                border-radius: 8px;
                background-color: #2b2b2b;
                color: #e0e0e0;
                font-size: 13px;
            }
            QListWidget::item:selected {
                background-color: #4a4a4a;
                color: #ffffff;
                border: 1px solid #777777;
            }
            QCheckBox {
                color: #e0e0e0;
                font-weight: bold;
                font-size: 13px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #555555;
                background: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                background: #4a4a4a;
                border: 1px solid #e0e0e0;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMSA2TDUgMTBMMTEgMSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIi8+PC9zdmc+);
            }
            QDoubleSpinBox {
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 4px;
                background-color: #2b2b2b;
                color: #e0e0e0;
            }
        """

        self.STYLE_UNICORN = """
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #FFC0CB, stop:0.5 #E0FFFF, stop:1 #DDA0DD);
            }
            QLabel {
                color: #4B0082;
                font-weight: bold;
                font-size: 14px;
            }
            QGroupBox {
                border: 3px solid #FF1493;
                border-radius: 15px;
                margin-top: 20px;
                font-weight: bold;
                color: #C71585;
                background-color: rgba(255, 255, 255, 0.8);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                color: #FF1493;
            }
            QPushButton {
                background-color: #FF69B4;
                color: white;
                border-radius: 15px;
                padding: 10px 20px;
                font-weight: bold;
                border: 3px solid #C71585;
            }
            QPushButton:hover {
                background-color: #FF1493;
            }
            QPushButton:pressed {
                background-color: #C71585;
            }
            QLineEdit {
                border: 3px dashed #FF69B4;
                border-radius: 10px;
                padding: 5px;
                background-color: #FFF0F5;
                color: #C71585;
            }
            QListWidget {
                border: 3px solid #9370DB;
                border-radius: 10px;
                background-color: #E6E6FA;
                color: #4B0082;
            }
            QListWidget::item:selected {
                background-color: #FF69B4;
                color: white;
            }
            QCheckBox {
                color: #C71585;
                font-weight: bold;
                font-size: 14px;
            }
            QDoubleSpinBox {
                border: 2px solid #FF69B4;
                border-radius: 5px;
                padding: 2px;
                background-color: white;
            }
        """

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Header Container
        header_layout = QVBoxLayout()
        
        self.header_label = QLabel("Sw2IR")
        self.header_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.header_label)
        
        self.subheader_label = QLabel("By Luna")
        self.subheader_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.subheader_label)

        # Magic Toggle (Top Rightish)
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        self.magic_toggle = QCheckBox("🦄 Magic Mode")
        self.magic_toggle.setCursor(Qt.PointingHandCursor)
        self.magic_toggle.stateChanged.connect(self.toggle_style)
        top_bar.addWidget(self.magic_toggle)
        top_bar.addStretch() # Center it? Or right align? Let's Center the toggle under title
        
        layout.addLayout(header_layout)
        layout.addLayout(top_bar)

        # Ref Sweep
        ref_group = QGroupBox("Reference Sweep")
        ref_layout = QHBoxLayout()
        self.ref_input = DropLineEdit()
        self.ref_input.setPlaceholderText("Drag Reference Sweep Here...")
        self.ref_input.setStyleSheet("") # Reset override
        ref_layout.addWidget(self.ref_input)
        ref_btn = QPushButton("Browse")
        ref_btn.clicked.connect(self.browse_ref)
        ref_layout.addWidget(ref_btn)
        ref_group.setLayout(ref_layout)
        layout.addWidget(ref_group)

        # Output Folder
        out_group = QGroupBox("Output Folder")
        out_layout = QHBoxLayout()
        self.out_input = QLineEdit()
        self.out_input.setReadOnly(True) 
        self.out_input.setPlaceholderText("Select Output Folder...")
        self.out_input.setStyleSheet("") # Reset override
        out_layout.addWidget(self.out_input)
        out_btn = QPushButton("Browse")
        out_btn.clicked.connect(self.browse_out)
        out_layout.addWidget(out_btn)
        out_group.setLayout(out_layout)
        layout.addWidget(out_group)

        # Sweeps List
        list_group = QGroupBox("Input Sweeps")
        list_layout = QVBoxLayout()
        self.list_widget = DropListWidget()
        list_layout.addWidget(self.list_widget)
        
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Files")
        add_btn.clicked.connect(self.add_sweeps)
        clear_btn = QPushButton("Clear")
        # Custom red-ish gradient for clear
        clear_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #eb3349, stop:1 #f45c43);
                border: 1px solid rgba(255,255,255,0.5);
            }
            QPushButton:hover {
                 background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f45c43, stop:1 #eb3349);
            }
        """)
        clear_btn.clicked.connect(self.clear_list)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_selected)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addWidget(clear_btn)
        list_layout.addLayout(btn_layout)
        
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)

        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("IR Duration (s):"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setValue(3.0)
        self.duration_spin.setRange(0.1, 30.0)
        # self.duration_spin.setStyleSheet("") # Use global
        settings_layout.addWidget(self.duration_spin)
        
        self.align_check = QCheckBox("Align IR", self)
        self.align_check.setToolTip("Aligns the Direct Sound to 10ms")
        self.align_check.setChecked(False)
        # self.align_check.setStyleSheet("") # Use global
        settings_layout.addWidget(self.align_check)

        self.norm_check = QCheckBox("Normalize", self)
        self.norm_check.setToolTip("Peak normalize the output")
        self.norm_check.setChecked(True)
        # self.norm_check.setStyleSheet("") # Use global
        settings_layout.addWidget(self.norm_check)

        settings_layout.addStretch()
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Process Button
        self.process_btn = QPushButton("Process Sweeps")
        self.process_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #11998e, stop:1 #38ef7d);
                color: white;
                font-size: 18px;
                font-weight: 900;
                padding: 15px;
                border-radius: 12px;
                border: 2px solid rgba(255,255,255,0.7);
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #38ef7d, stop:1 #11998e);
            }
            QPushButton:pressed {
                background-color: #0b6e4f;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.process_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.process_btn)

        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            border: 1px solid #bcccdc;
            border-radius: 6px;
            background-color: #f0f4f8;
            color: #102a43;
            font-family: "Menlo", "Consolas", monospace;
            font-size: 11px;
        """)
        layout.addWidget(self.log_text)

        # Apply Initial Style
        self.update_style(is_magic=False)

    def browse_ref(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Reference Sweep", "", "WAV Files (*.wav)")
        if f:
            self.ref_input.setText(f)
            self.log(f"Selected Reference: {f}")

    def browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d:
            self.out_input.setText(d)
            self.log(f"Selected Output: {d}")

    def add_sweeps(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Sweeps", "", "WAV Files (*.wav)")
        for f in files:
            items = [self.list_widget.item(i).text() for i in range(self.list_widget.count())]
            if f not in items:
                self.list_widget.addItem(f)
        if files:
            self.log(f"Added {len(files)} files.")

    def clear_list(self):
        self.list_widget.clear()
        self.log("Cleared sweep list.")

    def remove_selected(self):
        for item in self.list_widget.selectedItems():
            self.list_widget.takeItem(self.list_widget.row(item))

    def log(self, text):
        self.log_text.append(text)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def start_processing(self):
        ref = self.ref_input.text()
        out = self.out_input.text()
        dur = self.duration_spin.value()
        align = self.align_check.isChecked()
        norm = self.norm_check.isChecked()
        
        sweeps = [self.list_widget.item(i).text() for i in range(self.list_widget.count())]

        if not ref or not os.path.exists(ref):
             self.log("❌ Error: Invalid Reference Sweep.")
             return
        if not out:
             self.log("❌ Error: Please select an output folder.")
             return
        if not sweeps:
             self.log("❌ Error: No sweeps to process.")
             return

        self.process_btn.setEnabled(False)
        self.process_btn.setText("Processing...")
        
        self.processing_thread = ProcessingThread(ref, out, dur, align, norm, sweeps)
        self.processing_thread.log_signal.connect(self.log)
        self.processing_thread.finished_signal.connect(self.processing_finished)
        self.processing_thread.start()

    def processing_finished(self):
        self.process_btn.setEnabled(True)
        # Restore text based on mode
        if self.magic_toggle.isChecked():
            self.process_btn.setText("✨ MAKE IT HAPPEN! ✨")
        else:
            self.process_btn.setText("PROCESS SWEEPS")
        self.log("Done.")

    def toggle_style(self):
        self.update_style(self.magic_toggle.isChecked())

    def update_style(self, is_magic):
        if is_magic:
            self.setStyleSheet(self.STYLE_UNICORN)
            self.setWindowTitle("Sw2IR 🌈🦄✨")
            self.header_label.setText("🌈 Sw2IR: The Magical Deconvolver 🦄")
            self.header_label.setStyleSheet("font-size: 24px; color: #FF1493; margin: 10px;")
            self.subheader_label.setText("✨ By Luna ✨")
            self.subheader_label.setStyleSheet("font-size: 14px; color: #C71585; margin-bottom: 10px; font-style: italic;")
            self.process_btn.setText("✨ MAKE IT HAPPEN! ✨")
        else:
            self.setStyleSheet(self.STYLE_VIBRANT)
            self.setWindowTitle("Sw2IR")
            self.header_label.setText("Sw2IR")
            self.header_label.setStyleSheet("font-size: 32px; color: #e0e0e0; margin-top: 15px; font-weight: 800; letter-spacing: 2px;")
            self.subheader_label.setText("By Luna")
            self.subheader_label.setStyleSheet("font-size: 14px; color: #aaaaaa; margin-bottom: 20px; font-style: italic;")
            self.process_btn.setText("PROCESS SWEEPS")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Enable High DPI
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)

    window = Sw2IR()
    window.show()
    sys.exit(app.exec())
