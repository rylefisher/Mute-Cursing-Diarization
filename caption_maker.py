import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import subprocess
import threading
import os
import sys
import tempfile
import math
import time # Used in srt formatting
import traceback

import sv_ttk # For detailed error logging

# Attempt imports and provide guidance if missing
try:
    import torch
except ImportError:
    messagebox.showerror("Error", "PyTorch not found. Please install it (see PyTorch website).")
    sys.exit(1)

try:
    import whisperx
    import whisperx.vad # Explicitly import vad
except ImportError:
    messagebox.showerror("Error", "WhisperX not found. Please install it (`pip install -U whisperx`). Ensure Faster-Whisper dependencies are met.")
    sys.exit(1)

class WhisperXApp:
    # --- Default Options (Based on user context) ---
    _default_asr_options = {
        "log_prob_threshold": -3.0, # Accept low confidence words
        "no_speech_threshold": 0.1, # Catch quieter speech
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], # Robustness
        "condition_on_previous_text": True, # Context for long audio
        "compression_ratio_threshold": None, # Disable filter
        "word_timestamps": True, # Needed for alignment
        "without_timestamps": False,
        "beam_size": 5, # Balance accuracy/speed
        "best_of": 5, # Balance accuracy/speed
        "patience": None, # Faster decoding
        "length_penalty": 1.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [], # Ensure no tokens suppressed
        "max_initial_timestamp": 1.0,
        "prepend_punctuations": "",
        "append_punctuations": "",
        "suppress_numerals": False,
        "max_new_tokens": None,
        "clip_timestamps": "0",
        "hallucination_silence_threshold": None,
        "hotwords": None, # Could add curses if needed
    }
    _default_vad_options = {
        "vad_threshold": 0.15, # VAD sensitivity
        "min_speech_duration_ms": 50, # Catch short speech
        "max_speech_duration_s": float("inf"),
        "min_silence_duration_ms": 300, # Silence duration sensitivity
        "window_size_samples": 1024, # Default from whisperx example
        "speech_pad_ms": 500, # Padding around speech
    }
    # Note: transcribe_config like chunk_size is often handled internally by load_model/transcribe
    # We'll keep batch_size from the GUI for the model.transcribe call
    _default_align_config = {
        "return_char_alignments": False,  # Not needed for words
    }
    _default_diarize_options = {
        "min_speakers": None,  # Auto-detect
        "max_speakers": None,  # Auto-detect or set upper limit
    }
    # SRT formatting defaults (kept separate)
    _default_srt_options = {
        "max_line_len": 42,
        "max_lines": 2,
    }

    def __init__(self, master):
        self.master = master
        self.master.title("WhisperX SRT Generator")
        self.master.geometry("650x800") # Increased height for VAD options

        # --- Input/Output ---
        self.file_path = tk.StringVar()

        # --- Core Settings ---
        self.model_size = tk.StringVar(value='base')
        self.device = tk.StringVar(value='cuda' if torch.cuda.is_available() else 'cpu')
        self.compute_type = tk.StringVar(value='float16' if torch.cuda.is_available() else 'int8')
        self.batch_size = tk.IntVar(value=16)
        self.language = tk.StringVar(value='') # Empty for auto-detect
        self.hf_token = tk.StringVar(value='') # Hugging Face Token

        # --- Processing Flags ---
        self.align_model_flag = tk.BooleanVar(value=True) # Renamed to avoid conflict
        self.diarize_flag = tk.BooleanVar(value=False) # Renamed

        # --- ASR Options ---
        self.beam_size_var = tk.IntVar(value=self._default_asr_options['beam_size'])
        self.best_of_var = tk.IntVar(value=self._default_asr_options['best_of'])
        self.log_prob_thresh_var = tk.DoubleVar(value=self._default_asr_options['log_prob_threshold'])
        self.no_speech_thresh_var = tk.DoubleVar(value=self._default_asr_options['no_speech_threshold'])
        self.condition_prev_text_var = tk.BooleanVar(value=self._default_asr_options['condition_on_previous_text'])
        self.length_penalty_var = tk.DoubleVar(value=self._default_asr_options['length_penalty'])
        self.suppress_numerals_var = tk.BooleanVar(value=self._default_asr_options['suppress_numerals'])

        # --- VAD Options ---
        self.vad_threshold_var = tk.DoubleVar(value=self._default_vad_options['vad_threshold'])
        self.min_speech_dur_var = tk.IntVar(value=self._default_vad_options['min_speech_duration_ms'])
        self.min_silence_dur_var = tk.IntVar(value=self._default_vad_options['min_silence_duration_ms'])
        self.speech_pad_var = tk.IntVar(value=self._default_vad_options['speech_pad_ms'])

        # --- Diarize Options ---
        self.min_speakers_var = tk.StringVar(value=str(self._default_diarize_options['min_speakers'] or '')) # Store as string for Entry
        self.max_speakers_var = tk.StringVar(value=str(self._default_diarize_options['max_speakers'] or '')) # Store as string for Entry

        # --- SRT Options ---
        self.srt_max_len_var = tk.IntVar(value=self._default_srt_options['max_line_len'])
        self.srt_max_lines_var = tk.IntVar(value=self._default_srt_options['max_lines'])

        # --- Status/Progress ---
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0)

        # --- Model Placeholders ---
        self.vad_model = None
        self.whisper_model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None

        # --- Runtime Merged Options (populated before processing) ---
        self.asr_options = {}
        self.vad_options = {}
        self.align_config = {}
        self.diarize_options = {}
        self.srt_options = {} # For SRT formatting

        self._create_widgets()
        self._check_ffmpeg()


    def _check_ffmpeg(self):
        # Checks ffmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
            self._update_status("FFmpeg found.", clear_after=5000)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self._update_status("FFmpeg not found! Please install FFmpeg and ensure it's in your PATH.", is_error=True)
            messagebox.showerror("Error", "FFmpeg not found. Please install FFmpeg and ensure it's in your system's PATH.")
            # Don't disable run button here, do it in _set_ui_state
            # self.run_button.config(state=tk.DISABLED)


    def _create_widgets(self):
        # Main Frame
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- File Selection ---
        file_frame = ttk.LabelFrame(main_frame, text="Input File", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="Video/Audio:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.file_path, width=50).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(file_frame, text="Browse...", command=self._select_file).grid(row=0, column=2, padx=5, pady=5)

        # --- Model & Compute Options ---
        model_compute_frame = ttk.LabelFrame(main_frame, text="Model & Compute Settings", padding="10")
        model_compute_frame.pack(fill=tk.X, pady=5)
        model_compute_frame.columnconfigure(1, weight=1)
        model_compute_frame.columnconfigure(3, weight=1)

        ttk.Label(model_compute_frame, text="Model Size:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        model_combo = ttk.Combobox(model_compute_frame, textvariable=self.model_size,
                                   values=['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'], state="readonly")
        model_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(model_compute_frame, text="Device:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        device_opts = ['cpu']
        if torch.cuda.is_available():
            device_opts.insert(0, 'cuda') # Prioritize cuda
        device_combo = ttk.Combobox(model_compute_frame, textvariable=self.device, values=device_opts, state="readonly")
        device_combo.grid(row=0, column=3, padx=5, pady=5, sticky=tk.EW)
        device_combo.bind("<<ComboboxSelected>>", self._update_compute_type_options)

        ttk.Label(model_compute_frame, text="Compute Type:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.compute_type_combo = ttk.Combobox(model_compute_frame, textvariable=self.compute_type, state="readonly")
        self.compute_type_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        self._update_compute_type_options()

        ttk.Label(model_compute_frame, text="Batch Size:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Spinbox(model_compute_frame, from_=1, to=128, textvariable=self.batch_size, width=10).grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)

        ttk.Label(model_compute_frame, text="Language:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(model_compute_frame, textvariable=self.language).grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(model_compute_frame, text="(blank=auto)").grid(row=2, column=2, columnspan=2, padx=5, pady=5, sticky=tk.W)

        ttk.Label(model_compute_frame, text="HF Token:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(model_compute_frame, textvariable=self.hf_token, show="*").grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(model_compute_frame, text="(optional, needed for diarize)").grid(row=3, column=2, columnspan=2, padx=5, pady=5, sticky=tk.W)


        # --- Processing Options ---
        process_frame = ttk.LabelFrame(main_frame, text="Processing Stages", padding="10")
        process_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(process_frame, text="Align Transcription (better timing)", variable=self.align_model_flag).grid(row=0, column=0, columnspan=2, padx=5, pady=2, sticky=tk.W)
        ttk.Checkbutton(process_frame, text="Diarize Speakers (needs align & HF token)", variable=self.diarize_flag).grid(row=1, column=0, columnspan=2, padx=5, pady=2, sticky=tk.W)

        # --- VAD Options ---
        vad_frame = ttk.LabelFrame(main_frame, text="VAD (Voice Activity Detection)", padding="10")
        vad_frame.pack(fill=tk.X, pady=5)
        vad_frame.columnconfigure(1, weight=1)
        vad_frame.columnconfigure(3, weight=1)

        ttk.Label(vad_frame, text="VAD Threshold:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(vad_frame, textvariable=self.vad_threshold_var, width=10).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(vad_frame, text="(0-1, lower=more sensitive)").grid(row=0, column=2, columnspan=2, padx=5, pady=2, sticky=tk.W)

        ttk.Label(vad_frame, text="Min Speech (ms):").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(vad_frame, from_=10, to=10000, textvariable=self.min_speech_dur_var, width=7).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(vad_frame, text="Min Silence (ms):").grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(vad_frame, from_=10, to=10000, textvariable=self.min_silence_dur_var, width=7).grid(row=1, column=3, padx=5, pady=2, sticky=tk.W)

        ttk.Label(vad_frame, text="Speech Pad (ms):").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(vad_frame, from_=0, to=5000, textvariable=self.speech_pad_var, width=7).grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)


        # --- ASR Advanced Options ---
        asr_frame = ttk.LabelFrame(main_frame, text="ASR Fine-tuning", padding="10")
        asr_frame.pack(fill=tk.X, pady=5)
        asr_frame.columnconfigure(1, weight=1)
        asr_frame.columnconfigure(3, weight=1)

        ttk.Label(asr_frame, text="Beam Size:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(asr_frame, from_=1, to=50, textvariable=self.beam_size_var, width=7).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(asr_frame, text="Best Of:").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(asr_frame, from_=1, to=50, textvariable=self.best_of_var, width=7).grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)

        ttk.Label(asr_frame, text="Log Prob Thresh:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(asr_frame, textvariable=self.log_prob_thresh_var, width=10).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(asr_frame, text="No Speech Thresh:").grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(asr_frame, textvariable=self.no_speech_thresh_var, width=10).grid(row=1, column=3, padx=5, pady=2, sticky=tk.W)

        ttk.Label(asr_frame, text="Length Penalty:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(asr_frame, textvariable=self.length_penalty_var, width=10).grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Checkbutton(asr_frame, text="Cond. on Prev Text", variable=self.condition_prev_text_var).grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Checkbutton(asr_frame, text="Suppress Numerals", variable=self.suppress_numerals_var).grid(row=3, column=2, padx=5, pady=2, sticky=tk.W)

        # --- Diarize Options Frame ---
        diarize_frame = ttk.LabelFrame(main_frame, text="Diarization Options", padding="10")
        diarize_frame.pack(fill=tk.X, pady=5)
        diarize_frame.columnconfigure(1, weight=1)
        diarize_frame.columnconfigure(3, weight=1)

        ttk.Label(diarize_frame, text="Min Speakers:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(diarize_frame, textvariable=self.min_speakers_var, width=10).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(diarize_frame, text="(blank=auto)").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)

        ttk.Label(diarize_frame, text="Max Speakers:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(diarize_frame, textvariable=self.max_speakers_var, width=10).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(diarize_frame, text="(blank=auto)").grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)


        # --- SRT Formatting Options ---
        srt_frame = ttk.LabelFrame(main_frame, text="SRT Formatting", padding="10")
        srt_frame.pack(fill=tk.X, pady=5)
        srt_frame.columnconfigure(1, weight=1)
        srt_frame.columnconfigure(3, weight=1)

        ttk.Label(srt_frame, text="Max Line Length:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Spinbox(srt_frame, from_=10, to=200, textvariable=self.srt_max_len_var, width=7).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(srt_frame, text="Max Lines/Caption:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Spinbox(srt_frame, from_=1, to=10, textvariable=self.srt_max_lines_var, width=7).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)

        # --- Controls & Status ---
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        self.run_button = ttk.Button(control_frame, text="Generate SRT", command=self._run_transcription)
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5,0))

    def _update_compute_type_options(self, event=None):
        # Update compute type based on device
        selected_device = self.device.get()
        if selected_device == 'cuda':
            self.compute_type_combo['values'] = ['float16', 'int8_float16', 'int8']
            if self.compute_type.get() not in self.compute_type_combo['values']:
                self.compute_type.set('float16')
        else: # cpu
            self.compute_type_combo['values'] = ['int8', 'float32']
            if self.compute_type.get() not in self.compute_type_combo['values']:
                self.compute_type.set('int8')

    def _select_file(self):
        # Opens file dialog
        filepath = filedialog.askopenfilename(
            title="Select Media File",
            filetypes=(("All Media Files", "*.*"),
                       ("Video Files", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                       ("Audio Files", "*.mp3 *.wav *.ogg *.flac *.m4a"))
        )
        if filepath:
            self.file_path.set(filepath)
            self._update_status(f"Selected: {os.path.basename(filepath)}", clear_after=5000)

    def _update_status(self, message, progress=None, is_error=False, clear_after=None):
        # Updates status bar - now safe to call from `after` with positional args
        self.status_var.set(message)
        if is_error:
            self.status_label.config(foreground="red")
        else:
            self.status_label.config(foreground="black") # Reset color on non-error

        if progress is not None:
            self.progress_var.set(progress)
        self.master.update_idletasks() # Ensure UI updates immediately

        if clear_after:
             # Use lambda to capture current message for comparison
             current_msg = message
             self.master.after(clear_after, lambda: self.status_var.set("Ready") if self.status_var.get() == current_msg else None)


    def _set_ui_state(self, state):
        # Disables/enables gui
        # Iterate safely over children, handling potential errors
        for child in self.master.winfo_children():
            self._recursive_set_state(child, state)

        # Special handling for the Run button
        if state == tk.DISABLED:
            self.run_button.config(text="Running...", state=tk.DISABLED)
        else:
            # Check ffmpeg again before enabling run button fully
            try:
                subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True)
                self.run_button.config(text="Generate SRT", state=tk.NORMAL)
            except (subprocess.CalledProcessError, FileNotFoundError):
                 self.run_button.config(text="Generate SRT", state=tk.DISABLED) # Keep disabled if ffmpeg missing
                 self._update_status("FFmpeg not found! Cannot run.", is_error=True) # Keep error visible


    def _recursive_set_state(self, widget, state):
        # Helper for _set_ui_state
        try:
            # Check if the widget has a 'state' option
            if 'state' in widget.configure():
                widget.configure(state=state)

            # Recurse into container widgets
            for child in widget.winfo_children():
                self._recursive_set_state(child, state)
        except tk.TclError: # Ignore widgets without 'state' or already destroyed
            pass
        except AttributeError: # Ignore widgets without winfo_children (e.g., scrollbars)
            pass


    def _run_transcription(self):
        # Starts process thread
        input_file = self.file_path.get()
        if not input_file or not os.path.exists(input_file):
            messagebox.showerror("Error", "Please select a valid input file.")
            return

        # --- Merge GUI Options with Defaults ---
        try:
            # ASR
            gui_asr_options = {
                "beam_size": self.beam_size_var.get(),
                "best_of": self.best_of_var.get(),
                "log_prob_threshold": self.log_prob_thresh_var.get(),
                "no_speech_threshold": self.no_speech_thresh_var.get(),
                "condition_on_previous_text": self.condition_prev_text_var.get(),
                "length_penalty": self.length_penalty_var.get(),
                "suppress_numerals": self.suppress_numerals_var.get(),
                "word_timestamps": True, # Always enable for align/diarize
            }
            self.asr_options = {**self._default_asr_options, **gui_asr_options}

            # VAD
            gui_vad_options = {
                "vad_threshold": self.vad_threshold_var.get(),
                "min_speech_duration_ms": self.min_speech_dur_var.get(),
                "min_silence_duration_ms": self.min_silence_dur_var.get(),
                "speech_pad_ms": self.speech_pad_var.get(),
            }
            self.vad_options = {**self._default_vad_options, **gui_vad_options}

            # Align (not many options here)
            self.align_config = {**self._default_align_config}

            # Diarize
            min_spk = self.min_speakers_var.get().strip() # Strip whitespace
            max_spk = self.max_speakers_var.get().strip() # Strip whitespace
            gui_diarize_options = {
                 "min_speakers": int(min_spk) if min_spk.isdigit() else None,
                 "max_speakers": int(max_spk) if max_spk.isdigit() else None,
            }
            self.diarize_options = {**self._default_diarize_options, **gui_diarize_options}

            # SRT Formatting
            gui_srt_options = {
                "max_line_len": self.srt_max_len_var.get(),
                "max_lines": self.srt_max_lines_var.get(),
            }
            self.srt_options = {**self._default_srt_options, **gui_srt_options}

        except tk.TclError as e: # Catch errors getting values (e.g., invalid float)
             messagebox.showerror("Invalid Option", f"Please check your numeric option values: {e}")
             return
        except ValueError as e: # Catch int conversion errors for speakers
             messagebox.showerror("Invalid Option", f"Speaker counts must be numbers (or blank): {e}")
             return

        # --- Start Processing ---
        self._set_ui_state(tk.DISABLED)
        self._update_status("Starting...", 0)

        self.processing_thread = threading.Thread(target=self._processing_task, args=(input_file,), daemon=True)
        self.processing_thread.start()


    def _load_models(self, load_diarization: bool):
        """Load VAD, Whisper, and optionally Diarization models lazily."""
        # Note: Alignment model loading is moved post-transcription if lang is auto-detect
        current_progress = 20 # Starting progress after audio loading
        specified_language = self.language.get() or None # Get lang if user specified

        try:
            # --- Load VAD Model (always needed for whisperx.load_model) ---
            if self.vad_model is None:
                self.master.after(0, self._update_status, "Loading VAD model...", current_progress)
                self.vad_model = whisperx.vad.load_vad_model(self.device.get())
                current_progress += 3
                self.master.after(0, self._update_status, "VAD model loaded.", current_progress)

            # --- Load Whisper Model ---
            if self.whisper_model is None:
                model_name = self.model_size.get()
                self.master.after(0, self._update_status, f"Loading Whisper model: {model_name}...", current_progress)

                effective_asr_options = {k: v for k, v in self.asr_options.items() if v is not None}
                effective_vad_options = {k: v for k, v in self.vad_options.items() if v is not None}

                self.whisper_model = whisperx.load_model(
                    model_name,
                    self.device.get(),
                    compute_type=self.compute_type.get(),
                    language=specified_language, # Pass specified lang (or None)
                    asr_options=effective_asr_options,
                    vad_options=effective_vad_options,
                    vad_model=self.vad_model,
                    task="transcribe",
                )
                current_progress += 10
                self.master.after(0, self._update_status, "Whisper model loaded.", current_progress)

            # --- Load Diarization Model (if needed) ---
            if load_diarization and self.diarize_model is None:
                token = self.hf_token.get()
                if not token:
                    # Don't raise error, just warn and skip
                    self.master.after(0, self._update_status, "Warning: Diarization requires HF token. Skipping diarization model load.", current_progress, False, 5000) # is_error=False
                    # No progress increase here
                else:
                    self.master.after(0, self._update_status, "Loading diarization model...", current_progress)
                    try:
                        self.diarize_model = whisperx.DiarizationPipeline(
                            use_auth_token=token, device=self.device.get()
                        )
                        current_progress += 10 # Increase progress if successful
                        self.master.after(0, self._update_status, "Diarization model loaded.", current_progress)
                    except Exception as e:
                        # Log error, warn user, skip diarization
                        print(f"ERROR loading diarization model: {e}\n{traceback.format_exc()}")
                        self.master.after(0, self._update_status, f"Error loading diarization model: {e}. Diarization skipped.", current_progress, True, 5000) # is_error=True
                        self.diarize_model = None # Ensure it's None

            return current_progress # Return progress after loading these models

        except Exception as e:
            # Propagate model loading errors (VAD/Whisper)
            raise RuntimeError(f"Core model loading failed: {e}") from e


    def _format_timestamp(self, seconds: float) -> str:
        # Formats seconds to SRT time
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)
        hours = milliseconds // 3_600_000
        milliseconds %= 3_600_000
        minutes = milliseconds // 60_000
        milliseconds %= 60_000
        seconds = milliseconds // 1_000
        milliseconds %= 1_000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _generate_srt(self, result_segments) -> str:
        # Generates srt string using runtime srt_options
        srt_content = ""
        count = 1
        max_line_len = self.srt_options.get("max_line_len", 42)
        max_lines = self.srt_options.get("max_lines", 2)

        for segment in result_segments:
            # Ensure start/end exist, skip segment if not (shouldn't happen with whisperx)
            if 'start' not in segment or 'end' not in segment:
                print(f"Warning: Skipping segment without start/end times: {segment.get('text', '')[:50]}...")
                continue

            start_time_fmt = self._format_timestamp(segment['start'])
            end_time_fmt = self._format_timestamp(segment['end'])
            text = segment.get('text', '').strip()
            speaker = f"Speaker {segment['speaker']}: " if 'speaker' in segment else ""

            words = text.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = f"{current_line} {word}".strip()
                if len(test_line) <= max_line_len:
                    current_line = test_line
                else:
                    if current_line: lines.append(current_line)
                    # Handle words longer than max_line_len (split them)
                    if len(word) > max_line_len:
                        # Simple split, might break words mid-syllable
                        lines.append(word[:max_line_len])
                        current_line = word[max_line_len:]
                    else:
                        current_line = word
            if current_line: lines.append(current_line)

            num_lines = len(lines)
            for i in range(0, num_lines, max_lines):
                chunk_lines = lines[i : i + max_lines]
                if not chunk_lines: continue

                srt_content += f"{count}\n"
                srt_content += f"{start_time_fmt} --> {end_time_fmt}\n"
                # Add speaker prefix only to the first line of the chunk
                srt_content += speaker + chunk_lines[0] + "\n"
                for line in chunk_lines[1:]:
                     srt_content += line + "\n"
                srt_content += "\n" # Blank line
                count += 1
        return srt_content

    def _processing_task(self, input_file):
        # Main processing logic using merged options and lazy loading
        temp_audio_file = None
        current_progress = 0
        result = None # Ensure result is defined for finally block cleanup checks
        audio = None # Ensure audio is defined

        try:
            output_basename = os.path.splitext(input_file)[0]
            output_srt_file = f"{output_basename}.srt"
            do_align = self.align_model_flag.get()
            do_diarize = self.diarize_flag.get()

            # --- 1. Extract Audio (if needed) ---
            self.master.after(0, self._update_status, "Extracting audio...", 5)
            is_audio = input_file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))
            audio_path = input_file
            if not is_audio:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    temp_audio_file = tmpfile.name
                audio_path = temp_audio_file
                ffmpeg_command = ["ffmpeg", "-i", input_file, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", audio_path]
                try:
                    subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0) # Hide console on windows
                except subprocess.CalledProcessError as e:
                     raise RuntimeError(f"FFmpeg Error: {e.stderr or e.stdout or 'Unknown'}") from e
                except FileNotFoundError:
                    raise RuntimeError("FFmpeg not found.") from None

            current_progress = 15
            self.master.after(0, self._update_status, "Loading audio data...", current_progress)
            audio = whisperx.load_audio(audio_path) # Load the audio file


            # --- 2. Load VAD, Whisper, and maybe Diarization Models ---
            # Note: Alignment model is loaded later if needed & lang is auto
            needs_diarize_model = do_diarize # Determine if diarization model load is attempted
            current_progress = self._load_models(load_diarization=needs_diarize_model)
            # self._load_models updates progress internally

            # --- 3. Transcribe ---
            self.master.after(0, self._update_status, "Transcribing audio...", current_progress)
            try:
                # Use the loaded whisper_model and its internal asr/vad options
                result = self.whisper_model.transcribe(
                    audio,
                    batch_size=self.batch_size.get() # Use GUI batch size
                )

            except Exception as e:
                 raise RuntimeError(f"Transcription failed: {e}") from e

            # --- Determine Language Code for Alignment ---
            # Use specified language, otherwise use detected language from result
            language_code = self.language.get() or result.get("language")
            if not language_code:
                 # If language still unknown, alignment cannot proceed
                 print("Warning: Language code unknown after transcription. Cannot perform alignment.")
                 do_align = False # Disable alignment if language is missing
            else:
                # Ensure language is set in GUI if auto-detected for clarity
                 if not self.language.get():
                     self.master.after(0, self.language.set, language_code) # Update GUI var
                     print(f"Detected language: {language_code}")


            current_progress = 60 # Approx progress after transcription
            self.master.after(0, self._update_status, "Transcription complete.", current_progress)


            # --- 4. Load Alignment Model (if needed and lang known) ---
            if do_align and self.align_model is None: # Only load if needed and not already loaded
                self.master.after(0, self._update_status, f"Loading alignment model ({language_code})...", current_progress + 2)
                try:
                    self.align_model, self.align_metadata = whisperx.load_align_model(
                        language_code=language_code, device=self.device.get()
                    )
                    current_progress += 7
                    self.master.after(0, self._update_status, "Alignment model loaded.", current_progress)
                except Exception as e:
                     # If specific lang model not found, align_model remains None
                     print(f"Warning: Could not load alignment model for '{language_code}': {e}. Alignment will be skipped.")
                     self.master.after(0, self._update_status, f"Warning: Alignment model for '{language_code}' not found. Skipping alignment.", current_progress, False, 5000)
                     do_align = False # Disable alignment step


            # --- 5. Align (if enabled and model loaded) ---
            if do_align: # Re-check flag in case loading failed
                if self.align_model and self.align_metadata:
                    self.master.after(0, self._update_status, "Aligning transcription...", current_progress + 2)
                    try:
                        result = whisperx.align(result["segments"], self.align_model, self.align_metadata, audio, self.device.get(), return_char_alignments=self.align_config["return_char_alignments"])
                        current_progress = 80 # Update progress after successful alignment
                        self.master.after(0, self._update_status, "Alignment complete.", current_progress)
                    except Exception as e:
                        print(f"Warning: Alignment process failed: {e}. Using unaligned results.")
                        self.master.after(0, self._update_status, f"Alignment failed: {e}. Using unaligned.", current_progress + 5)
                        # Ensure 'result' still has 'segments' key, might just be list from transcribe
                        if isinstance(result, dict) and 'segments' in result:
                            pass # Already has segments structure
                        elif isinstance(result, list):
                            result = {'segments': result} # Wrap list in dict
                        else:
                            # This case should ideally not happen if transcribe worked
                            raise RuntimeError("Alignment failed and transcription result format is unexpected.") from e
                else:
                     # This case should be covered by the loading step warning
                     pass # Already warned if model didn't load

            # --- 6. Diarize (if enabled and model loaded) ---
            # Make sure alignment was intended OR transcription provides timestamps
            # Diarization needs segments with 'start' and 'end'
            can_diarize = do_diarize and self.diarize_model and isinstance(result, dict) and 'segments' in result and result['segments'] and all('start' in x and 'end' in x for x in result['segments'])

            if do_diarize: # Check user intent first
                if not do_align and not self.align_model_flag.get(): # Warn if user disabled align but wants diarize
                     self.master.after(0, self._update_status, "Warning: Diarization works best with alignment enabled.", current_progress + 1, False, 5000)

                if not self.diarize_model:
                    self.master.after(0, self._update_status, "Skipping diarization (model not loaded).", current_progress + 1)
                elif not can_diarize:
                     self.master.after(0, self._update_status, "Warning: Cannot diarize (segments missing or lack timestamps). Skipping.", current_progress + 1)
                else:
                    # Proceed with diarization
                    self.master.after(0, self._update_status, "Performing diarization...", current_progress + 2)
                    try:
                        diarize_segments = self.diarize_model(audio, min_speakers=self.diarize_options['min_speakers'], max_speakers=self.diarize_options['max_speakers'])
                        self.master.after(0, self._update_status, "Assigning speakers...", current_progress + 4)

                        # Check if word timestamps are available for speaker assignment
                        # WhisperX alignment output puts word timestamps inside 'segments' -> 'words'
                        # Check if the 'words' key exists in the segments from the result dict
                        has_word_timestamps = all('words' in seg for seg in result.get('segments', []))

                        if has_word_timestamps:
                            result = whisperx.assign_word_speakers(diarize_segments, result)
                            current_progress = 90 # Update progress
                            self.master.after(0, self._update_status, "Diarization complete.", current_progress)
                        else:
                            # Fallback or warning if word timestamps aren't present
                            print("Warning: Word-level timestamps not found in segments. Speaker assignment might be less accurate or skipped.")
                            self.master.after(0, self._update_status, "Warning: Word timestamps missing. Skipping speaker assignment.", current_progress + 5, False, 5000)
                            # Optionally, could try segment-level assignment if available in whisperx, but assign_word_speakers is standard

                    except Exception as e:
                        print(f"ERROR during diarization: {e}\n{traceback.format_exc()}")
                        self.master.after(0, self._update_status, f"Diarization failed: {e}. Skipping speakers.", current_progress + 5, True)


            # --- 7. Generate SRT ---
            current_progress = 95 # Progress before final save
            self.master.after(0, self._update_status, "Generating SRT file...", current_progress)
            if not isinstance(result, dict) or 'segments' not in result or not result['segments']:
                 raise ValueError("No valid segments found in the result to generate SRT.")
            srt_content = self._generate_srt(result["segments"])


            # --- 8. Save SRT ---
            with open(output_srt_file, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            self.master.after(0, self._update_status, f"SRT saved: {output_srt_file}", 100)

            # --- 9. Success ---
            self.master.after(0, messagebox.showinfo, "Success", f"SRT file generated successfully:\n{output_srt_file}")

        except Exception as e:
            # --- Error Handling ---
            error_message = f"Error: {str(e)}"
            print(f"ERROR: {error_message}\n{traceback.format_exc()}") # Log full traceback
            # Use positional args for `after` callback: message, progress, is_error, clear_after
            self.master.after(0, self._update_status, error_message, None, True, None)
            self.master.after(0, messagebox.showerror, "Error", error_message)

        finally:
            # --- Cleanup ---
            self.master.after(0, self._update_status, "Cleaning up...", 99) # Indicate cleanup phase
            # Unload models safely
            models_to_del = {
                'whisper': self.whisper_model,
                'align': self.align_model,
                'diarize': self.diarize_model,
                'vad': self.vad_model
            }
            for name, model in models_to_del.items():
                if model:
                    print(f"Cleaning up {name} model...")
                    try:
                        del model
                    except Exception as del_e:
                        print(f"Note: Error during {name} model deletion: {del_e}")

            # Reset placeholders
            self.whisper_model = None
            self.align_model = None
            self.align_metadata = None
            self.diarize_model = None
            self.vad_model = None
            result = None # Clear result reference
            audio = None # Clear audio data reference


            if self.device.get() == 'cuda':
                try:
                    print("Clearing CUDA cache...")
                    torch.cuda.empty_cache()
                except Exception as cache_e:
                     print(f"Note: Error during CUDA cache clearing: {cache_e}")

            # Delete temporary audio file
            if temp_audio_file and os.path.exists(temp_audio_file):
                try:
                    print(f"Removing temporary audio file: {temp_audio_file}")
                    os.remove(temp_audio_file)
                except OSError as e:
                    print(f"Warning: Could not delete temp file {temp_audio_file}: {e}")

            # Re-enable UI
            self.master.after(0, self._set_ui_state, tk.NORMAL)
            # Reset status unless an error occurred and is still displayed
            final_status = self.status_var.get()
            is_error_displayed = self.status_label.cget("foreground") == "red"
            if not is_error_displayed or "Success" in final_status: # Clear if success or non-error state
                 self.master.after(100, self._update_status, "Finished.", 0) # Reset progress and status


if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperXApp(root)
    sv_ttk.set_theme("dark")
    root.mainloop()
