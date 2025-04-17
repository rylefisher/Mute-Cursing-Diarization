import tkinter as tk
from tkinter import (
    ttk,
    filedialog,
    scrolledtext,
    messagebox,
)  
import subprocess
import threading
import os
import tempfile
import sys
import platform
import torch

# Check if silero_vad is installed, provide helpful message if not
try:
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
except ImportError:
    print("Error: 'silero-vad' package not found.")
    print("Please install it using: pip install -q silero-vad")
    sys.exit(1)
import logging
import queue
import re  # For potential future ffmpeg progress parsing

# --- Constants ---
SAMPLING_RATE = 16000
APP_NAME = "Video Silence Remover (Silero VAD)"
FFMPEG_COMMAND = "ffmpeg"  # Assume in PATH
DEFAULT_OUTPUT_PLACEHOLDER = "output.mp4"  # Placeholder for comparison

# --- Logging Setup ---
log_queue = queue.Queue()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class QueueHandler(logging.Handler):
    """Sends log records to a queue."""

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


logger = logging.getLogger(__name__)
# Avoid adding handler multiple times if script is re-run in some environments
if not any(isinstance(h, QueueHandler) for h in logger.handlers):
    logger.addHandler(QueueHandler(log_queue))
logger.setLevel(logging.INFO)  # Ensure logger level is appropriate

# --- Helper Functions ---


def check_ffmpeg():
    """Check if ffmpeg exists."""
    try:
        # Use shorter timeout to avoid long hangs if ffmpeg is broken
        subprocess.run(
            [FFMPEG_COMMAND, "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=get_startup_info(),
            timeout=5,
        )
        logger.info("FFmpeg found.")
        return True
    except FileNotFoundError:
        logger.error(
            "FFmpeg command not found. Please install FFmpeg and ensure it's in your system's PATH."
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg command timed out. Check FFmpeg installation.")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(
            f"FFmpeg found but '-version' failed: {e.stderr.decode(errors='ignore')}"
        )
        # Allow continuing, maybe other commands work
        return True
    except Exception as e:
        logger.error(f"Error checking FFmpeg: {e}")
        return False


def get_startup_info():
    """Hide console window for subprocess on Windows."""
    if platform.system() == "Windows":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        # Ensure handles are not inherited (good practice)
        startupinfo.dwFlags |= subprocess.STARTF_USESTDHANDLES
        # Prevent console window pop-up for subprocess.run/Popen
        CREATE_NO_WINDOW = 0x08000000
        startupinfo.dwFlags |= CREATE_NO_WINDOW  # Requires python 3.7+? Check if needed
        return startupinfo
    return None


def merge_overlapping_timestamps(timestamps, buffer_s):
    """Merge overlapping/adjacent speech segments after applying buffer."""
    if not timestamps:
        return []

    # Apply buffer
    buffered_timestamps = [
        {"start": max(0, ts["start"] - buffer_s), "end": ts["end"] + buffer_s}
        for ts in timestamps
    ]

    # Sort by start time
    buffered_timestamps.sort(key=lambda x: x["start"])

    merged = []
    if not buffered_timestamps:
        return merged

    current_start, current_end = (
        buffered_timestamps[0]["start"],
        buffered_timestamps[0]["end"],
    )

    for next_ts in buffered_timestamps[1:]:
        next_start, next_end = next_ts["start"], next_ts["end"]
        # Merge if overlapping or adjacent
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            merged.append({"start": current_start, "end": current_end})
            current_start, current_end = next_start, next_end

    merged.append({"start": current_start, "end": current_end})  # Add last segment
    return merged


def format_time(seconds):
    """Convert seconds to HH:MM:SS.ms"""
    try:
        seconds = float(seconds)
        ms = int((seconds - int(seconds)) * 1000)
        s = int(seconds)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    except Exception:
        return "00:00:00.000"  # Fallback


# --- Main Application Class ---


class VadApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("700x600")  # Increased height for progress bar

        # --- Model Loading ---
        self.vad_model = None
        # Defer model loading until needed or after GUI setup for faster startup
        # self.load_vad_model()

        # --- Variables ---
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.buffer_ms = tk.IntVar(value=250)  # Default buffer
        self.codec_choice = tk.StringVar(value="libx264")  # Default CPU codec

        # --- GUI Setup ---
        self._create_widgets()
        self.is_processing = False
        self.temp_dir = None  # For temporary files

        # --- Check Dependencies & Load Model ---
        # Perform checks after GUI is visible to avoid blocking startup
        self.root.after(100, self._initialize_app)

        self.check_log_queue()  # Start polling log queue

    def _initialize_app(self):
        """Check dependencies, update codecs, and load VAD model."""
        if not check_ffmpeg():
            messagebox.showerror(
                "Dependency Error",
                "FFmpeg not found or not working. Please install it and add to PATH. Processing is disabled.",
            )
            self.start_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Error (FFmpeg missing)")
        else:
            # Update codec list now that ffmpeg check is done
            self._update_codec_options()

        # Load VAD model (can be done in parallel or after FFmpeg check)
        # Using another 'after' call to allow GUI to remain responsive
        self.status_label.config(text="Status: Loading VAD model...")
        self.root.after(150, self.load_vad_model)  # Load slightly after init

    def load_vad_model(self):
        """Load the Silero VAD model."""
        try:
            self._log("Loading Silero VAD model...")
            self._update_status("Loading VAD model")
            torch.set_num_threads(
                max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
            )  # Use reasonable threads
            self.vad_model = load_silero_vad(onnx=False)  # Using PyTorch version
            self._log("Silero VAD model loaded successfully.")
            self._update_status("Idle")  # Update status once loaded
            # Enable start button only if ffmpeg is also okay
            if check_ffmpeg():  # Re-check ffmpeg status briefly
                self.start_button.config(state=tk.NORMAL)

        except Exception as e:
            logger.exception("Failed to load Silero VAD model.")
            messagebox.showerror(
                "Model Load Error", f"Could not load Silero VAD model: {e}"
            )
            self._update_status("Error (VAD model)")
            # Disable processing if model load fails critically
            if hasattr(self, "start_button"):
                self.start_button.config(state=tk.DISABLED)

    def _create_widgets(self):
        """Create and arrange GUI elements."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Input File ---
        input_frame = ttk.LabelFrame(main_frame, text="Input Video", padding="10")
        input_frame.pack(fill=tk.X, pady=5)

        self.input_entry = ttk.Entry(
            input_frame, textvariable=self.input_file, width=60
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(input_frame, text="Browse...", command=self._select_input).pack(
            side=tk.LEFT
        )

        # --- Output File ---
        output_frame = ttk.LabelFrame(main_frame, text="Output Video", padding="10")
        output_frame.pack(fill=tk.X, pady=5)

        self.output_entry = ttk.Entry(
            output_frame, textvariable=self.output_file, width=60
        )
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(output_frame, text="Browse...", command=self._select_output).pack(
            side=tk.LEFT
        )

        # --- Options ---
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.pack(fill=tk.X, pady=5)

        # Buffer
        buffer_frame = ttk.Frame(options_frame)
        buffer_frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(buffer_frame, text="Speech Buffer (ms):").pack(
            side=tk.LEFT, padx=(0, 10)
        )
        self.buffer_scale = ttk.Scale(
            buffer_frame,
            from_=0,
            to=1000,
            orient=tk.HORIZONTAL,
            variable=self.buffer_ms,
            length=150,
            command=lambda s: self.buffer_ms.set(int(float(s))),
        )
        self.buffer_scale.pack(side=tk.LEFT, padx=5)
        # Validate numeric input for buffer entry
        validate_cmd = (self.root.register(self._validate_numeric), "%P")
        self.buffer_entry = ttk.Entry(
            buffer_frame,
            textvariable=self.buffer_ms,
            width=5,
            validate="key",
            validatecommand=validate_cmd,
        )
        self.buffer_entry.pack(side=tk.LEFT)

        # Codec / Hardware Acceleration
        codec_frame = ttk.Frame(options_frame)
        codec_frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(codec_frame, text="Video Codec:").pack(side=tk.LEFT, padx=(0, 10))
        # Initialize with basic codecs, will be updated later
        self.codec_menu = ttk.Combobox(
            codec_frame,
            textvariable=self.codec_choice,
            values=["libx264"],
            state="readonly",
            width=20,
        )
        self.codec_menu.pack(side=tk.LEFT)

        # --- Progress Bar & Status ---
        progress_frame = ttk.Frame(main_frame, padding="5 0")
        progress_frame.pack(fill=tk.X, pady=5)
        self.status_label = ttk.Label(
            progress_frame, text="Status: Initializing..."
        )  # Initial status
        self.status_label.pack(side=tk.LEFT, padx=(5, 10))
        self.progress_bar = ttk.Progressbar(
            progress_frame, orient=tk.HORIZONTAL, length=300, mode="indeterminate"
        )
        # Pack later when needed self.progress_bar.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)

        # --- Progress / Log ---
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=10, state=tk.DISABLED, wrap=tk.WORD
        )  # Slightly reduced height
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- Start Button ---
        self.start_button = ttk.Button(
            main_frame,
            text="Start Processing",
            command=self._start_processing,
            state=tk.DISABLED,
        )  # Initially disabled
        self.start_button.pack(pady=(10, 5))

    def _validate_numeric(self, P):
        """Validation function for numeric entry fields."""
        if P == "" or P.isdigit():
            return True
        else:
            return False

    def _update_codec_options(self):
        """Dynamically find available ffmpeg codecs (HW accel)."""
        self._log("Checking available FFmpeg encoders...")
        codecs = ["libx264", "libx265"]  # Basic CPU codecs first
        hw_available = {"nvidia": False, "intel_qsv": False, "apple_vt": False}
        preferred_codec = "libx264"  # Default fallback

        try:
            result = subprocess.run(
                [FFMPEG_COMMAND, "-encoders"],
                capture_output=True,
                text=True,
                startupinfo=get_startup_info(),
                timeout=10,
                encoding="utf-8",
                errors="ignore",
            )
            encoders = result.stdout
            # NVIDIA
            if re.search(r"^\s*V.....\s*h264_nvenc", encoders, re.MULTILINE):
                codecs.append("h264_nvenc")
                hw_available["nvidia"] = True
            if re.search(r"^\s*V.....\s*hevc_nvenc", encoders, re.MULTILINE):
                codecs.append("hevc_nvenc")
                hw_available["nvidia"] = True  # Mark nvidia available
                preferred_codec = "hevc_nvenc"  # Prefer newer HW codec

            # Intel QSV
            if re.search(r"^\s*V.....\s*h264_qsv", encoders, re.MULTILINE):
                codecs.append("h264_qsv")
                hw_available["intel_qsv"] = True
                if not hw_available["nvidia"]:
                    preferred_codec = "h264_qsv"  # Prefer HW if no nvidia
            if re.search(r"^\s*V.....\s*hevc_qsv", encoders, re.MULTILINE):
                codecs.append("hevc_qsv")
                hw_available["intel_qsv"] = True
                if not hw_available["nvidia"]:
                    preferred_codec = "hevc_qsv"  # Prefer newer HW if no nvidia

            # Apple VideoToolbox
            if platform.system() == "Darwin":  # Only check on macOS
                if re.search(r"^\s*V.....\s*h264_videotoolbox", encoders, re.MULTILINE):
                    codecs.append("h264_videotoolbox")
                    hw_available["apple_vt"] = True
                    if not hw_available["nvidia"] and not hw_available["intel_qsv"]:
                        preferred_codec = "h264_videotoolbox"
                if re.search(r"^\s*V.....\s*hevc_videotoolbox", encoders, re.MULTILINE):
                    codecs.append("hevc_videotoolbox")
                    hw_available["apple_vt"] = True
                    if not hw_available["nvidia"] and not hw_available["intel_qsv"]:
                        preferred_codec = "hevc_videotoolbox"

            self.codec_menu["values"] = codecs
            # Set a sensible default (prefer HW if available)
            if preferred_codec in codecs:
                self.codec_choice.set(preferred_codec)
            elif codecs:
                self.codec_choice.set(codecs[0])  # Fallback to first available
            self._log(f"Available codecs updated. Default: {self.codec_choice.get()}")

        except subprocess.TimeoutExpired:
            self._log(
                "FFmpeg '-encoders' check timed out. Using basic codecs.",
                level=logging.WARNING,
            )
            self.codec_menu["values"] = ["libx264", "libx265"]
            self.codec_choice.set("libx264")
        except Exception as e:
            self._log(
                f"Could not reliably check for hardware encoders: {e}. Using basic codecs.",
                level=logging.WARNING,
            )
            self.codec_menu["values"] = ["libx264", "libx265"]
            self.codec_choice.set("libx264")

    def _log(self, message, level=logging.INFO):
        """Safely update the log text widget from any thread."""
        # Use logger which uses the queue handler
        if level == logging.ERROR:
            logger.error(message)
        elif level == logging.WARNING:
            logger.warning(message)
        else:
            logger.info(message)

    def _update_status(self, message):
        """Safely update the status label from any thread."""
        # This needs to run in the main thread
        self.root.after(0, lambda: self.status_label.config(text=f"Status: {message}"))

    def check_log_queue(self):
        """Check the queue for log messages and update the text widget."""
        while True:
            try:
                record = log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, record + "\n")
                self.log_text.see(tk.END)  # Scroll to end
                self.log_text.config(state=tk.DISABLED)
        self.root.after(100, self.check_log_queue)  # Poll queue every 100ms

    def _select_input(self):
        """Open file dialog to select input video and suggest output."""
        filetypes = [
            ("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
            ("All files", "*.*"),
        ]
        filepath = filedialog.askopenfilename(
            title="Select Input Video", filetypes=filetypes
        )
        if filepath:
            self.input_file.set(filepath)
            self._log(f"Input file selected: {filepath}")

            # --- Auto-suggest output name (Updated Logic) ---
            try:
                # Get directory, base name, and extension
                input_dir = os.path.dirname(filepath)
                base, ext = os.path.splitext(os.path.basename(filepath))

                # Construct the suggested output filename
                suggested_output_name = f"{base}_edited{ext}"
                suggested_output_path = os.path.join(input_dir, suggested_output_name)

                # Prevent suggesting the exact same path as input
                if suggested_output_path == filepath:
                    suggested_output_name = (
                        f"{base}_edited_1{ext}"  # Add suffix if clash
                    )
                    suggested_output_path = os.path.join(
                        input_dir, suggested_output_name
                    )

                self.output_file.set(suggested_output_path)
                self._log(f"Suggested output file: {suggested_output_path}")

            except Exception as e:
                self._log(
                    f"Error suggesting output filename: {e}", level=logging.WARNING
                )
                # Fallback: clear output or set a default if suggestion fails
                self.output_file.set(
                    ""
                )  # Or set a generic default like 'output_edited.mp4'

    def _select_output(self):
        """Open file dialog to select output video."""
        # Suggest default extension based on chosen codec if possible
        codec = self.codec_choice.get()
        if "265" in codec or "hevc" in codec:
            default_ext = ".mp4"  # Or .mkv is also common for H.265
            filetypes = [
                ("MP4 Video", "*.mp4"),
                ("MKV Video", "*.mkv"),
                ("MOV Video", "*.mov"),
                ("All files", "*.*"),
            ]
        elif "264" in codec:
            default_ext = ".mp4"
            filetypes = [
                ("MP4 Video", "*.mp4"),
                ("MKV Video", "*.mkv"),
                ("MOV Video", "*.mov"),
                ("All files", "*.*"),
            ]
        else:  # Default if codec unknown
            default_ext = ".mp4"
            filetypes = [
                ("MP4 Video", "*.mp4"),
                ("MKV Video", "*.mkv"),
                ("MOV Video", "*.mov"),
                ("All files", "*.*"),
            ]

        # Use the directory of the current output path (or input path if output is empty) as initial dir
        current_output = self.output_file.get()
        if current_output and os.path.dirname(current_output):
            initial_dir = os.path.dirname(current_output)
            initial_file = os.path.basename(current_output)
        elif self.input_file.get() and os.path.dirname(self.input_file.get()):
            initial_dir = os.path.dirname(self.input_file.get())
            initial_file = DEFAULT_OUTPUT_PLACEHOLDER
        else:
            initial_dir = "."
            initial_file = DEFAULT_OUTPUT_PLACEHOLDER

        filepath = filedialog.asksaveasfilename(
            title="Save Output Video As",
            filetypes=filetypes,
            defaultextension=default_ext,
            initialdir=initial_dir,
            initialfile=initial_file,
        )
        if filepath:
            self.output_file.set(filepath)
            self._log(f"Output file set to: {filepath}")

    def _start_processing(self):
        """Validate inputs and start the processing thread."""
        in_path = self.input_file.get()
        out_path = self.output_file.get()
        try:
            buffer_val = self.buffer_ms.get()
        except (
            tk.TclError,
            ValueError,
        ):  # Catch error if entry has non-numeric value during get()
            messagebox.showerror(
                "Error",
                "Invalid buffer value. Please enter a whole number (milliseconds).",
            )
            return

        if not in_path or not os.path.exists(in_path):
            messagebox.showerror("Error", "Please select a valid input video file.")
            return
        if not out_path:
            messagebox.showerror("Error", "Please select an output video file path.")
            return
        if os.path.normpath(in_path) == os.path.normpath(
            out_path
        ):  # Robust path comparison
            messagebox.showerror(
                "Error", "Output file cannot be the same as the input file."
            )
            return
        # Check if output directory exists, create if not? Or rely on ffmpeg? Let's warn.
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir)
                self._log(f"Created output directory: {out_dir}")
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Output directory does not exist and cannot be created:\n{out_dir}\nError: {e}",
                )
                return

        if buffer_val < 0:
            messagebox.showerror("Error", "Buffer must be zero or positive.")
            return
        if self.vad_model is None:
            # This might happen if initialization failed or is slow
            messagebox.showerror(
                "Error", "VAD model not ready. Please wait or check logs."
            )
            return
        if self.is_processing:
            messagebox.showwarning("Busy", "Processing is already in progress.")
            return

        self.is_processing = True
        self.start_button.config(state=tk.DISABLED, text="Processing...")
        # Disable browse buttons during processing
        for widget in self.input_entry.master.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.config(state=tk.DISABLED)
        for widget in self.output_entry.master.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.config(state=tk.DISABLED)

        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)  # Clear previous logs
        self.log_text.config(state=tk.DISABLED)
        self._log(f"--- Starting New Process ---")
        self._log(f"Input: {os.path.basename(in_path)}")
        self._log(f"Output: {os.path.basename(out_path)}")
        self._log(f"Buffer: {buffer_val} ms")
        self._log(f"Codec: {self.codec_choice.get()}")
        self._update_status("Initializing...")

        # Show and start progress bar
        self.progress_bar.pack(
            fill=tk.X, expand=True, side=tk.LEFT, padx=5
        )  # Pack it now
        self.progress_bar.start()

        # Create temp dir *before* starting thread
        try:
            self.temp_dir = tempfile.TemporaryDirectory(prefix="silero_vad_gui_")
            self._log(f"Temp dir: {self.temp_dir.name}")
        except Exception as e:
            self._log(f"Error creating temporary directory: {e}", level=logging.ERROR)
            messagebox.showerror("Error", f"Failed to create temporary directory: {e}")
            self._reset_gui_state(error=True)  # Reset GUI since we can't proceed
            return

        # Run processing in a separate thread
        self.process_thread = threading.Thread(
            target=self._process_video_thread,
            args=(
                in_path,
                out_path,
                buffer_val / 1000.0,
                self.codec_choice.get(),
                self.temp_dir.name,
            ),  # Pass buffer in seconds
            daemon=True,
        )
        self.process_thread.start()

    def _process_video_thread(
        self, input_path, output_path, buffer_s, codec, temp_dir_path
    ):
        """The core video processing logic (runs in a thread)."""
        try:
            # 1. Extract Audio using ffmpeg
            self._log("Step 1/4: Extracting audio...")
            self._update_status("Extracting audio")
            audio_path = os.path.join(temp_dir_path, "extracted_audio.wav")
            extract_cmd = [
                FFMPEG_COMMAND,
                "-y",  # Overwrite if exists in temp
                "-i",
                input_path,
                "-vn",  # No video
                "-acodec",
                "pcm_s16le",  # Standard WAV format
                "-ar",
                str(SAMPLING_RATE),  # Required sample rate
                "-ac",
                "1",  # Mono audio
                "-progress",
                "-",
                "-nostats",  # Output progress to stderr for potential future parsing
                audio_path,
            ]
            self._log(f"FFmpeg cmd: {' '.join(extract_cmd)}")
            process = subprocess.run(
                extract_cmd,
                check=True,
                capture_output=True,
                text=True,
                startupinfo=get_startup_info(),
                encoding="utf-8",
                errors="replace",
            )
            self._log("Audio extraction successful.")

            # 2. Perform VAD
            self._log("Step 2/4: Analyzing audio for speech...")
            self._update_status("Analyzing speech")
            try:
                # Reading audio can be time-consuming for long files
                self._log(f"Reading audio file: {audio_path}")
                wav = read_audio(audio_path, sampling_rate=SAMPLING_RATE)
                self._log(f"Audio samples read: {len(wav)}")
                # Apply VAD
                # Increased min_silence_duration_ms for less fragmented speech
                # speech_pad_ms adds padding symmetrically around detected speech *before* merging.
                # buffer_s is then applied again during merging for asymmetric extension.
                speech_timestamps = get_speech_timestamps(
                    wav,
                    self.vad_model,
                    sampling_rate=SAMPLING_RATE,
                    threshold=0.45,  # Threshold for speech probability
                    min_speech_duration_ms=250,  # Ignore very short speech bursts
                    min_silence_duration_ms=400,  # Ignore short silences within speech
                    window_size_samples=512,  # Samples per VAD window
                    speech_pad_ms=50,  # Small padding during initial detection
                )
                self.vad_model.reset_states()  # Reset model state after processing
            except Exception as e:
                self._log(f"Error during VAD processing: {e}", level=logging.ERROR)
                logger.exception("VAD Error details:")
                raise  # Re-raise to be caught by outer try-except

            if not speech_timestamps:
                self._log("No speech detected in the audio.", level=logging.WARNING)
                raise ValueError("No speech segments found.")

            total_segments = len(speech_timestamps)
            self._log(f"Found {total_segments} initial speech segments.")
            # Convert samples to seconds
            speech_timestamps_s = [
                {"start": ts["start"] / SAMPLING_RATE, "end": ts["end"] / SAMPLING_RATE}
                for ts in speech_timestamps
            ]

            # 3. Apply Buffer and Merge Timestamps
            self._log(
                f"Step 3/4: Applying {buffer_s*1000:.0f}ms buffer and merging segments..."
            )
            self._update_status("Merging speech segments")
            # Pass the *full* user buffer here for merging/extending segments
            final_timestamps = merge_overlapping_timestamps(
                speech_timestamps_s, buffer_s
            )
            merged_segments = len(final_timestamps)
            if merged_segments < total_segments:
                self._log(
                    f"Merged into {merged_segments} final segments after buffering."
                )
            else:
                self._log(
                    f"Found {merged_segments} final segments (no merge required)."
                )
            # Log first few segments for verification
            for i, ts in enumerate(final_timestamps[: min(5, len(final_timestamps))]):
                self._log(
                    f"  Segment {i+1}: {format_time(ts['start'])} -> {format_time(ts['end'])}"
                )
            if len(final_timestamps) > 5:
                self._log("  ...")

            # 4. Re-encode Video with FFmpeg using select/aselect filters
            self._log("Step 4/4: Filtering video/audio and re-encoding...")
            self._update_status("Encoding final video")
            if (
                not final_timestamps
            ):  # Should not happen if checked earlier, but safeguard
                raise ValueError("No final speech segments to process after merging.")

            # Build filtergraph segments: 'between(t,start1,end1)+between(t,start2,end2)+...'
            # Increased precision for timestamps in filter
            select_filter_parts = [
                f"between(t,{ts['start']:.6f},{ts['end']:.6f})"
                for ts in final_timestamps
            ]
            select_filter = (
                f"select='{'+'.join(select_filter_parts)}',setpts=N/FRAME_RATE/TB"
            )
            aselect_filter = (
                f"aselect='{'+'.join(select_filter_parts)}',asetpts=N/SR/TB"
            )

            # Handle codec options
            codec_opts = ["-c:v", codec]
            # Add quality/preset flags (example for libx264/nvenc)
            if codec == "libx264":
                codec_opts.extend(
                    ["-preset", "medium", "-crf", "23"]
                )  # Standard quality/speed
            elif codec == "libx265":
                codec_opts.extend(["-preset", "medium", "-crf", "28"])  # Standard x265
            elif "nvenc" in codec:
                # Consider 'p5' (slower, better quality) or 'p6' (faster) based on need
                codec_opts.extend(
                    ["-preset", "p5", "-cq", "24", "-rc", "vbr"]
                )  # NVENC quality mode
            elif "qsv" in codec:
                codec_opts.extend(
                    ["-preset", "medium", "-global_quality", "24"]
                )  # Intel QSV (preset affects speed)
            elif "videotoolbox" in codec:
                # Use average bitrate for VT? Quality can be inconsistent. Or higher q:v.
                codec_opts.extend(
                    ["-profile:v", "main", "-q:v", "75"]
                )  # Apple VT quality (adjust as needed, lower=better)

            # FFmpeg command
            ffmpeg_cmd = [
                FFMPEG_COMMAND,
                "-y",  # Overwrite output file
                "-i",
                input_path,
                "-vf",
                select_filter,
                "-af",
                aselect_filter,
                "-map",
                "0:v:0?",  # Map first video stream if exists
                "-map",
                "0:a:0?",  # Map first audio stream if exists
                # To copy all streams (video, audio, subtitles): remove -map lines, add -c copy (but filtering won't apply)
                # To copy specific other streams: -map 0:s:0? -c:s copy etc.
            ]
            ffmpeg_cmd.extend(codec_opts)
            ffmpeg_cmd.extend(
                [
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",  # Common audio codec
                    "-vsync",
                    "vfr",  # Variable Frame Rate sync, needed with setpts
                    # '-copyts', # copyts can sometimes cause issues with filtering, test carefully
                    "-avoid_negative_ts",
                    "make_zero",  # Handle potential timestamp issues
                    "-progress",
                    "-",
                    "-nostats",  # Output progress to stderr
                    output_path,
                ]
            )

            self._log(f"FFmpeg cmd: {' '.join(ffmpeg_cmd)}")
            # Run and capture output
            process = subprocess.run(
                ffmpeg_cmd,
                check=True,
                capture_output=True,
                text=True,
                startupinfo=get_startup_info(),
                encoding="utf-8",
                errors="replace",
            )

            # Log ffmpeg output sparsely to avoid flooding
            if process.stderr:
                stderr_lines = process.stderr.strip().splitlines()
                log_limit = 15  # Log first/last lines of ffmpeg output
                if len(stderr_lines) > log_limit * 2:
                    # Filter progress lines unless they are near start/end
                    logged_lines_count = 0
                    filtered_lines = []
                    for i, line in enumerate(stderr_lines):
                        is_progress = line.startswith(
                            (
                                "frame=",
                                "size=",
                                "time=",
                                "bitrate=",
                                "speed=",
                                "progress=",
                            )
                        )
                        if (
                            i < log_limit
                            or i >= len(stderr_lines) - log_limit
                            or not is_progress
                        ):
                            filtered_lines.append(f"  [FFmpeg] {line}")
                            logged_lines_count += 1
                        elif (
                            i == log_limit and logged_lines_count == log_limit
                        ):  # Indicate omission only once
                            filtered_lines.append(
                                f"  [FFmpeg] ... (omitted {len(stderr_lines) - log_limit*2} progress lines) ..."
                            )
                    for line in filtered_lines:
                        self._log(line)

                else:
                    for line in stderr_lines:
                        self._log(f"  [FFmpeg] {line}")

            self._log("--- Processing Completed Successfully ---")
            self._update_status("Completed")
            # Schedule messagebox success confirmation in main thread
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Success",
                    f"Video processed successfully!\nOutput saved to:\n{output_path}",
                ),
            )

        except subprocess.CalledProcessError as e:
            self._log(
                f"--- FFmpeg Command Failed (Exit Code {e.returncode}) ---",
                level=logging.ERROR,
            )
            stderr_tail = (
                e.stderr[-1500:] if e.stderr else "(No stderr)"
            )  # Limit stderr length
            self._log(
                f"FFmpeg stderr (last 1500 chars):\n{stderr_tail}", level=logging.ERROR
            )
            self._update_status("Error (FFmpeg)")
            # Schedule messagebox in main thread
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "FFmpeg Error",
                    f"FFmpeg failed. Check logs for details.\n\nError message tail:\n{stderr_tail}",
                ),
            )
        except ValueError as e:  # Catch specific errors like "No speech"
            self._log(f"--- Processing Error: {e} ---", level=logging.ERROR)
            self._update_status(f"Error ({e})")
            self.root.after(0, lambda: messagebox.showerror("Processing Error", str(e)))
        except Exception as e:
            self._log(f"--- An Unexpected Error Occurred: {e} ---", level=logging.ERROR)
            logger.exception("Detailed error trace:")  # Log full traceback
            self._update_status("Error (Unknown)")
            # Schedule messagebox in main thread
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Error", f"An unexpected error occurred: {e}"
                ),
            )
        finally:
            # --- Cleanup and GUI Reset ---
            if self.temp_dir:
                try:
                    self.temp_dir.cleanup()
                    self._log(f"Temporary directory cleaned up.")
                except Exception as e:
                    # Log warning but don't stop execution for cleanup failure
                    self._log(
                        f"Warning: Could not clean up temp directory {self.temp_dir.name}: {e}",
                        level=logging.WARNING,
                    )
            self.temp_dir = None  # Reset temp dir holder
            self.is_processing = False
            # Schedule GUI update from the main thread, pass error status
            was_error = self.status_label.cget("text").startswith("Status: Error")
            self.root.after(0, self._reset_gui_state, was_error)

    def _reset_gui_state(self, error=False):
        """Reset button state and progress bar after processing finishes."""
        self.start_button.config(state=tk.NORMAL, text="Start Processing")
        # Re-enable browse buttons
        for widget in self.input_entry.master.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.config(state=tk.NORMAL)
        for widget in self.output_entry.master.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.config(state=tk.NORMAL)

        self.progress_bar.stop()
        self.progress_bar.pack_forget()  # Hide the progress bar again
        # Keep "Completed" or "Error" status, or reset to Idle if desired
        if not error:
            # Optionally reset status after a delay if successful
            self.root.after(
                5000,
                lambda: (
                    self._update_status("Idle")
                    if self.status_label.cget("text") == "Status: Completed"
                    else None
                ),
            )
        else:
            # Keep error status visible
            pass


# --- Main Execution ---
if __name__ == "__main__":
    # Enable high DPI awareness on Windows if possible
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass  # Ignore if not on Windows or ctypes fails

    root = tk.Tk()
    # Set a theme for better visuals if available
    style = ttk.Style(root)
    available_themes = style.theme_names()
    # Prefer modern themes based on OS
    preferred_themes = []
    if platform.system() == "Windows":
        preferred_themes = ["vista", "win11", "xpnative"]  # Added win11
    elif platform.system() == "Darwin":
        preferred_themes = ["aqua"]
    else:  # Linux/other
        preferred_themes = ["clam", "alt", "default"]

    chosen_theme = None
    for theme in preferred_themes:
        if theme in available_themes:
            try:
                style.theme_use(theme)
                chosen_theme = theme
                break
            except tk.TclError:
                continue  # Try next preferred theme

    if not chosen_theme:
        print(
            f"Preferred ttk themes not available ({preferred_themes}), using default."
        )

    app = VadApp(root)
    root.mainloop()
