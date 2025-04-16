import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import json
import os
import threading
import gc
import torch
from typing import List, Dict, Any

# --- WhisperX Import ---
try:
    import whisperx
except ImportError:
    print("Error: whisperx not found.")  # Explain: Lib not found
    print(
        "Please install it: pip install git+https://github.com/m-bain/whisperX.git"
    )  # Explain: Install cmd
    exit()

# --- Configuration ---
CONFIG_FILE = "whisperx_gui_config.json"
DEFAULT_CONFIG = {
    "audio_file_path": "",
    "speaker_names_str": "Speaker 1,Speaker 2",
    "chars_per_line": "42",
    "max_lines_per_entry": "2",  # Max lines config
    "model_size": "base",
    "hf_token": "",  # Default no token
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "compute_type": ("float16" if torch.cuda.is_available() else "int8"),
}


def load_config() -> Dict[str, Any]:
    """Loads config from file or returns defaults."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Ensure all default keys exist
            loaded_config = DEFAULT_CONFIG.copy()
            loaded_config.update(config)  # Overwrite defaults with loaded

            # Auto-adjust device/compute type if necessary
            if loaded_config.get("device") == "cuda" and not torch.cuda.is_available():
                loaded_config["device"] = "cpu"
                loaded_config["compute_type"] = "int8"
            elif loaded_config.get("device") == "cpu":
                loaded_config["compute_type"] = "int8"

            return loaded_config
        except json.JSONDecodeError:
            print(f"Warn: Config corrupted. Using defaults.")  # Explain: Config error
            return DEFAULT_CONFIG.copy()
        except Exception as e:
            print(
                f"Warn: Error loading config: {e}. Using defaults."
            )  # Explain: Config error
            return DEFAULT_CONFIG.copy()
    else:
        return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any]):
    """Saves config to file."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
    except IOError as e:
        print(f"Error saving config: {e}")  # Explain: Save fail


def format_timestamp(seconds: float) -> str:
    """Converts seconds to SRT timestamp format HH:MM:SS,ms."""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def split_text_into_blocks(text: str, max_chars: int, max_lines: int) -> List[str]:
    """
    Splits text into lines respecting max_chars,
    then groups lines into blocks respecting max_lines.
    """
    words = text.split()
    lines = []
    current_line = ""

    # Split into lines based on max_chars
    for word in words:
        if not current_line:
            current_line = word
        elif len(current_line) + len(word) + 1 <= max_chars:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:  # Append the last line
        lines.append(current_line)

    # Group lines into blocks based on max_lines
    blocks = []
    for i in range(0, len(lines), max_lines):
        block = "\n".join(lines[i : i + max_lines])
        blocks.append(block)

    # Handle empty input case resulting in empty blocks
    if not blocks and text.strip():
        # If original text wasn't empty, but splitting made it so (e.g., only spaces)
        # Or if splitting resulted in no lines somehow. Add empty block? Or handle upstream.
        pass  # Let it return empty list if no content lines

    return blocks


def generate_srt_content(
    result: Dict[str, Any],
    speaker_names_map: Dict[str, str],
    chars_per_line: int,
    max_lines_per_entry: int,
) -> str:
    """Generates SRT content from whisperx result with speaker info."""
    srt_content = []
    entry_count = 1

    # Check if segments exist in the result dictionary
    if "segments" not in result or not isinstance(result["segments"], list):
        print(
            "Error: 'segments' key missing or not a list in result."
        )  # Explain: Bad format
        return "Error: Invalid result format for SRT generation."

    for i, segment in enumerate(result["segments"]):
        start_time = segment.get("start")
        end_time = segment.get("end")
        text = segment.get("text", "").strip()
        # Get speaker assigned by assign_word_speakers
        speaker = segment.get("speaker", "UNKNOWN")  # Explain: Use assigned

        if start_time is None or end_time is None or not text:
            print(
                f"Skipping segment {i+1}: missing data (time or text)."
            )  # Explain: Skip segment
            continue

        # Map speaker ID (e.g., SPEAKER_00) to the actual name
        speaker_name = speaker_names_map.get(speaker, speaker)  # Use ID if no map

        # Prepend speaker name to the text
        formatted_text_with_speaker = f"{text}"  # Add speaker prefix

        # Split the prefixed text into display blocks
        text_blocks = split_text_into_blocks(
            formatted_text_with_speaker, chars_per_line, max_lines_per_entry
        )

        # Create SRT entries for each block from the segment
        for block in text_blocks:
            if not block.strip():  # Skip empty blocks
                continue

            start_ts = format_timestamp(start_time)
            end_ts = format_timestamp(end_time)

            # Add SRT entry: Index, Timestamp, Text Block
            srt_entry = f"{entry_count}\n"
            srt_entry += f"{start_ts} --> {end_ts}\n"
            srt_entry += f"{block}\n"  # Add the text block
            srt_content.append(srt_entry)
            entry_count += 1  # Increment for each *block*

    return "\n".join(srt_content)


# --- GUI Class ---
class WhisperXGUI:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("WhisperX SRT Generator")
        master.geometry("750x650")  # Give it a bit more space

        self.config = load_config()
        self.processing_thread = None

        # --- File Selection ---
        self.file_frame = tk.Frame(master)
        self.file_frame.pack(pady=5, padx=10, fill=tk.X)

        tk.Label(self.file_frame, text="Audio File:").pack(side=tk.LEFT, padx=(0, 5))
        self.file_path_var = tk.StringVar(value=self.config.get("audio_file_path", ""))
        self.file_entry = tk.Entry(
            self.file_frame, textvariable=self.file_path_var, width=60  # Wider entry
        )
        self.file_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.browse_button = tk.Button(
            self.file_frame, text="Browse...", command=self.select_file
        )
        self.browse_button.pack(side=tk.LEFT)

        # --- Settings ---
        self.settings_frame = tk.Frame(master)
        self.settings_frame.pack(pady=10, padx=10, fill=tk.X)

        # Configure grid weights for responsiveness
        self.settings_frame.columnconfigure(1, weight=1)
        self.settings_frame.columnconfigure(3, weight=1)

        row_idx = 0
        # Speaker Names
        tk.Label(self.settings_frame, text="Speaker Names (comma-sep):").grid(
            row=row_idx, column=0, sticky=tk.W, padx=5, pady=3
        )
        self.speaker_names_var = tk.StringVar(
            value=self.config.get("speaker_names_str", "")
        )
        self.speaker_entry = tk.Entry(
            self.settings_frame,
            textvariable=self.speaker_names_var,  # Width auto managed by grid
        )
        self.speaker_entry.grid(
            row=row_idx, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=3
        )
        row_idx += 1

        # Formatting Controls (Chars/Line, Lines/Entry)
        tk.Label(self.settings_frame, text="Max Chars/Line:").grid(
            row=row_idx, column=0, sticky=tk.W, padx=5, pady=3
        )
        self.chars_var = tk.StringVar(value=self.config.get("chars_per_line", "42"))
        self.chars_entry = tk.Entry(
            self.settings_frame, textvariable=self.chars_var, width=10
        )
        self.chars_entry.grid(row=row_idx, column=1, sticky=tk.W, padx=5, pady=3)

        tk.Label(self.settings_frame, text="Max Lines/Entry:").grid(
            row=row_idx, column=2, sticky=tk.W, padx=5, pady=3
        )
        self.max_lines_var = tk.StringVar(
            value=self.config.get("max_lines_per_entry", "2")
        )
        self.max_lines_entry = tk.Entry(
            self.settings_frame, textvariable=self.max_lines_var, width=10
        )
        self.max_lines_entry.grid(row=row_idx, column=3, sticky=tk.W, padx=5, pady=3)
        row_idx += 1

        # Model Size
        tk.Label(self.settings_frame, text="Model Size:").grid(
            row=row_idx, column=0, sticky=tk.W, padx=5, pady=3
        )
        self.model_var = tk.StringVar(value=self.config.get("model_size", "base"))
        self.model_entry = tk.Entry(
            self.settings_frame, textvariable=self.model_var, width=15
        )
        # Consider adding options like: tiny, base, small, medium, large
        self.model_entry.grid(row=row_idx, column=1, sticky=tk.W, padx=5, pady=3)
        row_idx += 1

        # Device and Compute Type (Readonly)
        tk.Label(self.settings_frame, text="Device:").grid(
            row=row_idx, column=0, sticky=tk.W, padx=5, pady=3
        )
        self.device_var = tk.StringVar(value=self.config.get("device", "cpu"))
        self.device_entry = tk.Entry(
            self.settings_frame,
            textvariable=self.device_var,
            width=10,
            state="readonly",
        )
        self.device_entry.grid(row=row_idx, column=1, sticky=tk.W, padx=5, pady=3)

        tk.Label(self.settings_frame, text="Compute Type:").grid(
            row=row_idx, column=2, sticky=tk.W, padx=5, pady=3
        )
        self.compute_type_var = tk.StringVar(
            value=self.config.get("compute_type", "int8")
        )
        self.compute_type_entry = tk.Entry(
            self.settings_frame,
            textvariable=self.compute_type_var,
            width=10,
            state="readonly",
        )
        self.compute_type_entry.grid(row=row_idx, column=3, sticky=tk.W, padx=5, pady=3)
        row_idx += 1

        # HF Token
        tk.Label(self.settings_frame, text="HF Token (Optional):").grid(
            row=row_idx, column=0, sticky=tk.W, padx=5, pady=3
        )
        self.hf_token_var = tk.StringVar(value=self.config.get("hf_token", ""))
        self.hf_token_entry = tk.Entry(
            self.settings_frame, textvariable=self.hf_token_var, show="*"
        )
        self.hf_token_entry.grid(
            row=row_idx, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=3
        )
        # row_idx += 1 # Not needed if last row

        # --- Actions ---
        self.action_frame = tk.Frame(master)
        self.action_frame.pack(pady=15, padx=10, fill=tk.X)

        self.generate_button = tk.Button(
            self.action_frame,
            text="Generate SRT",
            command=self.start_processing,
            width=20,
            height=2,  # Larger button
        )
        self.generate_button.pack(pady=5)  # Center button

        # --- Status Area ---
        tk.Label(master, text="Status Log:").pack(pady=(5, 0), padx=10, anchor=tk.W)
        self.status_text = scrolledtext.ScrolledText(
            master,
            height=12,
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.SUNKEN,
            borderwidth=1,
        )
        self.status_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        # --- Initial Status Update ---
        self.update_status(
            f"Config loaded. Device: {self.device_var.get()}, Compute: {self.compute_type_var.get()}"
        )
        if self.device_var.get() == "cuda":
            self.update_status("CUDA detected and selected.")
        else:
            self.update_status("CUDA not detected or available. Using CPU.")

    def update_status(self, message: str):
        """Appends message to the status text area in a thread-safe way."""

        def _update():
            try:
                self.status_text.config(state=tk.NORMAL)
                self.status_text.insert(tk.END, message + "\n")
                self.status_text.see(tk.END)  # Auto-scroll
                self.status_text.config(state=tk.DISABLED)
            except tk.TclError:
                # Handle cases where the widget might be destroyed during shutdown
                pass

        # Ensure GUI updates happen on the main thread
        if self.master.winfo_exists():
            self.master.after(0, _update)

    def select_file(self):
        """Opens file dialog to select audio file."""
        filepath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=(
                ("Audio Files", "*.mp3 *.wav *.m4a *.ogg *.flac"),
                ("All Files", "*.*"),
            ),
        )
        if filepath:
            self.file_path_var.set(filepath)
            self.update_status(
                f"Selected audio file: {os.path.basename(filepath)}"
            )  # Show only filename

    def get_speaker_map(self) -> Dict[str, str]:
        """Parses speaker names string into a map (SPEAKER_00 -> Name)."""
        names_str = self.speaker_names_var.get().strip()
        names = [name.strip() for name in names_str.split(",") if name.strip()]
        # Map SPEAKER_00 -> name1, SPEAKER_01 -> name2 etc.
        speaker_map = {f"SPEAKER_{i:02d}": name for i, name in enumerate(names)}
        if not speaker_map:
            self.update_status(
                "Warning: No speaker names provided. Using default IDs."
            )  # Explain: No names set
        return speaker_map

    def start_processing(self):
        """Validates inputs and starts the WhisperX processing thread."""
        # --- Input Validation ---
        audio_file = self.file_path_var.get()
        if not audio_file or not os.path.exists(audio_file):
            messagebox.showerror("Error", "Valid audio file path required.")
            return

        model_size = self.model_var.get().strip()
        if not model_size:
            messagebox.showerror("Error", "Model size cannot be empty.")
            return

        try:
            chars_per_line = int(self.chars_var.get().strip())
            if chars_per_line <= 0:
                raise ValueError("Must be positive")
        except ValueError:
            messagebox.showerror("Error", "Max Chars/Line must be a positive integer.")
            return

        try:
            max_lines_per_entry = int(self.max_lines_var.get().strip())
            if max_lines_per_entry <= 0:
                raise ValueError("Must be positive")
        except ValueError:
            messagebox.showerror("Error", "Max Lines/Entry must be a positive integer.")
            return

        speaker_names_str = self.speaker_names_var.get().strip()  # Can be empty
        hf_token = self.hf_token_var.get().strip() or None  # Use None if empty

        # --- Update Config ---
        self.config["audio_file_path"] = audio_file
        self.config["speaker_names_str"] = speaker_names_str
        self.config["chars_per_line"] = str(chars_per_line)
        self.config["max_lines_per_entry"] = str(max_lines_per_entry)
        self.config["model_size"] = model_size
        self.config["hf_token"] = (
            hf_token if hf_token else ""
        )  # Store empty string in config
        # Device/Compute Type are read-only, derived at load/runtime

        save_config(self.config)

        # --- Start Processing Thread ---
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning(
                "Busy", "Processing is already in progress."
            )  # Explain: Already running
            return

        self.generate_button.config(state=tk.DISABLED, text="Processing...")
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete("1.0", tk.END)  # Clear previous log
        self.status_text.config(state=tk.DISABLED)
        self.update_status(f"Starting processing for: {os.path.basename(audio_file)}")

        self.processing_thread = threading.Thread(
            target=self.run_whisperx_pipeline,
            args=(
                audio_file,
                model_size,
                chars_per_line,
                max_lines_per_entry,
                hf_token,
            ),
            daemon=True,  # Allow app exit even if thread runs
        )
        self.processing_thread.start()

    def run_whisperx_pipeline(
        self,
        audio_file: str,
        model_size: str,
        chars_per_line: int,
        max_lines_per_entry: int,
        hf_token: str | None,
    ):
        """Runs the core WhisperX logic in a separate thread."""
        model = None
        model_a = None
        diarize_model = None
        audio = None
        result = None  # Initialize result
        try:
            device = self.device_var.get()
            compute_type = self.compute_type_var.get()
            # Larger batch-size helps VRAM-rich GPUs, smaller helps low VRAM
            batch_size = 16 if device == "cuda" else 4  # Smaller batch for CPU

            # --- Load Audio ---
            self.update_status("Loading audio...")
            audio = whisperx.load_audio(audio_file)
            self.update_status("Audio loaded successfully.")

            # --- Load Transcription Model ---
            self.update_status(
                f"Loading model '{model_size}' on {device} [{compute_type}]..."
            )
            model = whisperx.load_model(model_size, device, compute_type=compute_type)
            self.update_status("Transcription model loaded.")

            # --- Transcribe ---
            self.update_status("Starting transcription...")
            result = model.transcribe(
                audio, batch_size=batch_size
            )  # Initial transcription
            language_code = result.get("language", "unknown")
            self.update_status(f"Transcription complete (Language: {language_code}).")

            # --- Unload Transcription Model (Save VRAM/RAM) ---
            del model
            model = None  # Clear reference before GC
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            self.update_status("Transcription model unloaded.")

            # --- Check Transcription Result ---
            if not result or "segments" not in result or not result["segments"]:
                raise ValueError("Transcription failed or produced no segments.")

            # --- Alignment (Optional but improves timing) ---
            # Skip alignment if language not supported or model load fails
            try:
                self.update_status(
                    f"Loading alignment model for language: {language_code}..."
                )
                model_a, metadata = whisperx.load_align_model(
                    language_code=language_code, device=device
                )
                self.update_status("Alignment model loaded. Aligning segments...")
                # Align modifies 'result' in place or returns the modified structure
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    device,
                    return_char_alignments=False,
                )
                self.update_status("Alignment complete.")

                # --- Unload Alignment Model ---
                del model_a
                model_a = None
                del metadata
                metadata = None
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                self.update_status("Alignment model unloaded.")

            except Exception as align_error:
                self.update_status(
                    f"Warning: Alignment failed ({align_error}). Proceeding without alignment."
                )  # Explain: Align fail warn

            # Ensure 'result' is a dict with 'segments' after potential alignment failure
            if not isinstance(result, dict) or "segments" not in result:
                # If alignment failed and didn't return the expected dict, reconstruct it.
                # This assumes 'result' might have become just the list of segments.
                if isinstance(result, list):
                    result = {"segments": result, "language": language_code}  # Re-wrap
                else:
                    raise ValueError(
                        "Result format is unexpected after alignment step."
                    )

            # --- Diarization ---
            self.update_status("Loading diarization model...")
            # Use token for gated models like pyannote/speaker-diarization
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token, device=device
            )
            self.update_status("Performing speaker diarization...")
            # You might need to adjust min_speakers/max_speakers if known
            diarize_segments = diarize_model(audio)  # Pass audio tensor
            self.update_status("Diarization complete.")

            # --- Unload Diarization Model ---
            del diarize_model
            diarize_model = None
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            self.update_status("Diarization model unloaded.")

            # --- Assign Speakers ---
            self.update_status("Assigning speakers to segments...")
            # assign_word_speakers modifies segments within the 'result' dictionary
            result = whisperx.assign_word_speakers(diarize_segments, result)
            self.update_status("Speaker assignment complete.")

            # --- Generate SRT ---
            self.update_status("Generating SRT file...")
            speaker_map = self.get_speaker_map()
            srt_output = generate_srt_content(
                result, speaker_map, chars_per_line, max_lines_per_entry
            )

            # --- Save SRT ---
            if "Error:" in srt_output:  # Check if SRT generation itself failed
                raise ValueError(f"SRT generation failed: {srt_output}")

            srt_filename = os.path.splitext(audio_file)[0] + ".srt"
            with open(srt_filename, "w", encoding="utf-8") as f:
                f.write(srt_output)
            self.update_status(f"SRT file saved successfully: {srt_filename}")
            self.master.after(
                0,
                messagebox.showinfo,
                "Success",
                f"SRT file generated:\n{srt_filename}",
            )  # Use main thread dialog

        except Exception as e:
            error_message = f"An error occurred during processing: {e}"
            print(f"Error: {error_message}\nTraceback:")  # Log details
            import traceback

            traceback.print_exc()  # Print traceback to console
            self.update_status(f"ERROR: {e}")  # Show simplified error in GUI
            self.master.after(
                0, messagebox.showerror, "Processing Error", error_message
            )  # Use main thread dialog
        finally:
            # --- Final Cleanup ---
            # Ensure button is re-enabled regardless of success/failure
            self.master.after(0, self.reset_button)
            # Release resources explicitly
            del model, model_a, diarize_model, audio, result, diarize_segments
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.update_status("Processing finished. Resources released.")

    def reset_button(self):
        """Resets the generate button state (called via master.after)."""
        try:
            if self.master.winfo_exists():
                self.generate_button.config(state=tk.NORMAL, text="Generate SRT")
        except tk.TclError:
            pass  # Widget might be destroyed


# --- Main Execution ---
if __name__ == "__main__":
    # Optional: Try importing and using a theme like sv_ttk
    try:
        import sv_ttk

        USE_THEME = True
    except ImportError:
        sv_ttk = None
        USE_THEME = False
        print("sv_ttk not found, using default theme.")  # Explain: Theme missing

    root = tk.Tk()
    gui = WhisperXGUI(root)

    if USE_THEME and sv_ttk:
        sv_ttk.set_theme("dark")  # Or "light"

    # Handle window close event gracefully
    def on_closing():
        if gui.processing_thread and gui.processing_thread.is_alive():
            if messagebox.askokcancel("Quit", "Processing is ongoing. Quit anyway?"):
                # Note: Force quitting might leave resources unclean.
                # A more robust solution would involve signaling the thread to stop.
                root.destroy()
            else:
                return  # Don't close if user cancels
        else:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
