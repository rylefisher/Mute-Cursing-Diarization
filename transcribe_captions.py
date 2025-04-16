import tkinter as tk
import tkinter.filedialog
import whisperx
import datetime
import os
import threading
import torch # Added for device check
import tkinter.messagebox as messagebox  # Explicit import if not already done
import sv_ttk

class WhisperXApp:
    """GUI App for WhisperX SRT generation."""

    def __init__(
        self,
        root_window,
        whisper_model_name: str = "large-v3-turbo",
        language_code: str = "en",
        device="cuda",
        compute_type: str = "float16",
        batch_size: int = 8,
        hf_token: str | None = None,  # HF token for diarization
        diarize: bool = True,  # Flag to enable diarization
        asr_options: dict | None = None,
        vad_options: dict | None = None,
        transcribe_config: dict | None = None,
        align_config: dict | None = None,
        diarize_options: dict | None = None,  # min/max speakers
    ):

        self.whisper_model_name = whisper_model_name
        self.language_code = language_code
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Use float16 on CUDA, int8 on CPU unless specified
        self.compute_type = compute_type if self.device == "cuda" else "int8"
        self.batch_size = batch_size
        self.hf_token = hf_token  # Store HF token
        self.diarize = diarize  # Store diarization flag

        """Initialize the application."""
        self.root = root_window
        self.root.title("WhisperX SRT Generator")
        self.root.geometry("500x300") # Set window size

        # --- Configuration ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.batch_size = 16
        self.model = None # Load later
        self.align_model = None
        self.align_metadata = None
        self.loaded_lang = None

        # --- GUI Elements ---
        # File Selection
        self.file_label = tk.Label(self.root, text="Audio File:")
        self.file_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.selected_file_label = tk.Label(self.root, text="No file selected", wraplength=350)
        self.selected_file_label.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.browse_button = tk.Button(self.root, text="Browse", command=self._select_file)
        self.browse_button.grid(row=0, column=2, padx=10, pady=10)

        # Max Chars per Line
        self.max_chars_label = tk.Label(self.root, text="Max Chars/Line:")
        self.max_chars_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.max_chars_entry = tk.Entry(self.root)
        self.max_chars_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        self.max_chars_entry.insert(0, "42") # Default value

        # Max Lines per Timestamp
        self.max_lines_label = tk.Label(self.root, text="Max Lines/Entry:")
        self.max_lines_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.max_lines_entry = tk.Entry(self.root)
        self.max_lines_entry.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
        self.max_lines_entry.insert(0, "2") # Default value

        # Process Button
        self.process_button = tk.Button(self.root, text="Generate SRT", command=self._start_processing)
        self.process_button.grid(row=3, column=0, columnspan=3, padx=10, pady=20)

        # Status Label
        self.status_label = tk.Label(self.root, text="Status: Idle")
        self.status_label.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky="w")

        # --- Variables ---
        self.audio_file_path = None

        self._default_asr_options = {
            "log_prob_threshold": -3.0,  # Accept low confidence words
            "no_speech_threshold": 0.1,  # Catch quieter speech
            "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Robustness
            "condition_on_previous_text": True,  # Context for long audio
            "compression_ratio_threshold": None,  # Disable filter
            "word_timestamps": True,  # Needed for alignment
            "without_timestamps": False,
            "beam_size": 5,  # Balance accuracy/speed
            "best_of": 5,  # Balance accuracy/speed
            "patience": None,  # Faster decoding
            "length_penalty": 1.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "prompt_reset_on_temperature": 0.5,
            "initial_prompt": None,
            "prefix": None,
            "suppress_blank": True,
            "suppress_tokens": [],  # Ensure no tokens suppressed
            "max_initial_timestamp": 1.0,
            "prepend_punctuations": "",
            "append_punctuations": "",
            "suppress_numerals": False,
            "max_new_tokens": None,
            "clip_timestamps": "0",
            "hallucination_silence_threshold": None,
            "hotwords": None,  # Could add curses if needed
        }
        self._default_vad_options = {
            "vad_threshold": 0.15, # VAD sensitivity
            "min_speech_duration_ms": 50, # Catch short speech
            "max_speech_duration_s": float("inf"),
            "min_silence_duration_ms": 300, # Silence duration sensitivity
            "window_size_samples": 1024,
            "speech_pad_ms": 500, # Padding around speech
        }
        self._default_transcribe_config = {
            "chunk_size": 25,
            "print_progress": True,
        }
        self._default_align_config = {
            "return_char_alignments": False,  # Not needed for words
            "print_progress": True,  # Less console noise
        }
        self._default_diarize_options = {
            "min_speakers": None,  # Auto-detect
            "max_speakers": None,  # Auto-detect or set upper limit
        }

        self.asr_options = {**self._default_asr_options, **(asr_options or {})}
        self.vad_options = {**self._default_vad_options, **(vad_options or {})}
        self.transcribe_config = {
            **self._default_transcribe_config,
            **(transcribe_config or {}),
        }
        self.align_config = {**self._default_align_config, **(align_config or {})}
        self.diarize_options = {
            **self._default_diarize_options,
            **(diarize_options or {}),
        }

        # Placeholder for models
        self.vad_model = None
        self.whisper_model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None  # Diarization model placeholder

        # Configure grid expansion
        self.root.columnconfigure(1, weight=1)

    def _select_file(self):
        """Open file dialog to select audio."""
        filepath = tk.filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=(("Audio Files", "*.wav *.mp3 *.m4a *.ogg *.flac"), ("All Files", "*.*"))
        )
        if filepath:
            self.audio_file_path = filepath
            filename = os.path.basename(filepath)
            self.selected_file_label.config(text=filename)
            self.status_label.config(text="Status: File selected")
        else:
            self.audio_file_path = None
            self.selected_file_label.config(text="No file selected")
            self.status_label.config(text="Status: Idle")

    def _format_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format HH:MM:SS,ms."""
        delta = datetime.timedelta(seconds=seconds)
        hours, remainder = divmod(delta.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

    def _split_line(self, text: str, max_chars: int) -> list[str]:
        """Split text into lines <= max_chars."""
        lines = []
        current_line = ""
        words = text.split()

        if not words:
            return []

        for word in words:
            # If line empty, add word
            if not current_line:
                if len(word) <= max_chars:
                    current_line = word
                else:
                    # Handle very long word
                    lines.append(word[:max_chars])
                    remaining = word[max_chars:]
                    while len(remaining) > max_chars:
                        lines.append(remaining[:max_chars])
                        remaining = remaining[max_chars:]
                    if remaining:
                        current_line = remaining # Start new line
                    else:
                        current_line = "" # Word fit exactly
            # Check adding next word
            elif len(current_line) + 1 + len(word) <= max_chars:
                current_line += " " + word
            # Start new line
            else:
                lines.append(current_line)
                if len(word) <= max_chars:
                    current_line = word
                else:
                    # Handle very long word again
                    lines.append(word[:max_chars])
                    remaining = word[max_chars:]
                    while len(remaining) > max_chars:
                        lines.append(remaining[:max_chars])
                        remaining = remaining[max_chars:]
                    if remaining:
                        current_line = remaining # Start new line
                    else:
                        current_line = "" # Word fit exactly

        if current_line: # Add last line
            lines.append(current_line)

        return lines

    def _generate_srt(self, result: dict, output_path: str, max_chars: int, max_lines: int):
        """Generate SRT content from aligned results."""
        """Generate SRT content from word timings."""
        srt_content = []
        entry_index = 1

        # check data
        if not result or "word_segments" not in result or not result["word_segments"]:
            # self.status_label.config(text="Status: Error - No word segments") # Use self if in class
            messagebox.showerror(
                "Error", "Transcription result is empty or lacks word segments."
            )
            return

        words = result["word_segments"]
        num_words = len(words)
        word_idx = 0

        while word_idx < num_words:
            # start entry
            entry_lines = []
            current_entry_start_time = -1.0
            current_entry_end_time = -1.0

            # build lines
            for _ in range(max_lines):
                if word_idx >= num_words:
                    break  # no words left

                current_line = ""
                line_char_count = 0
                line_start_idx = word_idx  # track words in line
                current_line_end_time = -1.0

                # build line words
                while word_idx < num_words:
                    word_data = words[word_idx]

                    # validate word
                    if (
                        "word" not in word_data
                        or "start" not in word_data
                        or "end" not in word_data
                    ):
                        print(f"Skipping invalid word data: {word_data}")
                        word_idx += 1
                        continue  # skip malformed

                    word_text = word_data["word"].strip()
                    if not word_text:
                        word_idx += 1
                        continue  # skip empty

                    # set times
                    if current_entry_start_time < 0:  # first word?
                        current_entry_start_time = word_data["start"]

                    # check length
                    word_len = len(word_text)
                    space_len = 1 if current_line else 0
                    potential_len = line_char_count + space_len + word_len

                    # fits?
                    if potential_len <= max_chars:
                        current_line += (" " if current_line else "") + word_text
                        line_char_count = len(current_line)  # use len()
                        current_line_end_time = word_data["end"]
                        word_idx += 1
                    else:
                        # line full
                        if not current_line:  # word too long?
                            current_line = word_text  # add anyway
                            line_char_count = len(current_line)
                            current_line_end_time = word_data["end"]
                            word_idx += 1
                            # break inner (word loop) after adding this single word
                        break  # word doesn't fit

                # line built
                if current_line:
                    entry_lines.append(current_line)
                    current_entry_end_time = current_line_end_time  # update entry end
                else:
                    # No words added (e.g., only invalid words remained)
                    break  # stop adding lines

            # entry built
            if entry_lines:
                # format times
                start_time_fmt = self._format_time(current_entry_start_time)
                end_time_fmt = self._format_time(current_entry_end_time)

                # create entry
                srt_entry = f"{entry_index}\n"
                srt_entry += f"{start_time_fmt} --> {end_time_fmt}\n"
                srt_entry += "\n".join(entry_lines) + "\n\n"
                srt_content.append(srt_entry)
                entry_index += 1
            elif current_entry_start_time < 0 and word_idx < num_words:
                # Handle cases where loop finished but no valid words were found to start entry
                # This might happen if remaining words are all invalid/empty
                print(
                    f"Warning: Could not form entry starting near word index {word_idx}"
                )
                # Advance index manually if stuck on invalid data, though inner loops should handle this
                # word_idx += 1 # Be cautious with manual increments here

        # check content
        if not srt_content:
            # self.status_label.config(text="Status: Warn - No content generated") # Use self if in class
            messagebox.showwarning(
                "Warning", "No SRT content was generated. Check word segments."
            )
            return

        # write file
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.writelines(srt_content)
            # self.status_label.config(text=f"Status: Saved {os.path.basename(output_path)}") # Use self if in class
            messagebox.showinfo(
                "Success", f"SRT file generated successfully:\n{output_path}"
            )
        except IOError as e:
            # self.status_label.config(text="Status: Error writing file") # Use self if in class
            messagebox.showerror("File Error", f"Could not write SRT file: {e}")
        except Exception as e:
            # self.status_label.config(text="Status: Unknown write error") # Use self if in class
            messagebox.showerror(
                "Error", f"An unexpected error occurred during SRT writing: {e}"
            )
    def _process_audio(self):
        """Load models, transcribe, align, and generate SRT."""
        if not self.audio_file_path:
            tk.messagebox.showwarning("Warning", "Please select an audio file first.")
            return

        try:
            max_chars = int(self.max_chars_entry.get())
            max_lines = int(self.max_lines_entry.get())
            if max_chars <= 0 or max_lines <= 0:
                raise ValueError("Values must be positive.")
        except ValueError:
            tk.messagebox.showwarning("Warning", "Max Chars and Max Lines must be positive integers.")
            return

        self.process_button.config(state=tk.DISABLED) # Disable btn
        self.status_label.config(text="Status: Initializing...")
        self.vad_model = whisperx.vad.load_vad_model(self.device)
        self.root.update_idletasks() # Force UI update
        effective_asr_options = {
            k: v for k, v in self.asr_options.items() if v is not None
        }
        try:

            # 1. Load base model (if not loaded)
            if self.model is None:
                self.status_label.config(text="Status: Loading base model...")
                self.root.update_idletasks()
                self.model = whisperx.load_model(
                    self.whisper_model_name,
                    self.device,
                    compute_type=self.compute_type,
                    language=self.language_code or None,  # Pass None if lang not set
                    asr_options=effective_asr_options,
                    vad_options=self.vad_options,
                    vad_model=self.vad_model,
                    task="transcribe",
                )

            # 2. Load audio
            self.status_label.config(text="Status: Loading audio...")
            self.root.update_idletasks()
            try:
                audio = whisperx.load_audio(self.audio_file_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load audio: {e}")

            # 3. Transcribe
            self.status_label.config(text="Status: Transcribing...")
            self.root.update_idletasks()
            transcribe_args = {
                "batch_size": self.batch_size,
                "language": self.language_code,  # Pass language code
                **self.transcribe_config,
            }
            transcribe_args = {k: v for k, v in transcribe_args.items() if v is not None}

            try:
                result = self.model.transcribe(audio, **transcribe_args)
            except Exception as e:
                raise RuntimeError(f"Transcription failed: {e}")

            detected_language = result["language"]
            self.status_label.config(text=f"Status: Detected language: {detected_language}")
            self.root.update_idletasks()

            # 4. Load alignment model (if not loaded or lang changed)
            if self.align_model is None or self.loaded_lang != detected_language:
                self.status_label.config(text="Status: Loading alignment model...")
                self.root.update_idletasks()
                try:
                    self.align_model, self.align_metadata = whisperx.load_align_model(
                        language_code=self.language_code, device=self.device
                    )
                    self.loaded_lang = detected_language
                except Exception as e:
                    raise RuntimeError(f"Failed to load alignment model: {e}")

            # 5. Align
            self.status_label.config(text="Status: Aligning transcription...")
            self.root.update_idletasks()
            try:
                # Ensure segments are valid before passing
                valid_segments = [seg for seg in result.get("segments", []) if isinstance(seg, dict) and 'text' in seg]
                if not valid_segments:
                    raise ValueError("No valid text segments found for alignment.")

                aligned_result = whisperx.align(valid_segments, self.align_model, self.align_metadata, audio, self.device, return_char_alignments=False)
            except Exception as e:
                raise RuntimeError(f"Alignment failed: {e}")

            # 6. Generate SRT
            self.status_label.config(text="Status: Generating SRT file...")
            self.root.update_idletasks()
            output_filename = os.path.splitext(self.audio_file_path)[0] + ".srt"
            self._generate_srt(aligned_result, output_filename, max_chars, max_lines)

        except RuntimeError as e:
            self.status_label.config(text=f"Status: Error - {e}")
            tk.messagebox.showerror("Processing Error", str(e))
        except ValueError as e:
            self.status_label.config(text=f"Status: Error - {e}")
            tk.messagebox.showerror("Input Error", str(e))
        except Exception as e:
            self.status_label.config(text="Status: An unexpected error occurred.")
            tk.messagebox.showerror("Error", f"An unexpected error occurred: {e}")
            import traceback
            print("--- Traceback ---") # For debugging
            traceback.print_exc()
            print("-----------------")
        finally:
            self.process_button.config(state=tk.NORMAL) # Re-enable btn

    def _start_processing(self):
        """Start processing in a separate thread."""
        # Run _process_audio in a thread to avoid GUI freeze
        processing_thread = threading.Thread(target=self._process_audio, daemon=True)
        processing_thread.start()


if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperXApp(root)
    sv_ttk.set_theme("dark")
    root.mainloop()
