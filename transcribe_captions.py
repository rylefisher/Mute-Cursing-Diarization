import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import whisperx
import datetime
import os
import threading
import torch # Added for device check

class WhisperXApp:
    """GUI App for WhisperX SRT generation."""

    def __init__(self, root_window):
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
        srt_content = []
        entry_index = 1

        if not result or "segments" not in result:
            self.status_label.config(text="Status: Error - No segments found")
            tk.messagebox.showerror("Error", "Transcription result is empty or invalid.")
            return

        for segment in result["segments"]:
            if 'start' not in segment or 'end' not in segment or 'text' not in segment:
                print(f"Skipping invalid segment: {segment}") # Debug print
                continue # Skip malformed segments

            segment_text = segment["text"].strip()
            if not segment_text:
                continue # Skip empty segments

            start_time = self._format_time(segment['start'])
            end_time = self._format_time(segment['end'])

            # Split text by max chars
            split_lines = self._split_line(segment_text, max_chars)

            # Group lines by max lines per entry
            for i in range(0, len(split_lines), max_lines):
                chunk = split_lines[i:i + max_lines]
                srt_entry = f"{entry_index}\n"
                srt_entry += f"{start_time} --> {end_time}\n"
                srt_entry += "\n".join(chunk) + "\n\n"
                srt_content.append(srt_entry)
                entry_index += 1
                # Note: All chunks from one segment share the same timestamp

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(srt_content)
            self.status_label.config(text=f"Status: SRT saved to {os.path.basename(output_path)}")
            tk.messagebox.showinfo("Success", f"SRT file generated successfully:\n{output_path}")
        except IOError as e:
            self.status_label.config(text="Status: Error writing SRT file")
            tk.messagebox.showerror("File Error", f"Could not write SRT file: {e}")
        except Exception as e:
             self.status_label.config(text="Status: Unknown SRT writing error")
             tk.messagebox.showerror("Error", f"An unexpected error occurred during SRT writing: {e}")


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
        self.root.update_idletasks() # Force UI update

        try:
            # 1. Load base model (if not loaded)
            if self.model is None:
                self.status_label.config(text="Status: Loading base model...")
                self.root.update_idletasks()
                self.model = whisperx.load_model("large-v3-turbo", self.device, compute_type=self.compute_type)

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
            try:
                result = self.model.transcribe(audio, batch_size=self.batch_size)
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
                     self.align_model, self.align_metadata = whisperx.load_align_model(language_code=detected_language, device=self.device)
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
    root.mainloop()