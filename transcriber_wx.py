import torch
import whisperx
import gc
import os
import numpy as np
import scipy.io.wavfile as wavfile  # For dummy audio creation
from _globals import *

class WhisperXTranscriber:
    """Handles transcription, alignment, and diarization."""

    def __init__(
        self,
        whisper_model_name: str = "large-v3-turbo",
        language_code: str = "en",
        device: str | None = None,
        compute_type: str = "float16",
        batch_size: int = 16,
        hf_token: str | None = None,  # HF token for diarization
        diarize: bool = False,  # Flag to enable diarization
        asr_options: dict | None = None,
        vad_options: dict | None = None,
        transcribe_config: dict | None = None,
        align_config: dict | None = None,
        diarize_options: dict | None = None,  # min/max speakers
    ):
        """Initialize the transcriber."""
        self.whisper_model_name = whisper_model_name
        self.language_code = language_code
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Use float16 on CUDA, int8 on CPU unless specified
        self.compute_type = compute_type if self.device == "cuda" else "int8"
        self.batch_size = batch_size
        self.hf_token = hf_token  # Store HF token
        self.diarize = diarize  # Store diarization flag

        # Default configurations
        self._default_asr_options = {
            # Lower = more words accepted
            "log_prob_threshold": None,
            # Lower = less likely silence
            "no_speech_threshold": 0.01,
            # Sampling temperatures
            "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            # Use prev text as prompt
            "condition_on_previous_text": True,
            # Disable compression check
            "compression_ratio_threshold": None,
            # Enable word timestamps
            "word_timestamps": True,
            # Output with timestamps
            "without_timestamps": False,
            # Num beams for search
            "beam_size": 10,
            # Candidates per segment
            "best_of": 10,
            # Disable patience factor
            "patience": None,
            # Neutral length penalty
            "length_penalty": 1.0,
            # No repetition penalty
            "repetition_penalty": 1.0,
            # Allow repeating n-grams
            "no_repeat_ngram_size": 0,
            # Reset prompt condition
            "prompt_reset_on_temperature": 0.5,
            # Optional initial prompt
            "initial_prompt": None,
            # Optional token prefix
            "prefix": None,
            # Do not suppress blanks
            "suppress_blank": False,
            # Do not suppress tokens
            "suppress_tokens": [],
            # Max initial timestamp
            "max_initial_timestamp": 1.0,
            # Chars prepended to words
            "prepend_punctuations": "\"'“¿([{-",
            # Chars appended to words
            "append_punctuations": "\"'.。,，!！?？:：”)]}、",
            # Do not suppress numbers
            "suppress_numerals": False,
            # Unlimited new tokens
            "max_new_tokens": None,
            # Timestamp clipping method
            "clip_timestamps": "0",
            # Disable hallucination filter
            "hallucination_silence_threshold": None,
            # Optional hotwords list
            "hotwords": None,
        }
        self._default_vad_options = {
            "vad_threshold": 0.1,
            # Min duration for speech (ms)
            "min_speech_duration_ms": 50,
            # Max duration for speech (s)
            "max_speech_duration_s": float("inf"),
            # Merge segments gap (ms)
            "min_silence_duration_ms": 700,
            # VAD processing window size
            "window_size_samples": 1024,
            # Padding around speech (ms)
            "speech_pad_ms": 500,
        }
        self._default_transcribe_config = {
            "chunk_size": 30,
            "print_progress": False,
        }
        self._default_align_config = {
            "return_char_alignments": False,
            "print_progress": False,
        }
        self._default_diarize_options = {
            "min_speakers": None,
            "max_speakers": None,
        }

        # Merge user options with defaults
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

        if self.diarize and not self.hf_token:
            # Warn if diarization enabled without token
            print("Warning: Diarization enabled but no Hugging Face token provided.")
            # Consider raising an error if token is strictly required

        print(f"Using device: {self.device}")

    def _load_models(self, load_diarization: bool = False):
        """Load necessary models lazily."""
        # Load VAD and Whisper model if needed
        if self.whisper_model is None:
            print("Loading VAD model...")
            self.vad_model = whisperx.vad.load_vad_model(self.device)
            print("VAD model loaded.")

            print(f"Loading Whisper model: {self.whisper_model_name}...")
            # Ensure asr_options are passed correctly
            effective_asr_options = {
                k: v for k, v in self.asr_options.items() if v is not None
            }
            self.whisper_model = whisperx.load_model(
                self.whisper_model_name,
                self.device,
                compute_type=self.compute_type,
                language=self.language_code or None,  # Pass None if lang not set
                asr_options=effective_asr_options,
                vad_options=self.vad_options,
                vad_model=self.vad_model,
                task="transcribe",
            )
            print("Whisper model loaded.")

        # Load Alignment model if needed
        if self.align_model is None:
            print("Loading alignment model...")
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=self.language_code, device=self.device
            )
            print("Alignment model loaded.")

        # Load Diarization model if requested and not already loaded
        if load_diarization and self.diarize_model is None:
            if not self.hf_token:
                print("Skipping diarization model load: Hugging Face token missing.")
                return  # Cannot load without token
            print("Loading diarization model...")
            try:
                self.diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=self.hf_token, device=self.device
                )
                print("Diarization model loaded.")
            except Exception as e:
                print(f"Error loading diarization model: {e}")
                print("Check your Hugging Face token and network connection.")
                self.diarize_model = None  # Ensure it's None if loading failed

    def _load_audio(self, audio_path: str):
        """Load audio file."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        print(f"Loading audio: {audio_path}...")
        audio = whisperx.load_audio(audio_path)
        print("Audio loaded.")
        return audio

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio file."""
        self._load_models(load_diarization=False)  # Don't need diarization model yet
        audio = self._load_audio(audio_path)

        print("Starting transcription...")
        transcribe_args = {
            "batch_size": self.batch_size,
            "language": self.language_code,  # Pass language code
            **self.transcribe_config,
        }
        # Filter out None values from transcribe_args if necessary
        transcribe_args = {k: v for k, v in transcribe_args.items() if v is not None}

        result = self.whisper_model.transcribe(audio, **transcribe_args)
        print("Transcription finished.")
        return result

    def align(self, transcription_result: dict, audio_path: str) -> dict:
        """Align transcription results."""
        # Load models if not already loaded (includes alignment model)
        self._load_models(load_diarization=False)
        audio = self._load_audio(audio_path)  # Load audio for alignment

        print("Starting alignment...")
        align_args = {
            "model": self.align_model,
            "align_model_metadata": self.align_metadata,
            "device": self.device,
            **self.align_config,
        }
        result_aligned = whisperx.align(
            transcription_result["segments"], audio=audio, **align_args
        )
        print("Alignment finished.")
        return result_aligned

    def diarize_audio(self, audio_path: str, aligned_result: dict) -> dict:
        """Perform diarization and assign speakers."""
        if not self.diarize:
            print("Diarization disabled, skipping.")
            return aligned_result  # Return original aligned result

        # Ensure diarization model is loaded
        self._load_models(load_diarization=True)
        if self.diarize_model is None:
            print("Cannot perform diarization: Model not loaded (check token/errors).")
            return aligned_result  # Return original if model failed to load

        audio = self._load_audio(audio_path)  # Load audio for diarization

        print("Starting diarization...")
        # Filter None values from diarize_options before passing
        diarize_run_options = {
            k: v for k, v in self.diarize_options.items() if v is not None
        }
        try:
            diarize_segments = self.diarize_model(audio, **diarize_run_options)
            print("Diarization finished.")

            print("Assigning speakers to words...")
            # Ensure aligned_result has segments before proceeding
            if "segments" not in aligned_result or not aligned_result["segments"]:
                print(
                    "Warning: No segments found in alignment result for speaker assignment."
                )
                return aligned_result

            result_with_speakers = whisperx.assign_word_speakers(
                diarize_segments, aligned_result
            )
            print("Speaker assignment finished.")
            return result_with_speakers
        except Exception as e:
            print(f"Error during diarization or speaker assignment: {e}")
            return aligned_result  # Return aligned result on error

    def process_audio(self, audio_path: str) -> dict:
        """Perform full transcription, alignment, and optional diarization."""
        # 1. Transcription
        transcription_result = self.transcribe(audio_path)
        if (
            "segments" not in transcription_result
            or not transcription_result["segments"]
        ):
            print("Warning: Transcription produced no segments.")
            return transcription_result  # Return early if no segments

        # 2. Alignment
        aligned_result = self.align(transcription_result, audio_path)
        if "segments" not in aligned_result or not aligned_result["segments"]:
            print("Warning: Alignment produced no segments.")
            # Decide whether to return transcription or aligned result here
            return aligned_result  # Return aligned result even if empty for consistency

        # 3. Diarization (if enabled)
        final_result = self.diarize_audio(audio_path, aligned_result)

        return final_result

    def print_aligned_segments(self, final_result: dict):
        """Print aligned word segments neatly, including speaker if available."""
        print("\nProcessed Segments (Aligned/Diarized):")
        if not final_result or "segments" not in final_result:
            print("No segments found in the result.")
            return

        for segment in final_result.get("segments", []):
            # Combine words into segment text for context
            segment_text = segment.get("text", "[Segment text not found]")
            segment_start = segment.get("start")
            segment_end = segment.get("end")
            segment_start_str = (
                f"{segment_start:.2f}s" if segment_start is not None else "N/A"
            )
            segment_end_str = (
                f"{segment_end:.2f}s" if segment_end is not None else "N/A"
            )

            print(f"\n--- Segment [{segment_start_str} -> {segment_end_str}] ---")
            # print(f"Text: {segment_text}") # Optional: print full segment text

            if "words" not in segment or not segment["words"]:
                print("  (No word timings in this segment)")
                continue

            for word_info in segment.get("words", []):
                start = word_info.get("start")
                end = word_info.get("end")
                word = word_info.get("word", "[UNKNOWN]")
                score = word_info.get("score")
                speaker = word_info.get("speaker")  # Get speaker info

                start_str = f"{start:.2f}s" if start is not None else "N/A"
                end_str = f"{end:.2f}s" if end is not None else "N/A"
                score_str = f"(Score: {score:.2f})" if score is not None else ""
                speaker_str = (
                    f"[{speaker}]" if speaker is not None else ""
                )  # Format speaker

                print(
                    f"  {speaker_str:<10} [{start_str} -> {end_str}] {word} {score_str}"
                )

    def print_configuration(self, audio_path: str | None = None):
        """Print the current configuration."""
        print("\n--- Current Configuration ---")
        if audio_path:
            print(f"Audio File: {audio_path}")
        print(f"Device: {self.device}")
        print(f"Compute Type: {self.compute_type}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Whisper Model: {self.whisper_model_name}")
        print(f"Language: {self.language_code}")
        print(f"Diarization Enabled: {self.diarize}")
        if self.diarize:
            print(f"Hugging Face Token Provided: {'Yes' if self.hf_token else 'No'}")

        print("\nASR Options:")
        for key, value in self.asr_options.items():
            print(f"  {key}: {value}")
        print("\nVAD Options:")
        for key, value in self.vad_options.items():
            print(f"  {key}: {value}")
        print("\nTranscription Config:")
        for key, value in self.transcribe_config.items():
            print(f"  {key}: {value}")
        print("\nAlignment Config:")
        align_print_config = {
            k: v
            for k, v in self.align_config.items()
            if k not in ["model", "align_model_metadata"]
        }
        for key, value in align_print_config.items():
            print(f"  {key}: {value}")
        if self.diarize:
            print("\nDiarization Options:")
            for key, value in self.diarize_options.items():
                print(f"  {key}: {value}")
        print("-----------------------------")

    def cleanup(self):
        """Release resources."""
        print("Cleaning up models...")
        # Use try-except for individual deletions to prevent cascade failure
        try:
            del self.whisper_model
        except AttributeError:
            pass
        try:
            del self.align_model
        except AttributeError:
            pass
        try:
            del self.vad_model
        except AttributeError:
            pass
        try:
            del self.diarize_model  # Clean up diarization model
        except AttributeError:
            pass

        self.whisper_model = None
        self.align_model = None
        self.align_metadata = None
        self.vad_model = None
        self.diarize_model = None

        gc.collect()  # Explicit garbage collection
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()  # Clear CUDA cache
            except Exception as e:
                print(f"Error clearing CUDA cache: {e}")
        print("Cleanup complete.")

    def __del__(self):
        """Ensure cleanup on object deletion."""
        self.cleanup()


# Example usage:
if __name__ == "__main__":
    # --- Configuration ---
    audio_file = "test.wav"  # Make sure this file exists or is created
    output_dir = "output_dir"  # Example (not directly used by class)

    # --- !!! IMPORTANT: Set your Hugging Face Token !!! ---
    # Best practice: Use an environment variable
    # hf_token = os.environ.get("HF_TOKEN")
    # Or set it directly (less secure):
    try:
        transcriber = WhisperXTranscriber(
            diarize=True,  # Enable diarization
            hf_token=HF_TOKEN,  # Pass the token
            diarize_options={  # Optional: provide speaker hints
                # "min_speakers": 2,
                # "max_speakers": 2,
            },
            transcribe_config={"print_progress": True},
            align_config={"print_progress": True},
        )

        # Print the final configuration being used
        transcriber.print_configuration(audio_path=audio_file)

        # Check if diarization can proceed
        if transcriber.diarize and not transcriber.hf_token:
            print("\n*** Diarization is enabled, but no HF token was provided. ***")
            print("*** Speaker assignment will be skipped.              ***")

        # Process the audio file
        final_result = transcriber.process_audio(audio_file)['word_segments']

        # Print the results with speaker info
        transcriber.print_aligned_segments(final_result)

        # Cleanup is handled by __del__ or can be called explicitly
        # transcriber.cleanup()

        print("\nProcessing finished.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Consider more specific exception handling
