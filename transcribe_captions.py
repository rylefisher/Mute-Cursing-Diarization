import concurrent.futures
from threading import Thread
from tkinter.colorchooser import askcolor
import stable_whisper
import json
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import subprocess
import shutil
import os
import moviepy.editor as mp
from process_files import *
from censorship import *
import re
from datetime import datetime, timedelta
import syncio

MODEL_SIZE = "turbo"
SPLIT_IN_MS = 30
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("loading model")
MODEL = stable_whisper.load_model(MODEL_SIZE, device="cuda")


def clean_path(path_str):
    path = Path(path_str)
    clean_name = re.sub(r"[^a-zA-Z0-9]+", "_", path.stem)
    clean_name = re.sub(r"_+", "_", clean_name)
    return path.with_stem(clean_name)


def copy_file_with_time_stamp(file_path):
    directory = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    name, ext = os.path.splitext(file_name)
    truncated_name = name[:5]
    current_time = datetime.now().strftime("%M%S")
    new_file_name = f"{truncated_name}_{current_time}{ext}"
    new_file_path = os.path.join(directory, new_file_name)
    shutil.copy(file_path, new_file_path)
    print(f"File copied to {new_file_path}")
    return new_file_path


def split_audio(audio_file, output_dir, segment_duration=SPLIT_IN_MS):
    # Prepare paths and output pattern
    audio_path = clean_path(audio_file)
    output_dir = audio_path.parent
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_pattern = str(output_dir / f"{audio_path.stem}_{timestamp}_%03d.wav")

    # Command for splitting the audio
    cmd = [
        "ffmpeg",
        "-hwaccel",
        "auto",  # Enable hardware acceleration
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-y",
        "-segment_time",
        str(segment_duration),
        "-c:a",
        "pcm_s16le",  # Uncompressed audio for best quality
        "-ac",
        "2",
        "-ar",
        "44100",
        output_pattern,
    ]

    # Execute the command with ffmpeg output directly to the console
    try:
        subprocess.run(cmd, check=True)
        print(f"Audio has been successfully split and saved to {output_dir}")
    except subprocess.CalledProcessError:
        print("Failed to split audio.")
        return []

    # Return sorted list of segment files
    segment_files = sorted(output_dir.glob(f"{audio_path.stem}_{timestamp}_*.wav"))
    return [str(file) for file in segment_files]


def video_to_audio(video_file, audio_file):
    video = mp.VideoFileClip(video_file)
    audio = video.audio
    audio.write_audiofile(audio_file)


def choose_color_hex():
    root = tk.Tk()
    root.withdraw()
    color = askcolor()
    root.destroy()
    if color[1]:
        return color[1]
    else:
        return None


def select_audio_or_video():
    while True:
        root = tk.Tk()
        root.withdraw()
        root.call("wm", "attributes", ".", "-topmost", "1")
        av_path = filedialog.askopenfilename(
            title="Select A/V files", filetypes=[("A/V files", "*.mp3 *.wav *.mp4")]
        )
        root.destroy()
        if not av_path:
            return None, False
        temp = copy_file_with_time_stamp(av_path)
        video_bi = {"status": False, "path": ""}
        video_path = ""
        if "mp4" in av_path or "mov" in av_path:
            ext = av_path[av_path.rfind(".") :]
            video_bi["status"] = True
            video_bi["path"] = av_path
            av_path = convert_video_to_audio(av_path, av_path.replace(".mp4", ".wav"))
        if av_path:
            print(f"Audio/Video file selected: {av_path}")
            folder = Path(av_path).parent / Path(av_path).stem
            folder.mkdir(parents=True, exist_ok=True)
            try:
                shutil.rmtree(folder)
            except Exception as e:
                print(str(e))
            folder.mkdir(parents=True, exist_ok=True)
            av_new = str(folder / Path(av_path).name)
            shutil.copy(av_path, clean_path(av_new))
            return av_new, video_bi
        return None, video_bi["status"]


def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    return Path(folder_path)


def convert_video_to_audio(video_file, audio_output):
    cmd = [
        "ffmpeg",
        "-hwaccel",
        "auto",
        "-y",
        "-i",
        video_file,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        audio_output,
    ]
    subprocess.run(cmd, check=True)
    return audio_output


def remove_audio_from_video(video_file, video_output):
    cmd = [
        "ffmpeg",
        "-hwaccel",
        "auto",
        "-y",
        "-i",
        video_file,
        "-c:v",
        "copy",
        "-an",  # Remove audio
        video_output,
    ]
    subprocess.run(cmd, check=True)


def combine_txt_files(txtfiles):
    txt_parts = ""
    newpath = ""
    for i, p in enumerate(txtfiles):
        newpath = p
        with open(p, "r") as f:
            txt_parts = txt_parts + f"\n\npart number {i}\n" + f.read()
    p = os.path.dirname(os.path.dirname(newpath))
    with open(p + "\\all_parts.txt", "w") as f:
        f.write(txt_parts)


def add_audio_to_video(video_file, audio_file, output_video):
    video_no_audio = video_file.replace(".mp4", "temp_.mp4")
    remove_audio_from_video(video_file, video_no_audio)
    cmd = [
        "ffmpeg",
        "-hwaccel",
        "auto",
        "-i",
        video_no_audio,
        "-i",
        audio_file,
        "-y",
        "-vcodec",
        "libx264",
        "-preset",
        "fast",  # Balanced preset for speed and quality
        "-crf",
        "23",  # Lower CRF for better quality
        "-c:a",
        "aac",
        "-b:a",
        "192k",  # Higher bitrate for improved audio quality
        "-ac",
        "2",
        "-ar",
        "44100",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        output_video,
    ]
    if os.path.exists(output_video):
        os.remove(output_video)
    subprocess.run(cmd, text=True)
    os.remove(video_no_audio)



class SrtProcessor: # Encapsulate methods within a class

    def parse_time(self, time_str):
        # Helper function to parse SRT time format HH:MM:SS,ms
        match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", time_str)
        if match:
            h, m, s, ms = map(int, match.groups())
            return timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)
        raise ValueError(f"Invalid time format: {time_str}")

    def format_time(self, td):
        # Helper function to format timedelta back to SRT time format HH:MM:SS,ms
        total_seconds = int(td.total_seconds())
        milliseconds = int(td.microseconds / 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    def add_time(self, time_str, seconds_to_add):
        # Adjusted add_time to handle seconds directly
        original_time = self.parse_time(time_str)
        new_time = original_time + timedelta(seconds=seconds_to_add)
        return self.format_time(new_time)

    def srt_combine(self, paths):
        global SPLIT_IN_MS # Declare intent to use the global variable
        combined_content = ""
        subtitle_number = 1
        additional_seconds = 0 # Changed variable name from additional_minutes to additional_seconds

        for index, file_path in enumerate(paths):
            if index > 0:
                # Increment offset by the global SPLIT_IN_MS value for subsequent files
                additional_seconds += SPLIT_IN_MS
                combined_content += "\n\n" # Maintain separation between file contents

            try: # Add error handling for file operations
                with open(file_path, "r", encoding="utf-8") as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"Warning: File not found {file_path}, skipping.")
                # If a file is missing, decide how to handle the time offset.
                # Option 1: Still increment time as if the segment existed (current behavior).
                # Option 2: Don't increment time (requires adjusting the increment logic).
                # Let's stick with Option 1 for now, meaning the gap remains.
                continue # Skip processing this file
            except Exception as e:
                print(f"Error reading file {file_path}: {e}, skipping.")
                continue # Skip processing this file


            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.isdigit():
                    # Check if the next line exists and contains '-->'
                    if i + 1 < len(lines) and "-->" in lines[i+1]:
                        combined_content += f"{subtitle_number}\n"
                        subtitle_number += 1
                        i += 1 # Move to the time line
                        time_line = lines[i].strip()
                        try: # Add error handling for time parsing/adding
                            start_time, end_time = time_line.split(" --> ")
                             # Use additional_seconds for time adjustment
                            combined_content += f"{self.add_time(start_time, additional_seconds)} --> {self.add_time(end_time, additional_seconds)}\n"
                        except ValueError as e:
                            print(f"Warning: Skipping invalid time format in {file_path} at line {i+1}: {time_line} ({e})")
                            # Skip the invalid entry (number, time, text)
                            subtitle_number -= 1 # Decrement subtitle number as it was invalid
                            while i + 1 < len(lines) and lines[i+1].strip() != "":
                                i += 1 # Skip text lines associated with the invalid time
                            i += 1 # Move past the blank line or end of file
                            continue # Continue to the next potential subtitle number
                        except Exception as e: # Catch other potential errors during time processing
                            print(f"Warning: Error processing time in {file_path} at line {i+1}: {time_line} ({e})")
                            subtitle_number -= 1
                            while i + 1 < len(lines) and lines[i+1].strip() != "":
                                i += 1
                            i += 1
                            continue
                    else:
                        # If it's a number but not followed by a time line, treat as text (or skip)
                        # For now, let's assume it might be part of subtitle text if malformed
                        combined_content += line + "\n"
                elif "-->" in line:
                     # Handle case where time line appears without preceding number (malformed SRT)
                     # We might choose to ignore these or try to process if needed.
                     # For simplicity, let's ignore lines starting with '-->' if not preceded by a number.
                     print(f"Warning: Skipping malformed time line without preceding number in {file_path}: {line}")
                     # Skip associated text lines until a blank line or new number
                     while i + 1 < len(lines) and lines[i+1].strip() != "" and not lines[i+1].strip().isdigit():
                         i += 1
                elif line: # Only add non-empty lines that are not numbers or timecodes
                    combined_content += line + "\n"
                # Handle blank lines separating subtitles - already handled by loop structure and \n concatenation

                i += 1
            # Ensure the last subtitle block ends with a newline if needed by the format
            if combined_content and not combined_content.endswith("\n\n"):
                 if not combined_content.endswith("\n"):
                     combined_content += "\n" # Add one newline if missing


        if not paths: # Handle case with no input paths
            print("Warning: No SRT files provided to combine.")
            return

        try: # Add error handling for output path generation/writing
            # Construct output path relative to the *first* input file's parent directory
            first_path = Path(paths[0])
            # Use stem of the first file for the output name
            output_name = f"{first_path.stem}_combined.srt"
            # Place the combined file in the parent directory of the first file
            output_file_path = first_path.parent.parent / output_name

            with open(output_file_path, "w", encoding="utf-8") as file:
                # Remove potential trailing blank lines before writing
                file.write(combined_content.strip())
            print(f"Combined SRT saved to: {output_file_path}") # Provide feedback
        except IndexError:
             print("Error: Input path list is empty, cannot determine output path.")
        except Exception as e:
             print(f"Error writing combined SRT file: {e}")


class AudioTranscriber:
    def __init__(self, model_size="large-v3", device="cuda"):
        global MODEL
        try:
            self.model = MODEL
        except Exception as e:
            print(
                f"Error loading model: {e} Dont panic, I got this. You should really use nvidia for good results."
            )
            self.model = stable_whisper.load_model("medium", device="cpu")
        self.audio_paths = []
        self.index = len(self.audio_paths) - 1
        self.clean_audio_paths = []
        self.srt_paths = []
        self.srt_paths_small = []
        self.clean_json = ""
        self.clean_json_paths = []
        self.srt_small = ""
        self.text_paths = []
        self.text_parts = 0

    def add_time(self, time_str, minutes=1):
        """Add minutes to SRT timestamp, adjusting for fractional seconds."""
        base_time = datetime.strptime(time_str.split(",")[0], "%H:%M:%S")
        milliseconds = int(time_str.split(",")[1]) if "," in time_str else 0
        added_time = base_time + timedelta(minutes=minutes, milliseconds=milliseconds)
        return added_time.strftime("%H:%M:%S,") + f"{milliseconds:03d}"

    def srt_combine(self, paths):
        global SPLIT_IN_MS
        processor = SrtProcessor()
        processor.srt_combine(paths)



    def transcribe_audio(self, audio_path, language="en", beam_size=5):
        """Transcribe the given audio file and return the transcription result."""
        self.audio_paths.append(audio_path)
        self.json_paths = []
        return self.model.transcribe(
            audio_path,
            compression_ratio_threshold=5,
            verbose=True,
            word_timestamps=True,
            language=language,
        )

    def save_transcription(self, audio_path, result, small=False):
        """Save the transcription to .srt and .json files based on the audio file path."""
        if small:
            audio_path = audio_path.replace(".wav", "_small.wav")
        if "wav" in audio_path:
            srt_path = audio_path.replace(".wav", ".srt")
            txt_path = audio_path.replace(".wav", ".txt")
            json_path = audio_path.replace(".wav", ".json")
        else:
            srt_path = audio_path.replace(".mp3", ".srt")
            json_path = audio_path.replace(".mp3", ".json")
        print("outputting transcript files")

        if not small:
            result.to_srt_vtt(srt_path, word_level=False)
            result.to_txt(txt_path)
            self.text_paths.append(txt_path)
            self.srt_paths.append(srt_path)
            self.text_parts += 1
        else:
            result.to_srt_vtt(srt_path, word_level=False)
            self.srt_small = srt_path

        result.to_txt(f"{srt_path}".replace(".srt", ".txt"))
        result.save_as_json(json_path)
        self.json_path = json_path
        print("completed transcript files")

        if small:
            self.srt_paths_small.append(srt_path)
        else:
            self.srt_paths.append(srt_path)

    def censor_cursing(self, audio_path):
        return censor_cursewords(audio_path, self.json_path)

    def transcribe_and_censor(self, audio_path):
        """Process an audio file, transcribe it and save the results."""
        result = self.transcribe_audio(audio_path)
        resultSmall = result
        result.split_by_length(max_chars=42)
        self.save_transcription(audio_path, result)
        resultSmall.split_by_length(max_chars=35)
        self.save_transcription(audio_path, resultSmall, True)
        aud, self.clean_json = self.censor_cursing(audio_path)
        self.clean_audio_paths.append(aud)
        self.clean_json_paths.append(self.clean_json)


def select_files():
    """This function uses tkinter to provide a GUI for selecting multiple audio or video files."""
    root = tk.Tk()
    root.withdraw()
    root.call("wm", "attributes", ".", "-topmost", "1")
    av_paths = filedialog.askopenfilenames(
        title="Select A/V files", filetypes=[("A/V files", "*.mp3 *.wav *.mp4")]
    )
    root.destroy()
    return list(av_paths)


def process_files(av_paths):
    results = []
    for av_path in av_paths:
        temp = copy_file_with_time_stamp(av_path)
        video_bi = {"status": False, "path": ""}
        if "mp4" in av_path or "mov" in av_path:
            ext = av_path[av_path.rfind(".") :]
            cmd = [
                "ffmpeg",
                "-hwaccel",
                "auto",
                "-i",
                av_path,
                "-y",
                "-vcodec",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-c:a",
                # Change from 'aac' to 'pcm_s16le' for uncompressed audio
                "pcm_s16le",
                "-ar",
                "44100",  # Ensure sample rate is appropriate for WAV
                "-ac",
                "2",  # Stereo channels
                temp,
            ]
            try:
                subprocess.run(cmd, check=True)
                av_path = temp
                video_path = av_path
                av_path = convert_video_to_audio(
                    av_path, av_path.replace(".mp4", ".wav")
                )  # Convert to audio
                video_bi["status"] = True
                video_bi["path"] = video_path
            except subprocess.CalledProcessError as e:
                print(f"Error processing file {av_path}: {e.stderr}")
                results.append((None, video_bi))

        if av_path:
            print(f"Audio/Video file selected: {av_path}")
            folder = Path(av_path).parent / Path(av_path).stem
            folder.mkdir(parents=True, exist_ok=True)
            try:
                shutil.rmtree(folder)
            except Exception as e:
                print(str(e))
            folder.mkdir(parents=True, exist_ok=True)
            av_new = str(folder / Path(av_path).name)
            shutil.copy(av_path, clean_path(av_new))
            results.append((av_new, video_bi))
        else:
            results.append((None, video_bi))
    return results


def main(audio_path, video_):
    global transcript_paths
    transcript_paths = []
    print("loading model")
    transcriber = AudioTranscriber(model_size=MODEL_SIZE, device="cuda")
    print("finished")
    log_ = JSONLog(audio_path)
    enums = split_audio(audio_path, "output")
    temp_folder = None
    if enums:
        for counter, audio_path in enumerate(enums):
            if not temp_folder and audio_path:
                temp_folder = Path(audio_path).parent.__str__()
            print("wav_file_path type:", type(audio_path))
            print("wav_file_path content:", audio_path)
            print(
                f"\n\nProcessing {audio_path}...@@@@@@@@@@@@@@@@@@@\n\nindex {counter+1} of {len(enums)}\n\n@@@@@@@@@@@@@@@@@@@\n\n"
            )
            transcriber.transcribe_and_censor(audio_path)

    else:
        print(f"Processing {audio_path}...")
        transcriber.transcribe_and_censor(audio_path)
    try:
        combine_txt_files(transcriber.text_paths)
    except Exception as e:
        print(str(e))

    comb_path = combine_wav_files(transcriber.clean_audio_paths)

    transcriber.srt_combine(transcriber.srt_paths)
    transcriber.srt_combine(transcriber.srt_paths_small)
    orig_video = ""
    new_video = ""
    processed_audio = ""
    orig_video = video_["path"]
    processed_audio = comb_path
    synchronizer_path = orig_video.replace(".mp4", "_clean2.mp4")
    new_video = orig_video.replace(".mp4", "_clean.mp4")

    if video_["status"]:
        add_audio_to_video(orig_video, processed_audio, new_video)
        synchronizer = syncio.VideoAudioSynchronizer(
            orig_video, processed_audio, synchronizer_path
        )
        synchronizer.ensure_no_leading_trailing_silence(comb_path)
        synchronizer.synchronize_audio()
    files_ = ""
    try:
        for root, dirs, files in os.walk(temp_folder):
            for file in files:
                if ".txt" in file:
                    with open(file, "r") as f:
                        file_temp = f.read()
                        files_ = files_ + file_temp
                    txt = Path(temp_folder).parent
                    txt = txt + f"\\{Path(audio_path).stem}.txt"
                    with open(txt, "w") as l:
                        l.write("".join(files_))
    except Exception as e:
        print(str(e))
    if temp_folder:
        try:
            shutil.rmtree(temp_folder)  # Remove temp folder after processing
            print("\n\nsuccessfully deleted temp\n\n")
        except Exception as e:
            print(f"Error deleting temp folder: {e}")
    print("\nits\ndone\nnow\n")


def handler():
    file_paths = select_files()
    print("the init process can take a moment, please be patient. starting now.")
    processed_data = process_files(file_paths)

    def process_audio_files(audio_paths, videos_):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=8
        ) as executor:  # using ThreadPoolExecutor for CPU thread optimizations
            futures = [
                executor.submit(main, audio_path, video_)
                for audio_path, video_ in zip(audio_paths, videos_)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    audio_paths = [data[0] for data in processed_data if data[0] is not None]
    videos_ = [data[1] for data in processed_data]
    process_audio_files(audio_paths, videos_)


if __name__ == "__main__":
    handler()
