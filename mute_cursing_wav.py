import concurrent.futures
from threading import Thread
from tkinter.colorchooser import askcolor
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
from _globals import *
import sys
from transcriber_wx import WhisperXTranscriber

def clean_path(path_str):
    path = Path(path_str)
    clean_name = re.sub(r"[^a-zA-Z0-9]+", "_", path.stem)
    clean_name = re.sub(r"_+", "_", clean_name)
    return path.with_stem(clean_name)

# Function definition
def shorten_and_copy_file(file_path: str) -> str:
    # Get name, ext
    base_name: str = os.path.basename(file_path)
    name_part: str
    ext_part: str
    name_part, ext_part = os.path.splitext(base_name)
    # Get time hhmm
    time_str: str = datetime.now().strftime("%H%M")
    # Truncate name
    name_prefix: str = name_part[:4] # Max 4 chars
    # Combine parts
    new_filename: str = f"{name_prefix}{time_str}{ext_part}"
    # Get dir path
    dir_path: str = os.path.dirname(file_path)
    new_file_path: str = os.path.join(dir_path, new_filename)
    # Copy file
    try:
        shutil.copy2(file_path, new_file_path) # copy2 preserves meta
    except Exception as e:
        print(f"Error copying: {e}") # Basic error handle
        return file_path  # Return empty on fail
    return new_file_path

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


def split_audio(audio_file, output_dir, segment_duration=SPLIT_IN_MS, sr="44100"):
    # Prepare paths and output pattern
    audio_path = clean_path(audio_file)
    output_dir = audio_path.parent
    timestamp = datetime.now().strftime("%H%M%S")
    filename_patt = f"{audio_path.stem}_{timestamp}_%03d.wav"
    output_pattern = str(output_dir / filename_patt)

    # Command for splitting the audio
    cmd = [
        "ffmpeg",
        "-hwaccel",
        "auto",  # Enable hardware acceleration
        "-i",
        str(audio_file),
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
        sr,
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


class AudioTranscriber:
    def __init__(self, model_size="large-v3", device="cuda"):
        global MODEL

        self.transcriber = WhisperXTranscriber(
        diarize=True,  # Enable diarization
        hf_token=HF_TOKEN,  # Pass the token
        diarize_options={  # Optional: provide speaker hints
            # "min_speakers": 2,
            # "max_speakers": 2,
        },
        transcribe_config={"print_progress": True},
        align_config={"print_progress": True},
        )
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
        combined_content = ""
        subtitle_number = 1
        additional_minutes = 0

        for index, file_path in enumerate(paths):
            if index > 0:
                combined_content += (
                    "\n\n"  # Proper SRT separation between different files
                )
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()  # strip line-breaks to handle them manually
                if line.isdigit():
                    combined_content += f"{subtitle_number}\n"  # Add subtitle number
                    subtitle_number += 1
                elif "-->" in line:
                    start_time, end_time = line.split(" --> ")
                    combined_content += f"{self.add_time(start_time, additional_minutes)} --> {self.add_time(end_time, additional_minutes)}\n"
                else:
                    combined_content += line + "\n"  # Append other lines with newline
                i += 1
            additional_minutes += 1

        name = Path(paths[0]).stem
        output_file_prt = Path(paths[0]).parent.parent / f"{name}.srt"
        with open(str(output_file_prt), "w", encoding="utf-8") as file:
            file.write(combined_content)

    def transcribe_audio(self, audio_path, language="en", beam_size=5):
        """Transcribe the given audio file and return the transcription result."""
        self.audio_paths.append(audio_path)
        self.json_paths = []
        final_result = self.transcriber.process_audio(audio_path)['word_segments']
        return final_result

    def save_transcription(self, audio_path, result, small=False):
        """Save the transcription to .srt and .json files based on the audio file path."""
        if small:
            audio_path = audio_path.replace(".wav", "_small.wav")
        if "wav" in audio_path:
            json_path = audio_path.replace(".wav", ".json")
        else:
            json_path = audio_path.replace(".mp3", ".json")
        print("outputting transcript files")
        self.json_path = json_path
        with open(json_path, 'w') as f:
            json.dump(result, f)

    def transcribe_and_censor(self, audio_path):
        """Process an audio file, transcribe it and save the results."""
        result = self.transcribe_audio(audio_path)
        resultSmall = result
        self.save_transcription(audio_path, result)
        # Create the censorer instance
        censorer = AudioCensorer(
            censor_mode=CENSOR_MODE,
        )
        outdir = str(Path(audio_path).parent)
        aud, was_censored = censorer.censor_audio_file(
            audio_path, result, output_dir=outdir, log=True
        )
        temp_file = shorten_and_copy_file(aud)
        if was_censored:
            print('\n\nsecond go round\n\n')
            result = self.transcribe_audio(temp_file)
            self.save_transcription(temp_file, result)
            censorer = AudioCensorer(
                censor_mode=CENSOR_MODE,
            )
            aud2, was_censored = censorer.censor_audio_file(
                temp_file, result, output_dir=outdir, log=True
            )
            self.clean_audio_paths.append(aud2)
            self.clean_json_paths.append(self.clean_json)
            print(f"\nSuccess! Censored file saved to: {aud2}")
            return
        else:
            print('\n\nno cursing found\n\nexiting....')
            exit()
        if aud:
            print(f"\nSuccess! Censored file saved to: {aud}")
        self.clean_audio_paths.append(aud)
        self.clean_json_paths.append(self.clean_json)
        return

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


import signal


def cleanup():
    global temp_folder
    # Handle cleanup before exit
    try:
        shutil.rmtree(temp_folder)  # Remove temp folder after processing
        print("\n\nsuccessfully deleted temp\n\n")
    except Exception as e:
        print(f"Error deleting temp folder: {e}")


def signal_handler(sig, frame):
    # Intercept termination signals
    cleanup()
    sys.exit(0)


def main(audio_path):
    global transcript_paths, temp_folder
    transcript_paths = []
    print("loading model")
    transcriber = AudioTranscriber(model_size=MODEL_SIZE, device="cuda")
    print("finished")
    log_ = JSONLog(audio_path)
    # enums = split_audio(audio_path, "output", sr="44100")
    # enums = split_audio(audio_path, "output", sr='44100')
    temp_folder = None
    transcriber.transcribe_and_censor(audio_path)

    # comb_path = combine_wav_files(transcriber.clean_audio_paths)
    orig_video = ""

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
    for audio_path in file_paths:  # Iterating through data
        if audio_path:
            main(audio_path)


if __name__ == "__main__":
    # Register signal handlers for Windows (CTRL+C) and termination
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if os.name == "nt":  # Windows-specific
        signal.signal(signal.SIGBREAK, signal_handler)  # Handle CTRL+Break
    handler()
