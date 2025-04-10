import threading
import os
import re
import subprocess
import requests
import tkinter as tk
from tkinter import ttk
import undetected_chromedriver as uc
from urllib.parse import urlparse, parse_qs
import concurrent.futures
import time
import stable_whisper
from sv_ttk import set_theme
import webbrowser
import yt_dlp
import json
from tqdm import tqdm
from ssl import SSLError
import sys
sys.stdout.reconfigure(encoding='utf-8')

NUMB_THREADS = 4
TEMP_BASE_DIR = os.getcwd()

# Ensure yt-dlp is up-to-date
subprocess.run(['yt-dlp', '-U'], capture_output=True, text=True)


class WebDriverClipHandler:
    def __init__(self, username):
        self.username = username
        self.driver = uc.Chrome()
        self.clips = []

    def fetch_clips(self, scrolls=5, skips=0):
        base_url = f"https://twitchtracker.com/{self.username}/clips"
        self.driver.get(base_url)
        time.sleep(5)
        twitch_clips_links = []

        try:
            scrolls = int(scrolls.strip())  # Ensure scrolls is an integer
            skips = int(skips)  # Ensure skips is an integer
        except ValueError:
            scrolls = 5

        for i in range(scrolls):
            elements_before_scroll = len(self.driver.find_elements(By.XPATH, "//*[@data-litebox]"))

            if not self.scroller(self.driver, 0.8):  # Scroll and detect changes
                print("Reached the end of the page or no change in page content.")
                break

            elements_after_scroll = len(self.driver.find_elements(By.XPATH, "//*[@data-litebox]"))
            if elements_after_scroll == elements_before_scroll:
                print("No more new elements found.")
                break

            if i < skips:
                continue

            elements = self.driver.find_elements(By.XPATH, "//*[@data-litebox]")
            tmp = [f"https{element.get_attribute('data-litebox')}" for element in elements if "clips.twitch.tv" in element.get_attribute('data-litebox')]
            twitch_clips_links += tmp

        self.clips = list(set(twitch_clips_links))  # Remove duplicates

    def scroller(self, driver, scroll_pause_time):
        scroll_height = driver.execute_script("return document.documentElement.scrollHeight")
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(scroll_pause_time)
        new_scroll_height = driver.execute_script("return document.documentElement.scrollHeight")
        return new_scroll_height != scroll_height  # Return if the scroll height has changed

    def get_clip_urls(self):
        clipids_to_urls = {}
        for clip_link in self.clips:
            parsed_url = urlparse(clip_link)
            query_params = parse_qs(parsed_url.query)
            clip_value = query_params.get('clip', [None])[0]
            if clip_value:
                url = f"https://www.twitch.tv/{self.username}/clip/{clip_value}"
                clipids_to_urls[clip_value] = url
        return clipids_to_urls

    def quit(self):
        self.driver.quit()


def open_in_default_browser(filepath):
    if os.path.exists(filepath):
        abs_path = os.path.abspath(filepath)
        webbrowser.open(f'file://{abs_path}')
    else:
        print("File does not exist")


def get_clip_urls(channel_name):
    """Retrieve all clip URLs for the given channel."""
    command = ["twitch-dl", "clips", channel_name, "--all"]

    # Ensure UTF-8 encoding is used for capturing output
    result = subprocess.run(command, capture_output=True, encoding='utf-8')

    if result.returncode != 0:
        raise Exception(f"Error fetching clips: {result.stderr}")

    urls = []
    for line in result.stdout.splitlines():
        if line.startswith("https://clips.twitch.tv/"):
            urls.append(line.strip())

    return urls


class TwitchClipDownloader:
    def __init__(self, urls):
        self.urls = urls
        self.cookiefile = r"D:\whisp - Copy\cookies-01.txt"
        self.vidpath = []

    def extract_clip_id(self, url):
        # Use regex to extract the clip ID from the URL
        match = re.search(r'clip/([^/?]+)', url)
        if match:
            return match.group(1)
        elif "clips.twitch.tv/" in url:
            return url.split('clips.twitch.tv/')[1]

    def download_clip(self, url):
        clip_id = self.extract_clip_id(url)  # Extract the clip ID
        if not clip_id:
            print(f"Could not extract clip ID from URL: {url}")
            return

        # Modify download command to use clip ID for file naming
        download_command = [
            "yt-dlp",
            "--cookies", self.cookiefile,
            "-o", f"{clip_id}.%(ext)s",  # Use clip_id for the filename
            url
        ]

        result = subprocess.run(download_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error downloading clip: {result.stderr}")

        # Assuming the filename from the download follows the format {clip_id}.mp4
        video_filename = f"{clip_id}.mp4"
        if os.path.exists(video_filename):
            self.vidpath.append(video_filename)

    def convert_to_wav_and_delete(self, video_filename):
        wav_filename = os.path.splitext(video_filename)[0] + '.wav'
        conversion_command = f'ffmpeg -i "{video_filename}" "{wav_filename}"'
        if os.system(conversion_command) == 0:
            os.remove(video_filename)
            return wav_filename
        return None

    def download_all_clips(self):
        threads = []
        for url in self.urls:
            thread = threading.Thread(target=self.download_clip, args=(url,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        return self.vidpath
    
    
class LogWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.empty_file()

    def empty_file(self):
        with open(self.filepath, 'w') as file:
            pass

    def add(self, content):
        with open(self.filepath, 'a', errors='replace', encoding='utf-8') as file:
            file.write(content + '\n')


class Mp4ToWavConverter:
    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.renamed_files = []

    def fetch_mp4_files(self):
        # Fetch all .mp4 files in the working directory
        return [f for f in os.listdir(self.working_dir) if f.endswith('.mp4')]

    def rename_file(self, filename):
        # Rename the file to be alphanumeric
        new_name = re.sub(r'\W+', '_', os.path.splitext(filename)[0]) + '.mp4'
        try:
            os.rename(os.path.join(self.working_dir, filename), os.path.join(self.working_dir, new_name))
        except FileExistsError:
            pass  # Ignore if file already renamed
        self.renamed_files.append(new_name)
        return new_name

    def convert_to_wav(self, mp4_file):
        wav_file = os.path.splitext(mp4_file)[0] + '.wav'
        full_mp4_path = os.path.join(self.working_dir, mp4_file)
        full_wav_path = os.path.join(self.working_dir, wav_file)

        try:
            # Use subprocess with timeout
            subprocess.run(["ffmpeg", "-i", full_mp4_path, full_wav_path], check=True, capture_output=True, timeout=5)
            os.remove(full_mp4_path)  # Delete the original mp4 file
            return full_wav_path
        except subprocess.CalledProcessError as e:
            print(f"Error converting {mp4_file}: {e.stderr.decode().strip()}")
        except subprocess.TimeoutExpired:
            print(f"Timeout expired when converting {mp4_file}")
        except Exception as e:
            print(f"Unexpected error with {mp4_file}: {e}")

        try:
            os.remove(full_mp4_path)  # Attempt to delete mp4 if error occurs
        except FileNotFoundError:
            pass
        return None  # Return None to indicate a failed conversion

    def batch_convert(self):
        mp4_files = self.fetch_mp4_files()

        # Rename files first
        for mp4_file in mp4_files:
            self.rename_file(mp4_file)

        wav_paths = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Use tqdm for progress bar
            with tqdm(total=len(self.renamed_files), desc="Converting MP4 to WAV") as pbar:
                futures = {executor.submit(self.convert_to_wav, mp4): mp4 for mp4 in self.renamed_files}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()  # Get result from future
                        if result:  # Only append successful conversions
                            wav_paths.append(result)
                    except Exception as e:
                        print(f"Error processing file: {future}")
                    pbar.update(1)  # Update progress bar for each completed conversion
        return wav_paths


class VideoTranscriber:
    def __init__(self, model_name):
        self.model = stable_whisper.load_model(model_name)

    @staticmethod
    def make_valid_filename(name):
        return re.sub(r'[^a-zA-Z0-9]', '_', name)

    def find_wav_files(self):
        # Locate all .wav files in the working directory
        return [f for f in os.listdir('.') if f.endswith('.wav')]

    def convert_and_transcribe(self):
        wav_paths = self.find_wav_files()  # Fetch all wav files
        logger = LogWriter(os.path.join(TEMP_DIR, 'transcription_log.html'))
        logger.add(f"Transcription started at {time.ctime()}")

        with tqdm(total=len(wav_paths), desc="Transcribing") as pbar:
            for wav_path in wav_paths:
                file_name = os.path.basename(wav_path)
                base_name = os.path.splitext(file_name)[0]
                try:
                    contents = self.transcribe_audio_to_files(wav_path, base_name)
                    if contents:
                        logger.add(f"<br><br><h2>Transcribing {file_name}</h2><br><br>")
                        logger.add(f'<br>{file_name}:<br>')
                        logger.add(f'<br>{contents}<br>')
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
                pbar.update(1)

        # Handle deletion of all .wav files after transcription is done
        for wav_path in wav_paths:
            try:
                os.remove(wav_path)
            except FileNotFoundError:
                pass

        return logger.filepath

    def transcribe_audio_to_files(self, audio_path, base_name):
        if not os.path.exists(audio_path):
            print(f"Audio path does not exist: {audio_path}")
            return False

        result = self.model.transcribe(audio_path)

        t1 = os.path.join(TEMP_DIR, f"{base_name}.txt")
        t2 = os.path.join(TEMP_DIR, f"{base_name}.srt")
        result.to_txt(t1)
        result.to_srt_vtt(t2)
        return result.to_txt()
    
class VideoConverter:
    def __init__(self, folder_name=None):
        # Convert provided folder name to a full path within the current working directory
        if folder_name is not None:
            self.folder_path = os.path.join(os.getcwd(), folder_name)
        else:
            self.folder_path = os.getcwd()
        self.wav_paths = []

    def make_valid_filename(self, name):
        # Replace invalid filename characters with underscores
        return re.sub(r'[^a-zA-Z0-9]', '_', name)

    def convert_videos_to_wav(self):
        # List all files in the directory
        video_files = [file for file in os.listdir(self.folder_path) if file.endswith('.mp4')]
        print("Starting conversion of video files to WAV format...")

        # Initialize the tqdm progress bar
        for file in tqdm(video_files, desc="Converting videos"):
            # Generate a safe filename
            base_name = os.path.splitext(file)[0]
            safe_base_name = self.make_valid_filename(base_name)

            # Define full paths
            video_path = os.path.join(self.folder_path, file)
            wav_filename = safe_base_name + '.wav'
            wav_path = os.path.join(self.folder_path, wav_filename)

            # Convert to WAV using ffmpeg
            conversion_command = [
                "ffmpeg", "-i", video_path, wav_path
            ]

            # Execute the conversion command
            result = subprocess.run(conversion_command, capture_output=True, text=True)

            if result.returncode == 0:
                # Add the path to the list if conversion was successful
                self.wav_paths.append(wav_path)
                # Print the status of each file conversion
                print(f"Converted: {file} -> {wav_filename}")
                # Remove the original video file
                os.remove(video_path)
            else:
                # If conversion fails, print the error
                print(f"Failed to convert {file}: {result.stderr}")

        print("Conversion complete.")
        return self.wav_paths


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.config_file = "config.json"
        self.load_last_values()
        self.title("Twitch Clips Downloader & Transcriber")
        self.geometry("400x460")
        self.font = ("Arial", 14)

        self.username_label = tk.Label(self, text="Twitch Username:", font=self.font)
        self.username_label.pack(pady=5)
        self.username_entry = tk.Entry(self, font=self.font)
        self.username_entry.pack(pady=5, fill='x', padx=20)
        self.username_entry.insert(0, self.config.get("username", ""))

        self.pages_label = tk.Label(self, text="Number of Pages of Clips:", font=self.font)
        self.pages_label.pack(pady=5)
        self.pages_entry = tk.Entry(self, font=self.font)
        self.pages_entry.pack(pady=5, fill='x', padx=20)
        self.pages_entry.insert(0, self.config.get("pages", '5'))

        self.skips_label = tk.Label(self, text="Skip Pages (for Resuming):", font=self.font)
        self.skips_label.pack(pady=5)
        self.skips_entry = tk.Entry(self, font=self.font)
        self.skips_entry.pack(pady=5, fill='x', padx=20)
        self.skips_entry.insert(0, self.config.get("skips", '0'))

        self.model_label = tk.Label(self, text="Model (smaller == for smaller GPUs):", font=self.font)
        self.model_label.pack(pady=5)
        self.model_var = tk.StringVar(value=self.config.get("model", "medium"))
        self.model_dropdown = ttk.Combobox(self, textvariable=self.model_var,
                                           values=["large-v3", "medium", "base", "tiny"],
                                           state="readonly", font=self.font)
        self.model_dropdown.pack(pady=5, fill='x', padx=20)

        buttons_frame = tk.Frame(self)
        buttons_frame.pack(pady=20, fill='x', padx=20)

        # Updated button setup
        self.download_button = ttk.Button(buttons_frame, text="Download Clips", command=self.download_clips)
        self.download_button.pack(side='left', expand=True, fill='x')  # Call `self.download_clips` instead

        self.convert_button = ttk.Button(buttons_frame, text="Convert Clips", command=self.convert_clips)
        self.convert_button.pack(side='left', expand=True, fill='x')  # Call `self.convert_clips` instead

        self.transcribe_button = ttk.Button(buttons_frame, text="Transcribe Audio", command=self.transcribe_audio)
        self.transcribe_button.pack(side='left', expand=True, fill='x')  # Call `self.transcribe_audio` instead

    def load_last_values(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}

    def save_values(self):
        self.config["username"] = self.username_entry.get().strip()
        self.config["pages"] = self.pages_entry.get().strip()
        self.config["skips"] = self.skips_entry.get().strip()
        self.config["model"] = self.model_var.get()
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f)

    def download_clips(self):
        self.save_values()
        username = self.username_entry.get().strip()
        pages = self.pages_entry.get().strip()
        skips = self.skips_entry.get().strip()

        if not username:
            print("Username required")
            return

        global TEMP_DIR
        TEMP_DIR = os.path.join(TEMP_BASE_DIR, username)
        os.makedirs(TEMP_DIR, exist_ok=True)
        print(f'Temporary directory created {TEMP_DIR}')

        print('Downloading clips')  # Process to download clips

        print('Opening a browser to get links')
        urls = get_clip_urls(username)
        print('Downloading clips')
        downloader = TwitchClipDownloader(urls)
        downloader.download_all_clips()
        
    def convert_clips(self):
        print('Converting clips')  # Implement the logic to convert downloaded clips
        paths = Mp4ToWavConverter(os.getcwd()).batch_convert()

    def transcribe_audio(self):
        model_name = self.model_var.get()
        transcriber = VideoTranscriber(model_name)
        logger_path = transcriber.convert_and_transcribe()  # Ensure this logic is implemented correctly
        open_in_default_browser(logger_path)

    def start_process(self):
        self.save_values()
        pages = self.pages_entry.get().strip()
        skips = self.skips_entry.get().strip()
        username = self.username_entry.get().strip()
        model_name = self.model_var.get()

        if not username:
            print("Username required")
            return

        global TEMP_DIR
        TEMP_DIR = os.path.join(TEMP_BASE_DIR, username)
        os.makedirs(TEMP_DIR, exist_ok=True)
        print(f'Temporary directory created {TEMP_DIR}')

        try:
            skips = int(skips.strip())
        except ValueError:
            skips = 0

        print('Opening a browser to get links')
        # urls = get_clip_urls(username)
        print('Downloading clips')
        # downloader = TwitchClipDownloader(urls)
        # downloader.download_all_clips()
        # paths = Mp4ToWavConverter(os.getcwd()).batch_convert()
        # print('Transcribing clips')
        transcriber = VideoTranscriber(model_name)
        logger_path = transcriber.convert_and_transcribe()
        open_in_default_browser(logger_path)

    def on_closing(self):
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
