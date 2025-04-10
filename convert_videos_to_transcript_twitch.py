import os
import re
import subprocess
import stable_whisper

# 1. Load the Whisper model
model = stable_whisper.load_model('large-v3')
print('model loaded')
# Function to convert file name to only contain text and numbers
def make_valid_filename(name):
    return re.sub(r'[^a-zA-Z0-9]', '_', name)

# Function to transcribe an audio file
def transcribe_audio_to_files(audio_path, base_name):
    result = model.transcribe(audio_path)
    result.to_txt(f"{base_name}.txt")


# 2. Get the current working directory
p = 'E:\\download\\ryzoh'
paths = []
# 3. Walk through all files in the directory
for file_name in os.listdir('E:\\download\\ryzoh'):
    if file_name.endswith(('.mp4', '.mkv', '.avi', '.mov')):  # Add other video formats if needed
        paths.append(file_name)

for i, file_name in enumerate(paths):
    # 4. Rename file to contain only text and numbers
    base_name = make_valid_filename(os.path.splitext(file_name)[0])
    new_video_path = os.path.join(p, base_name + os.path.splitext(file_name)[1])
    os.rename(os.path.join(p, file_name), new_video_path)

    # 5. Convert video to .wav using ffmpeg
    wav_path = os.path.join(p, base_name + ".wav")
    subprocess.run(["ffmpeg", "-i", new_video_path, wav_path], check=True)

    # 6. Transcribe the audio and save to .txt and .srt
    transcribe_audio_to_files(wav_path, base_name)
