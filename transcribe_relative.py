import os
import re
import time
import stable_whisper

channel = "xanderhal"
TEMP_DIR = os.getcwd() + f"/{channel}/"


class LogWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(self.filepath, 'w') as f:
            f.write('')

    def add(self, content):
        with open(self.filepath, 'a') as f:
            f.write(content)


class VideoTranscriber:
    def __init__(self, model_name):
        self.model = stable_whisper.load_model(model_name)

    @staticmethod
    def make_valid_filename(name):
        return re.sub(r'[^a-zA-Z0-9]', '_', name)

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

    def convert_and_transcribe(self, wav_files, username):
        logger = LogWriter(os.path.join(TEMP_DIR, 'transcription_log.html'))
        logger.add(f"Transcription started at {time.ctime()}")

        for wav_file in wav_files:
            file_name = os.path.splitext(os.path.basename(wav_file))[0]
            url = f"https://twitch.tv/{username}/clip/{file_name}"

            try:
                contents = self.transcribe_audio_to_files(wav_file, file_name)
                if contents:
                    logger.add(f"<h2>Transcribing {file_name}</h2>")
                    logger.add(f'<a href="{url}" target="_blank">{url}</a><br>{contents}<br>')
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        return logger.filepath


def main():
    global channel
    model_name = 'base'
    wav_files = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if f.endswith('.wav')]

    transcriber = VideoTranscriber(model_name)
    log_path = transcriber.convert_and_transcribe(wav_files, channel)
    print(f"Logs can be found at {log_path}")


if __name__ == "__main__":
    main()
