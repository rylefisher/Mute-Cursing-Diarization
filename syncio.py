import subprocess

class VideoAudioSynchronizer:
    def __init__(self, video_file, processed_audio_file, output_file):
        self.video_file = video_file
        self.processed_audio_file = processed_audio_file
        self.output_file = output_file

    def synchronize_audio(self):
        video_duration = self.get_media_duration(self.video_file)
        audio_duration = self.get_media_duration(self.processed_audio_file)
        
        # Calculate the difference in duration
        duration_diff = video_duration - audio_duration
        if duration_diff > 0:
            # Pad the audio file if it is shorter than the video
            padded_audio_file = f"{self.processed_audio_file.split('.')[0]}_padded.wav"
            self.pad_audio(self.processed_audio_file, padded_audio_file, duration_diff)
            final_audio_file = padded_audio_file
        else:
            final_audio_file = self.processed_audio_file

        # Combine the video with the synchronized or original audio
        cmd = [
            'ffmpeg',
            '-i', self.video_file,
            '-i', final_audio_file,
            '-c:v', 'copy',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            '-y',
            self.output_file
        ]
        subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    def get_media_duration(self, file_path):
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            raise ValueError(f"Could not determine duration for file: {file_path}")

    def pad_audio(self, input_audio, output_audio, pad_duration):
        cmd = [
            'ffmpeg',
            '-i', input_audio,
            '-af', f"apad=pad_dur={pad_duration}",
            '-y',
            output_audio
        ]
        subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    @staticmethod
    def ensure_no_leading_trailing_silence(audio_file):
        cmd = [
            'ffmpeg',
            '-i', audio_file,
            '-af', 'afade=t=in:ss=0:d=0.5,afade=t=out:st=duration-0.5:d=0.5',
            '-y',
            audio_file.replace('.wav', '_nofade.wav')
        ]
        subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

# # Example usage:
# video_path = 'path/to/video.mp4'
# processed_audio_path = 'path/to/processed_audio.wav'
# output_video_path = 'path/to/output_video.mp4'

# synchronizer = VideoAudioSynchronizer(video_path, processed_audio_path, output_video_path)
# synchronizer.ensure_no_leading_trailing_silence(processed_audio_path)  # Ensuring no silences at the ends
# synchronizer.synchronize_audio()
