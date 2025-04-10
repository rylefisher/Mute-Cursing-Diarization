import subprocess
import concurrent.futures

def downloader(cliplink):
        # Using CLI to download video
        result = subprocess.run(
            ["TwitchDownloaderCLI.exe", "clipdownload", "--id", cliplink, "-o", f"{cliplink}.mp4"],
            capture_output=True, text=True)
        # Print the result stdout and stderr
        print(f"CLI stdout for {cliplink}:\n{result.stdout}")





twitch_clips_links = ['TallObservantMoonBIRB-YsbS9mjWD-k8a7c1']
downloader(twitch_clips_links[0])
# Use ThreadPoolExecutor to download multiple videos concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    executor.map(downloader, twitch_clips_links)
