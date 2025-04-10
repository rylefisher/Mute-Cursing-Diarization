import requests  # to handle downloads
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time
from urllib.parse import urlparse, parse_qs
import concurrent.futures 
import subprocess 

username = "rhyzohm"
page_url = f"https://twitchtracker.com/{username}/clips"
driver = uc.Chrome()
driver.get(page_url)
twitch_clips_links = []
time.sleep(5)
for _ in range(5):
    time.sleep(3)
    elements = driver.find_elements(By.XPATH, "//*[@data-litebox]")
    tmp = [element.get_attribute('data-litebox') for element in elements if "clips.twitch.tv" in element.get_attribute('data-litebox')]
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    twitch_clips_links = twitch_clips_links + tmp
# Pause for a few seconds to let the page load
my_clips = []
my_clips = [f"https{link}" for link in twitch_clips_links]
twitch_clips_links = []
clip_vals = []
array_without_duplicates = list(set(my_clips))

for clip_link in my_clips:
    parsed_url = urlparse(clip_link)
    query_params = parse_qs(parsed_url.query)
    clip_value = query_params.get('clip', [None])[0]
    url = f"https://www.twitch.tv/{username}/clip/" + clip_value
    twitch_clips_links.append(url)
    clip_vals.append(clip_value)

def download_clip(clip_link):
    try:
        result = subprocess.run(
            ["twitch-dl", "download", f"{clip_link}"],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"CLI error: {result.stderr}")

    except Exception as e:
        print(f"Failed to download from {clip_link}: {e}")


with open("clips.txt", "w") as f:
    for clip in twitch_clips_links:
        f.write(f"{clip}\n")
    
driver.quit()
