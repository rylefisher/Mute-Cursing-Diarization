import yt_dlp

def download_twitch_clip_audio(url: str, output_file: str): 
    ydl_opts = {
        'format': 'bestaudio/best',   
        'outtmpl': output_file,      
        'noplaylist': True,           
        'extractaudio': True,         
        'postprocessors': [{          
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',  
            'preferredquality': '192', 
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])  

# Example usage:
twitch_clip_url = 'https://www.twitch.tv/bowblax/clip/VivaciousSmallIguanaFeelsBadMan'
output_audio_file = 'twitch_clip_audio.mp3'
download_twitch_clip_audio(twitch_clip_url, output_audio_file)
