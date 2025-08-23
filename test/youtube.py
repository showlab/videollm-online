import youtube_dl  # This import statement should be before using the library

def download_video(url):
    # Define the options for youtube-dl
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Prioritize MP4 format with audio, then best available
        'outtmpl': '%(title)s.%(ext)s',  # Output file template
        'noplaylist': True,  # Avoid downloading playlists
        'merge_output_format': 'mp4'  # If audio/video are downloaded separately, merge them
    }

    # Create a youtube-dl object
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        # Download the video
        info_dict = ydl.extract_info(url, download=False)
        # Get the video title to display
        video_title = info_dict.get("title", "video")
        print(f"Downloading: {video_title}")
        ydl.download([url])

# URL of the YouTube video you want to download
video_url = 'https://www.youtube.com/watch?v=KwNUJ69RbwY'

# Call the function to download the video
download_video(video_url)