import json, os, argparse, subprocess, random, torchvision
import concurrent.futures
try:
    torchvision.set_video_backend('video_reader')
except:
    import av # otherwise, check if av is installed

def download_video(video_id, video_url, output_dir, ffmpeg_location=None):
    output_path = os.path.join(output_dir, f'{video_id}.mp4')
    if os.path.exists(output_path):
        try:
            ffmpeg_cmd = ["ffmpeg", "-v", "error", "-i", output_path, "-f", "null", "-"]
            if ffmpeg_location:
                ffmpeg_cmd[0] = os.path.join(ffmpeg_location, "ffmpeg")
            subprocess.run(ffmpeg_cmd, check=True)
            print(f'{output_path} has been downloaded and verified...')
            return
        except:
            print(f'{output_path} may be broken. Downloading it again...')
            os.remove(output_path)
    cmd = ["yt-dlp", "--username", "oauth2", "--password", "", "-f", "mp4", "-o", output_path, video_url]
    if ffmpeg_location:
        cmd.extend(["--ffmpeg-location", ffmpeg_location])
    subprocess.run(cmd, check=True)

def main(output_dir, json_path, num_workers, ffmpeg_location):
    annotations = json.load(open(json_path, 'r'))['database']
    annotations = list(annotations.items())
    random.shuffle(annotations)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(download_video, video_id, annotation['video_url'], output_dir, ffmpeg_location) for video_id, annotation in annotations]
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download videos in parallel using yt-dlp')
    parser.add_argument('--output_dir', type=str, default='datasets/coin/videos', help='Directory to save downloaded videos')
    parser.add_argument('--json_path', type=str, default='datasets/coin/coin.json', help='Path to the JSON file containing video data')
    parser.add_argument('--ffmpeg', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel downloads')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    main(args.output_dir, args.json_path, args.num_workers, args.ffmpeg)