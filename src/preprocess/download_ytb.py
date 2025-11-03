import json
import argparse
from tqdm import tqdm
from pathlib import Path
from yt_dlp import YoutubeDL

def parse_args():
    parser = argparse.ArgumentParser(description="Download YouTube videos based on JSON data.")
    parser.add_argument("--data-file", type=str, required=True, help="Path to the JSON file containing video IDs.")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save downloaded videos.")
    return parser.parse_args()

def main():
    list_error = []
    args = parse_args()
    data_json = json.load(open(args.data_file, "r"))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    data_ids = set(item["video_id"] for item in data_json)

    for video_id in tqdm(data_ids, total=len(data_ids)):
        url = f"https://www.youtube.com/watch?v={video_id}"

        if Path(save_dir / f"{video_id}.en.srt").exists():
            print(f"Video {video_id} already downloaded.")
            continue

        ydl_opts = {
            'outtmpl': str(save_dir / f"{video_id}.%(ext)s"),
            'format': 'mp4/bestvideo+bestaudio/best',
            # Download subtitles
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'subtitlesformat': 'srt',
        }
        
        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print(f"Error downloading {video_id}: {e}")
            list_error.append(url)
            continue

        # input("Press Enter to continue...")
    
    print(list_error)

if __name__ == "__main__":
    main()
    

# python src/preprocess/download_ytb.py --data-file minerva.json --save-dir data/videos/
