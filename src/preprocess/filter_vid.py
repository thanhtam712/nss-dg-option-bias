import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from moviepy import VideoFileClip


def parse_args():
    parser = argparse.ArgumentParser(description="Filter video files")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input video folder")
    # parser.add_argument("--output", type=Path, required=True, help="Path to the output video folder")
    return parser.parse_args()

def get_video_duration_moviepy(file_path):
    """
    Gets the duration of a video file in seconds using moviepy.
    """
    try:
        clip = VideoFileClip(str(file_path))
        duration_seconds = clip.duration
        clip.close()  # Important to close the clip
        return duration_seconds
    except Exception as e:
        print(f"Error getting video duration for {file_path} with moviepy: {e}")
        return None
    
def main():
    args = parse_args()
    input_folder = args.input
    count = 0
    # output_folder = args.output
    # output_folder.mkdir(parents=True, exist_ok=True)

    statis_duration = []
    
    for video_file in input_folder.glob("**/*.mp4"):
        duration = get_video_duration_moviepy(video_file)
        
        if duration is not None:
            
            if duration < 300:
                count += 1
            
                statis_duration.append(duration)
            print(f"Video {video_file.name} duration: {duration} seconds")
        else:
            print(video_file)
            print("Failed to get video duration.")
            continue
    print(f"Total videos under 300 seconds: {count}")

        # if duration < 300:
        #     shutil.copy(video_file, output_folder / video_file.name)
        #     if (video_file.with_suffix(".en.srt")).exists():
        #         shutil.copy(video_file.with_suffix(".en.srt"), output_folder / (video_file.name.replace(".mp4", ".en.srt")))
        #     print(f"Video duration: {duration} seconds")
        # else:
        #     print("Over 300 seconds")

    plt.hist(statis_duration, 30)
    plt.savefig("duration_histogram.png")
    
if __name__ == "__main__":
    main()
    
