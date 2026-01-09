import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    import cv2
except ImportError:
    print("Please install opencv-python: pip install opencv-python")
    exit(1)


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds. Returns -1 on error."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return -1
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        if fps > 0:
            return frame_count / fps
        return -1
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return -1


def find_videos_in_folder(folder_path: str, extensions: List[str] = ['.mp4', '.avi', '.mkv', '.mov']) -> List[str]:
    """Find all video files in a folder."""
    video_files = []
    for ext in extensions:
        video_files.extend(Path(folder_path).glob(f'*{ext}'))
        video_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
    return [str(v) for v in video_files]


def find_videos_recursive(root_folder: str, extensions: List[str] = ['.mp4', '.avi', '.mkv', '.mov']) -> List[Tuple[str, str]]:
    """Find videos in subfolders. Returns list of (folder_id, video_path)."""
    videos = []
    root_path = Path(root_folder)
    subfolders = [f for f in root_path.iterdir() if f.is_dir()]
    
    if not subfolders:
        return []
    
    for subfolder in subfolders:
        folder_id = subfolder.name
        video_files = find_videos_in_folder(str(subfolder), extensions)
        for video_file in video_files:
            videos.append((folder_id, video_file))
    
    return videos


def collect_durations(input_folder: str, mode: str = 'auto') -> Dict:
    """
    Collect durations from videos.
    
    Args:
        input_folder: Input folder path
        mode: 'flat', 'nested', or 'auto'
    
    Returns:
        Dictionary with durations and video info
    """
    durations = []
    video_info = []
    
    if mode == 'auto':
        root_path = Path(input_folder)
        subfolders = [f for f in root_path.iterdir() if f.is_dir()]
        direct_videos = find_videos_in_folder(input_folder)
        
        if direct_videos:
            mode = 'flat'
        elif subfolders:
            mode = 'nested'
        else:
            print(f"No videos or subfolders found in {input_folder}")
            return {'durations': [], 'video_info': [], 'mode': mode}
    
    if mode == 'flat':
        video_files = find_videos_in_folder(input_folder)
        print(f"Found {len(video_files)} videos")
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            duration = get_video_duration(video_path)
            if duration > 0:
                durations.append(duration)
                video_info.append({
                    'path': video_path,
                    'filename': Path(video_path).name,
                    'duration': duration,
                    'duration_str': f"{int(duration//60)}:{int(duration%60):02d}"
                })
    
    elif mode == 'nested':
        videos = find_videos_recursive(input_folder)
        print(f"Found {len(videos)} videos in subfolders")
        
        for folder_id, video_path in tqdm(videos, desc="Processing videos"):
            duration = get_video_duration(video_path)
            if duration > 0:
                durations.append(duration)
                video_info.append({
                    'folder_id': folder_id,
                    'path': video_path,
                    'filename': Path(video_path).name,
                    'duration': duration,
                    'duration_str': f"{int(duration//60)}:{int(duration%60):02d}"
                })
    
    return {
        'durations': durations,
        'video_info': video_info,
        'mode': mode
    }


def plot_duration_statistics(durations: List[float], output_path: str = 'duration_statistics.png', 
                             title: str = 'Video Duration Statistics'):
    """Plot duration statistics with 4 subplots: histogram, boxplot, cumulative distribution, and summary."""
    if not durations:
        print("No data to plot")
        return
    
    durations_arr = np.array(durations)
    durations_minutes = durations_arr / 60
    
    stats = {
        'mean': np.mean(durations_minutes),
        'median': np.median(durations_minutes),
        'std': np.std(durations_minutes),
        'min': np.min(durations_minutes),
        'max': np.max(durations_minutes),
        'q25': np.percentile(durations_minutes, 25),
        'q75': np.percentile(durations_minutes, 75)
    }
    
    fig = plt.figure(figsize=(16, 10))
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(durations_minutes, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.2f} min')
    ax1.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats["median"]:.2f} min')
    ax1.set_xlabel('Duration (minutes)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Histogram of Video Durations', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 2, 2)
    bp = ax2.boxplot(durations_minutes, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_edgecolor('black')
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)
    ax2.set_ylabel('Duration (minutes)', fontsize=12)
    ax2.set_title('Box Plot of Video Durations', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = plt.subplot(2, 2, 3)
    sorted_durations = np.sort(durations_minutes)
    cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations) * 100
    ax3.plot(sorted_durations, cumulative, linewidth=2, color='darkblue')
    ax3.axhline(50, color='red', linestyle='--', alpha=0.5, label='50%')
    ax3.axhline(90, color='orange', linestyle='--', alpha=0.5, label='90%')
    ax3.set_xlabel('Duration (minutes)', fontsize=12)
    ax3.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax3.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    stats_text = f"""
    SUMMARY STATISTICS
    {'='*40}
    Total videos:     {len(durations)}
    
    Mean:             {stats['mean']:.2f} min
    Median:           {stats['median']:.2f} min
    Std Dev:          {stats['std']:.2f} min
    
    Min:              {stats['min']:.2f} min
    Max:              {stats['max']:.2f} min
    
    Q1 (25%):         {stats['q25']:.2f} min
    Q3 (75%):         {stats['q75']:.2f} min
    
    Total Duration:   {np.sum(durations_minutes):.2f} min
                      ({np.sum(durations_minutes)/60:.2f} hours)
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser(
        description='Video duration statistics and visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,)
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to folder containing videos')
    parser.add_argument('--output', '-o', type=str, default='duration_statistics.png',
                       help='Output plot path (default: duration_statistics.png)')
    parser.add_argument('--mode', '-m', type=str, choices=['flat', 'nested', 'auto'],
                       default='auto',
                       help='Mode: flat (direct videos), nested (subfolders), auto (detect)')
    parser.add_argument('--title', '-t', type=str, default='Video Duration Statistics',
                       help='Plot title')
    parser.add_argument('--save-csv', type=str, default=None,
                       help='Save detailed info to CSV file (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Folder does not exist: {args.input}")
        return
    
    result = collect_durations(args.input, mode=args.mode)
    durations = result['durations']
    video_info = result['video_info']
    
    if not durations:
        print("No valid videos found!")
        return
    
    plot_duration_statistics(durations, args.output, args.title)
    
    if args.save_csv:
        import csv
        with open(args.save_csv, 'w', newline='', encoding='utf-8') as f:
            if video_info:
                fieldnames = video_info[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(video_info)
        print(f"Saved details to: {args.save_csv}")


if __name__ == '__main__':
    main()
    
# python src/analysis/analysis_duration.py --input data/Next_QA/NExTVideo --output statis_data/duration_stats_full_nextqa.png
