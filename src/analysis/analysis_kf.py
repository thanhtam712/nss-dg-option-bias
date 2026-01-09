import json
import argparse
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_video_ids_from_json(json_file):
    """
    Load unique video_ids from JSON file.
    
    Args:
        json_file: Path to JSON file
        
    Returns:
        set: Set of unique video_ids
    """
    if not json_file:
        return None
        
    json_path = Path(json_file)
    
    if not json_path.exists():
        raise ValueError(f"JSON file does not exist: {json_file}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    video_ids = set(item['video_id'] for item in data if 'video_id' in item)
    
    return video_ids


def count_keyframes_per_folder(root_folder, video_ids_filter=None):
    """
    Count the number of keyframes in each ID folder.
    
    Args:
        root_folder: Path to the root folder containing ID folders
        video_ids_filter: Set of video_ids to filter (only count these folders), None to count all
        
    Returns:
        dict: Dictionary with folder_id as key and keyframe count as value
        list: List of missing folders (empty if video_ids_filter is None)
    """
    root_path = Path(root_folder)
    keyframe_counts = {}
    missing_folders = []
    
    if not root_path.exists():
        raise ValueError(f"Folder does not exist: {root_folder}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    # If filter is provided, only check those folders
    if video_ids_filter:
        print(f"📊 Counting keyframes for {len(video_ids_filter)} video_ids from JSON...")
        
        for video_id in sorted(video_ids_filter):
            folder_path = root_path / video_id
            
            if folder_path.exists() and folder_path.is_dir():
                # Count image files in the ID folder
                image_files = [f for f in folder_path.iterdir() 
                              if f.is_file() and f.suffix.lower() in image_extensions]
                keyframe_counts[video_id] = len(image_files)
            else:
                # Folder doesn't exist
                missing_folders.append(video_id)
                keyframe_counts[video_id] = 0
    else:
        # Count all folders
        print("📊 Counting keyframes in all folders...")
        
        for id_folder in sorted(root_path.iterdir()):
            if id_folder.is_dir():
                # Count image files in the ID folder
                image_files = [f for f in id_folder.iterdir() 
                              if f.is_file() and f.suffix.lower() in image_extensions]
                keyframe_counts[id_folder.name] = len(image_files)
    
    return keyframe_counts, missing_folders


def generate_statistics(keyframe_counts):
    """
    Generate statistical summary of keyframe counts.
    
    Args:
        keyframe_counts: Dictionary with folder_id and keyframe count
        
    Returns:
        dict: Statistical summary
    """
    counts = list(keyframe_counts.values())
    
    if not counts:
        return {
            'total_folders': 0,
            'total_keyframes': 0,
            'mean': 0,
            'median': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'q1': 0,
            'q3': 0
        }
    
    stats = {
        'total_folders': len(keyframe_counts),
        'total_keyframes': sum(counts),
        'mean': np.mean(counts),
        'median': np.median(counts),
        'std': np.std(counts),
        'min': np.min(counts),
        'max': np.max(counts),
        'q1': np.percentile(counts, 25),
        'q3': np.percentile(counts, 75)
    }
    
    return stats


def plot_statistics(keyframe_counts, stats, missing_folders, output_file='./keyframe_statistics.png'):
    """
    Create comprehensive visualization with histogram, box plot, and statistics.
    
    Args:
        keyframe_counts: Dictionary with folder_id and keyframe count
        stats: Statistical summary dictionary
        missing_folders: List of missing folders
        output_file: Path to save the plot
    """
    counts = list(keyframe_counts.values())
    
    if not counts:
        print("⚠️  No data to plot")
        return
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with two subplots and extra space for text
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1], 
                          hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Histogram
    ax2 = fig.add_subplot(gs[0, 1])  # Box plot
    ax3 = fig.add_subplot(gs[1, :])  # Statistics text
    ax3.axis('off')
    
    # Histogram
    ax1.hist(counts, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Number of Keyframes', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Frequency (Number of Folders)', fontsize=14, fontweight='bold')
    ax1.set_title('Frequency Distribution of Keyframes per Folder', 
                 fontsize=16, fontweight='bold')
    
    # Add mean and median lines
    ax1.axvline(stats['mean'], color='red', linestyle='--', linewidth=2,
               label=f'Mean: {stats["mean"]:.2f}')
    ax1.axvline(stats['median'], color='green', linestyle='--', linewidth=2,
               label=f'Median: {stats["median"]:.2f}')
    
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    bp = ax2.boxplot(counts, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    ax2.set_ylabel('Number of Keyframes', fontsize=14, fontweight='bold')
    ax2.set_title('Box Plot of Keyframes per Folder', 
                 fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved statistics plot: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize keyframe statistics from ID folders'
    )
    parser.add_argument(
        '--keyframes_folder',
        type=str,
        required=True,
        help='Path to the root folder containing ID folders with keyframes'
    )
    parser.add_argument(
        '--json_file',
        type=str,
        default=None,
        help='(Optional) Path to JSON file containing video_ids to filter'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='./keyframe_statistics.png',
        help='Path to save output image (default: ./keyframe_statistics.png)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("KEYFRAME STATISTICS ANALYZER")
    print("=" * 80)
    print(f"\nKeyframes folder: {args.keyframes_folder}")
    if args.json_file:
        print(f"JSON filter file: {args.json_file}")
    else:
        print("JSON filter file: None (analyzing all folders)")
    print(f"Output file: {args.output_file}\n")
    
    # Load video_ids from JSON if provided
    video_ids = None
    if args.json_file:
        video_ids = load_video_ids_from_json(args.json_file)
    
    # Count keyframes
    keyframe_counts, missing_folders = count_keyframes_per_folder(
        args.keyframes_folder, 
        video_ids_filter=video_ids
    )
    
    if missing_folders:
        print(f"⚠️  Warning: {len(missing_folders)} folders not found in keyframes directory")
    
    stats = generate_statistics(keyframe_counts)
    
    plot_statistics(keyframe_counts, stats, missing_folders, args.output_file)

if __name__ == "__main__":
    main()

# python src/statis/statis_kf.py --keyframes_folder data/keyframes_full/ --output_file statis_data/keyframes_full_kf.png
# python src/statis/statis_kf.py --keyframes_folder data/keyframes_new/ --json_file data/baseline_caption_outline_openai.json --output_file statis_data/keyframes_new_kf.png
