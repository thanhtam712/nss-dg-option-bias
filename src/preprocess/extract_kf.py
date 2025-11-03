import os
import sys
import random
import shutil
import torch
import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append("/mlcv3/WorkingSpace/Personal/baotg/VBS/source/modules/encode_kf/models/beit3/")
from beit3 import encode_image, load_model

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, help="Input folder name")
    parser.add_argument("--o", type=str, help="Output folder name")
    parser.add_argument("--s", type=int, help="Frame step")
    parser.add_argument("--input", type=str, default=None, help="Input file list")
    parser.add_argument("--skip-lock", action="store_true", help="Skip locked videos")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Threshold for keyframe similarity",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="similarity",
        choices=["similarity", "uniform"],
        help="Extraction mode: 'similarity' uses cosine similarity threshold, 'uniform' extracts frames at regular intervals",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames to extract uniformly (only used with --mode uniform)",
    )

    return parser.parse_args()


def load_model_and_device():
    """Loads the BEiT3 model and sets the device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, _ = load_model(
        device=device,
        checkpoint="/mlcv3/WorkingSpace/Personal/baotg/TTam/Grad_Thesis/src/models/beit3/weights/beit3_large_patch16_384_coco_retrieval.pt",
        sentencepiece_model="/mlcv3/WorkingSpace/Personal/baotg/TTam/Grad_Thesis/src/models/beit3/weights/beit3.spm",
        input_size=384,
        model_name="beit3_large_patch16_384",
    )
    return model, processor, device


def check_thresholds(output_dir, video_dir, thresholds):
    """Check which thresholds need to be processed."""
    final_thresholds = []

    for threshold in thresholds:
        threshold_str = str(int(threshold * 100))
        keyframe_path = output_dir / "keyframes" / video_dir / f"{threshold_str}.npy"
        low_dir = output_dir / "low" / video_dir / threshold_str

        if keyframe_path.exists() and low_dir.exists():
            keyframe_count = np.load(keyframe_path).shape[0]
            image_count = len(list(low_dir.glob("*/*.jpg")))

            if keyframe_count == image_count and keyframe_count != 0:
                continue

            shutil.rmtree(low_dir, ignore_errors=False)
            print("Removed keyframes:", low_dir)

        final_thresholds.append(threshold)

    return final_thresholds


def process_video_uniform(
    video_path,
    frame_step,
    output_dir,
    num_frames=None,
):
    """
    Processes a video to extract keyframes uniformly at regular intervals.
    """
    print(f"Processing video (uniform mode): {video_path}")

    video_dir = video_path.stem
    cap = cv.VideoCapture(str(video_path))
    
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame step if num_frames is specified
    if num_frames is not None:
        frame_step = max(1, total_frames // num_frames)
        print(f"Extracting {num_frames} frames uniformly, frame_step: {frame_step}")
    
    dir_main_high = output_dir / video_dir
    dir_main_high.mkdir(parents=True, exist_ok=True)
    
    frame_idx = 0
    extracted_count = 0
    
    while cap.isOpened():
        ret, cv_frame = cap.read()
        
        if not ret:
            break
        
        if frame_idx % frame_step == 0:
            image_high = Image.fromarray(cv.cvtColor(cv_frame, cv.COLOR_BGR2RGB))
            name_save = f"{str(frame_idx).zfill(6)}.jpg"
            
            try:
                image_high.save(dir_main_high / name_save)
                extracted_count += 1
            except Exception as e:
                print(f"Error saving frame {frame_idx}: {e}")
        
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {extracted_count} frames from {video_path.name}")


def process_video(
    video_path,
    threshold: float,
    frame_step,
    output_dir,
    model,
    processor,
    device,
):
    """
    Processes a video to extract keyframes using similarity threshold.
    """

    print(f"Processing video (similarity mode): {video_path}")

    video_dir = video_path.stem
    # msb_df = pd.read_csv(msb_path, sep="\t")
    cap = cv.VideoCapture(str(video_path))

    total_frame_read = 0
    kf_features = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        # for idx, row in tqdm(msb_df.iterrows(), total=len(msb_df)):
        for idx, row in enumerate(tqdm(range(1))):
            # start_frame, end_frame = row["startframe"], row["endframe"]
            start_frame, end_frame = cap.get(cv.CAP_PROP_POS_FRAMES), cap.get(cv.CAP_PROP_FRAME_COUNT)

            last_frame, last_kf = None, None

            dir_main_high = output_dir / video_dir
            dir_main_high.mkdir(parents=True, exist_ok=True)

            while cap.isOpened():
                ret, cv_frame = cap.read()

                if not ret:
                    break

                if (
                    total_frame_read >= start_frame
                    and total_frame_read % frame_step == 0
                ):
                    width, height = cv_frame.shape[1], cv_frame.shape[0]

                    # resize but always keep the shorter side to 384, keep the ratio
                    if width > height:
                        height_to_resize = 384
                        width_to_resize = int(width * height_to_resize / height)
                    else:
                        width_to_resize = 384
                        height_to_resize = int(height * width_to_resize / width)

                    resized_frame = cv.resize(
                        cv_frame,
                        (width_to_resize, height_to_resize),
                        interpolation=cv.INTER_AREA,
                    )

                    image_low = Image.fromarray(
                        cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)
                    )

                    image_low_feat = encode_image(
                        model=model,
                        processor=processor,
                        image=image_low,
                        device=device,
                        return_torch=True,
                    )
                    name_save = f"{str(total_frame_read).zfill(6)}.jpg"

                    if (
                        last_frame is None
                        or (
                            last_kf is not None
                            and F.cosine_similarity(image_low_feat, last_kf).item()
                            < threshold
                        )
                        or F.cosine_similarity(image_low_feat, last_frame).item()
                        < threshold
                    ):
                        image_high = Image.fromarray(
                            cv.cvtColor(cv_frame, cv.COLOR_BGR2RGB)
                        )

                        try:
                            image_high.save(dir_main_high / name_save)
                        except Exception:
                            pass

                        last_kf = image_low_feat.clone()
                        kf_features.append(
                            image_low_feat.squeeze().detach().cpu().numpy()
                        )

                    last_frame = image_low_feat.clone()

                total_frame_read += 1

                if total_frame_read >= end_frame:
                    break

    cap.release()


def main():
    args = parse_arguments()

    folder_name = Path(args.f)
    output_dir = Path(args.o)
    frame_step = args.s
    threshold = args.threshold
    mode = args.mode
    num_frames = args.num_frames
    
    # Load model only if using similarity mode
    if mode == "similarity":
        model, processor, device = load_model_and_device()
    else:
        model, processor, device = None, None, None

    # Load video list
    if args.input:
        with open(args.input, "r") as f:
            video_list = [line.strip() for line in f if line.strip()]
    else:
        video_list = os.listdir(folder_name)

    random.shuffle(video_list)

    # Process videos
    print(f"Total videos: {len(video_list)}")
    print(f"Extraction mode: {mode}")
    if mode == "uniform" and num_frames:
        print(f"Target frames per video: {num_frames}")
    else:
        print(f"Frame step: {frame_step}")

    for _, video_id in enumerate(video_list):
        z = video_id
        if "." in z:
            z = video_id.split(".")[0]

        video_id = video_id.split(".")[0]
        video_path = next(
            (
                folder_name / f"{video_id}.{ext}"
                for ext in [
                    "mp4",
                    "mkv",
                    "avi",
                    "mov",
                    "m4v",
                    "flv",
                    "wmv",
                    "webm",
                    "mpe",
                ]
                if (folder_name / f"{video_id}.{ext}").exists()
            ),
            None,
        )
        
        if video_path:
            print(video_path)

            if mode == "uniform":
                process_video_uniform(
                    video_path,
                    frame_step,
                    output_dir,
                    num_frames,
                )
            else:  # similarity mode
                process_video(
                    video_path,
                    threshold,
                    frame_step,
                    output_dir,
                    model,
                    processor,
                    device,
                )
        else:
            print("Video not found:", video_id)


if __name__ == "__main__":
    main()

# Example usage:
# Similarity mode (default): python src/preprocess/extract_kf.py --f data/videos_filtered --o data/keyframes_new --s 10
# Uniform mode with frame step: python src/preprocess/extract_kf.py --f data/videos_filtered --o data/keyframes_new --s 30 --mode uniform
# Uniform mode with fixed number of frames: python src/preprocess/extract_kf.py --f data/videos_filtered --o data/keyframes_new --mode uniform --num-frames 100
