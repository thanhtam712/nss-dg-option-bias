import re
import json
import cv2 as cv
from pathlib import Path
from typing import List, Dict, Tuple
from natsort import natsorted
from PIL import Image, ImageDraw
import numpy as np


def extract_keyframes(video_path: str, output_dir: str, keyframe_indices: List[int], keyframe_bboxes: List[List[Dict[str, any]]]) -> Dict[int, str]:
    """
    Extracts keyframes from a video and saves them as images with bounding boxes drawn.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted keyframes will be saved.
        keyframe_indices (List[int]): List of frame indices to extract as keyframes.
        keyframe_bboxes (List[List[Dict]]): List of bboxes for each keyframe, format:
            [{'object_name': str, 'bboxes': [[x1, y1, x2, y2], ...]}, ...]

    Returns:
        Dict[int, str]: A dictionary mapping frame indices to their saved image paths.
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Open the video file
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    extracted_frames = {}
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    for idx, frame_index in enumerate(keyframe_indices):
        if frame_index < 0 or frame_index >= frame_count:
            print(f"Frame index {frame_index} is out of bounds for video with {frame_count} frames.")
            continue

        # Set the video position to the desired frame index
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB for PIL
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            # Draw bounding boxes for this keyframe
            if idx < len(keyframe_bboxes):
                img_height, img_width = frame.shape[:2]
                
                for obj_data in keyframe_bboxes[idx]:
                    obj_name = obj_data['object_name']
                    bboxes = obj_data['bboxes']
                    
                    # Draw each bbox for this object
                    for bbox in bboxes:
                        if len(bbox) == 4:
                            # Convert from [x_center, y_center, width, height] (normalized 0-1000)
                            # to [x1, y1, x2, y2] (pixel coordinates)
                            x_center, y_center, w, h = bbox
                            
                            x_center_px = x_center
                            y_center_px = y_center
                            w_px = w
                            h_px = h
                            # Denormalize from 0-100 to pixel coordinates
                            # x_center_px = (x_center / 10.0) * img_width
                            # y_center_px = (y_center / 10.0) * img_height
                            # w_px = (w / 10.0) * img_width
                            # h_px = (h / 10.0) * img_height
                            
                            # Convert to corner coordinates
                            x1 = int(x_center_px - w_px / 2)
                            y1 = int(y_center_px - h_px / 2)
                            x2 = int(x_center_px + w_px / 2)
                            y2 = int(y_center_px + h_px / 2)
                            
                            # Clamp to image boundaries
                            x1 = max(0, min(x1, img_width - 1))
                            y1 = max(0, min(y1, img_height - 1))
                            x2 = max(0, min(x2, img_width - 1))
                            y2 = max(0, min(y2, img_height - 1))
                            
                            # Draw rectangle
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                            # Draw label
                            draw.text((x1, max(0, y1 - 10)), obj_name, fill="red")
            
            # Convert back to BGR for OpenCV
            frame_with_bbox = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
            frame_filename = output_path / f"keyframe_{frame_index:05d}.jpg"
            cv.imwrite(str(frame_filename), frame_with_bbox)
            extracted_frames[frame_index] = str(frame_filename)
            print(f"Saved keyframe {frame_index} with {len(keyframe_bboxes[idx]) if idx < len(keyframe_bboxes) else 0} objects")
        else:
            print(f"Failed to read frame at index {frame_index}.")

    cap.release()
    return extracted_frames


def parse_evidence(evidence: str) -> List[Tuple[str, int, List[List[int]]]]:
    """
    Parse evidence string to extract object names, image indices, and bounding boxes.
    
    Args:
        evidence (str): Evidence string containing object annotations
        
    Returns:
        List[Tuple[str, int, List[List[int]]]]: List of tuples containing (object_name, image_index, bboxes)
        
    Example:
        Input: "the <obj_start>diner in image 2<obj_end><box_start>[[500, 519, 1000, 955], [689, 698, 155, 322]]<box_end>"
        Output: [('diner', 2, [[500, 519, 1000, 955], [689, 698, 155, 322]])]
    """
    # Pattern to match: <obj_start>object_name in image N<obj_end><box_start>[[...]]<box_end>
    pattern = r'<obj_start>(.*?)\s+in\s+image\s+(\d+)<obj_end><box_start>(\[\[.*?\]\])<box_end>'
    
    matches = re.findall(pattern, evidence)
    
    results = []
    for match in matches:
        object_name = match[0].strip()
        image_index = int(match[1])
        bbox_str = match[2]
        
        # Parse the bounding boxes
        bboxes = json.loads(bbox_str)
        
        results.append((object_name, image_index, bboxes))
    
    return results


def main():
    json_path = "data/VideoEspresso/bench_hard.json"
    data_video_root = "data/VideoEspresso/video/"
    data_json = json.load(open(json_path, "r"))
    
    # Process only first sample for testing
    for i, sample in enumerate(data_json[:10]):
        video_path = data_video_root + sample["video_path"]
        core_paths = sample["core_frames_paths"]
        question = sample["question"]
        options = sample["options"]
        evidence = sample["evidence"]
    
        keyframe_indices = [int(Path(p).stem.split("/")[-1].split("-")[-1].split("_")[-1]) for p in core_paths]
        
        # Parse objects from evidence
        objects = parse_evidence(evidence)
        
        # Print extracted objects
        print(f"\nVideo: {video_path}")
        print(f"Question: {question}")
        print(f"Core paths: {core_paths}")
        print(f"Keyframe Indices: {keyframe_indices}")
        print(f"\nExtracted Objects:")
        
        # Create bboxes structure for each keyframe
        keyframe_bboxes = [[] for _ in range(len(keyframe_indices))]
        
        for obj_name, img_idx, obj_bboxes in natsorted(objects, key=lambda x: x[1]):
            print(f"  - {obj_name} - {img_idx} - {obj_bboxes}")
            # img_idx starts from 1, convert to 0-based index
            keyframe_idx = img_idx - 1
            if 0 <= keyframe_idx < len(keyframe_indices):
                keyframe_bboxes[keyframe_idx].append({
                    "object_name": obj_name,
                    "bboxes": obj_bboxes
                })
        
        print("-" * 80)
        print(f"Bboxes per keyframe:")
        for idx, kf_bbox in enumerate(keyframe_bboxes):
            print(f"  Keyframe {idx + 1} (frame {keyframe_indices[idx]}): {kf_bbox}")
        print("=" * 80)
        
        # Extract and save keyframes with bboxes
        output_dir = f"data/VideoEspresso/keyframes_with_bbox/{Path(video_path).stem}"
        extracted = extract_keyframes(video_path, output_dir, keyframe_indices, keyframe_bboxes)
        print(f"\nExtracted {len(extracted)} keyframes to {output_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main() 
        
