import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Process VQA dataset")
    parser.add_argument("--f-neptune", type=Path, required=True, help="Path to the input JSON file dataset Neptune")
    parser.add_argument("--f-casual", type=Path, required=True, help="Path to the input JSON folder dataset Casual-VidQA")
    parser.add_argument("--output_file", type=Path, required=True, help="Path to the output JSON file")
    return parser.parse_args()

def main():
    args = parse_args()
    folder_videos = Path("data/videos_filtered")
    data_combined = []

    suffix_videos = list(file.stem for file in folder_videos.glob("*.mp4"))
    
    with open(args.f_neptune, "r") as f:
        neptune_data = json.load(f)

    for item in neptune_data:
        if item["video_id"] in suffix_videos:
            data_combined.append({
                "video_id": item["video_id"],
                "question": item["question"],
                "answer_choice_0": item["answer_choice_0"],
                "answer_choice_1": item["answer_choice_1"],
                "answer_choice_2": item["answer_choice_2"],
                "answer_choice_3": item["answer_choice_3"],
                "answer_choice_4": item["answer_choice_4"],
                "correct_answer_id": item["answer_id"],
                "reasoning": item["reasoning"]
            })

    for video_path in args.f_casual.glob("**/*.mp4"):
        video_path = Path(video_path)
        
        if video_path.stem not in suffix_videos:
            continue

        qa_data = video_path.with_suffix(".json")
        answer_data = video_path.parent / Path("answer.json")
        answer_f = json.load(open(answer_data, "r"))
        id_reason_predictive = answer_f["predictive"]["reason"]
        id_reason_counterfactual = answer_f["counterfactual"]["reason"]

        with open(qa_data, "r") as f:
            casual_data = json.load(f)
            data_combined.append({
                "video_id": qa_data.stem,
                "question": casual_data["predictive"]["question"],
                "answer_choice_0": casual_data["predictive"]["answer"][0],
                "answer_choice_1": casual_data["predictive"]["answer"][1],
                "answer_choice_2": casual_data["predictive"]["answer"][2],
                "answer_choice_3": casual_data["predictive"]["answer"][3],
                "answer_choice_4": casual_data["predictive"]["answer"][4],
                "correct_answer_id": answer_f["predictive"]["answer"],
                "reasoning": casual_data["predictive"]["reason"][id_reason_predictive]
            })
            
            data_combined.append({
                "video_id": qa_data.stem,
                "question": casual_data["counterfactual"]["question"],
                "answer_choice_0": casual_data["counterfactual"]["answer"][0],
                "answer_choice_1": casual_data["counterfactual"]["answer"][1],
                "answer_choice_2": casual_data["counterfactual"]["answer"][2],
                "answer_choice_3": casual_data["counterfactual"]["answer"][3],
                "answer_choice_4": casual_data["counterfactual"]["answer"][4],
                "correct_answer_id": answer_f["counterfactual"]["answer"],
                "reasoning": casual_data["counterfactual"]["reason"][id_reason_counterfactual]
            })

    with open(args.output_file, "w") as f:
        json.dump(data_combined, f, indent=4)


if __name__ == "__main__":
    main()
