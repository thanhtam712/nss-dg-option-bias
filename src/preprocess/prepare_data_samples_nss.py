import json
import random
import argparse
import jsonlines
from tqdm import tqdm
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for nss task")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/data_samples/baseline_gemini_v2.json",
        help="Path to the input data file (JSON format).",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/keyframes_full",
        help="Directory containing the images.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="NSS_DG/experiments/data/gemini_full_dataset_answer_str.json",
        help="Path to the output data file (JSONL format).",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    path_data = Path(args.data_path)
    data = json.load(open(path_data, "r"))
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_nss_new = []

    prompt_question = """
    Question:
    {question}

    Answer options:
    {options}

    Here is some additional information which are text descriptions based on the image to assist you in answering the later question. Note, the information could be irrelevant, missing some information or inaccurate, please use it with caution: {information}
    """

    for sample in tqdm(data[:-1000], desc="Processing samples", total=len(data[:-1000])):
        if "video_id" in sample:
            id = sample["video_id"]
            images = list((Path(args.images_dir) / id).iterdir())
        else:
            id = Path(sample["image_id"]).stem
            images = [Path(args.images_dir) / sample["image_id"]]
        
        number_choices = len([key for key in sample.keys() if key.startswith("answer_choice_")])
        options = "\n".join([f"{i}. {sample[f'answer_choice_{i}']}" for i in range(number_choices)])
        
        question = prompt_question.format(
            question=sample["question"],
            options=options,
            information=sample["caption_redirected"])
        
        answers_choice = [sample[f"answer_choice_{i}"] for i in range(number_choices)]
        correct_id = sample["correct_answer_id"]
        outline_id = sample["outline_answer_id"]
        
        answer_correct = sample[f"answer_choice_{correct_id}"]
        answer_outline = sample[f"answer_choice_{outline_id}"]
        
        data_nss_new.append({
            "id": id,
            "images": [str(image_path.name) for image_path in images],
            "value": str(correct_id) + ": " + answer_correct,
            "h_value": str(outline_id) + ": " + answer_outline,
            "question": question,
            "answers_choice": answers_choice,
            "correct_id": correct_id,
            "outline_id": outline_id
        })

    json.dump(data_nss_new, open(output_path, "w"), indent=4)

if __name__ == "__main__":
    main()
