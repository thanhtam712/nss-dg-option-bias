import sys
import json
import random
import argparse
from tqdm import tqdm
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
# from src.prompt.prompt_sft import prompt_user, assistant_match, assistant_corruption, assistant_irrelevance
from src.prompt.prompt_question import prompt_user_no_caption, prompt_user


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SFT data for LlamaFactory")
    parser.add_argument("--file-json", type=str, default="data/data_samples/Causal_Vid/baseline_caption_outline_openai_full_combined.json", help="Path to the input JSON file")
    parser.add_argument("--root-images", type=str, default="data/keyframes_full", help="Root directory for images")
    parser.add_argument("--file-save", type=str, default="data_llama_factory/causal_vid/causal_vid_sft.json", help="Path to save the prepared SFT data")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to process from the input JSON file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    file_json = args.file_json
    root_images = args.root_images
    file_save = args.file_save
    Path(file_save).parent.mkdir(parents=True, exist_ok=True)
    
    data_json = json.load(open(file_json, "r"))[:-args.num_samples]
    random.shuffle(data_json)
    data_json = data_json[:args.num_samples]
    
    data_train = []
    for item in tqdm(data_json, desc="Preparing SFT data", total=args.num_samples):
        question = item["question"]
        if "video_id" in item:
            video_id = item["video_id"]
            list_images = [str(p) for p in (Path(root_images) / video_id).glob("*.jpg")]
        else:
            list_images = [str(Path(root_images) / item["image_id"])]
        
        answer_correct_id = item["correct_answer_id"]
        
        number_choices = len([item[f"answer_choice_{i}"] for i in range(5) if f"answer_choice_{i}" in item])
        option_answers = "\n".join([f"{i}: {item[f'answer_choice_{i}']}" for i in range(number_choices)])

        caption_correct = item["caption_correct"]
        caption_completely_wrong = item["caption_completely_wrong"]
        caption_partially_wrong = item["caption_partially_wrong"]
        caption_redirected = item["caption_redirected"]
        caption_irrelevance = item["caption_irrelevance"]
        
        structure_correct = {'messages': [
            {"role": "user", "content": "<image> " * len(list_images) + prompt_user.format(caption=caption_correct, options=option_answers, question=question)},
            {"role": "assistant", "content": str(answer_correct_id)}
        ], 'images': list_images}
        
        structure_completely_wrong = {'messages': [
            {"role": "user", "content": "<image> " * len(list_images) + prompt_user.format(caption=caption_completely_wrong, options=option_answers, question=question)},
            {"role": "assistant", "content": str(answer_correct_id)}
        ], 'images': list_images}

        structure_partially_wrong = {'messages': [
            {"role": "user", "content": "<image> " * len(list_images) + prompt_user.format(caption=caption_partially_wrong, options=option_answers, question=question)},
            {"role": "assistant", "content": str(answer_correct_id)}
        ], 'images': list_images}
        
        structure_redirected = {'messages': [
            {"role": "user", "content": "<image> " * len(list_images) + prompt_user.format(caption=caption_redirected, options=option_answers, question=question)},
            {"role": "assistant", "content": str(answer_correct_id)}
        ], 'images': list_images}
        
        structure_irrelevance = {'messages': [
            {"role": "user", "content": "<image> " * len(list_images) + prompt_user.format(caption=caption_irrelevance, options=option_answers, question=question)},
            {"role": "assistant", "content": str(answer_correct_id)}
        ], 'images': list_images}
        
        structure_no_caption = {'messages': [
            {"role": "user", "content": "<image> " * len(list_images) + prompt_user_no_caption.format(options=option_answers, question=question)},
            {"role": "assistant", "content": str(answer_correct_id)}
        ], 'images': list_images}
        
        data_train.append(structure_correct)
        data_train.append(structure_completely_wrong)
        data_train.append(structure_partially_wrong)
        data_train.append(structure_redirected)
        data_train.append(structure_irrelevance)
        data_train.append(structure_no_caption)
        
    json.dump(data_train, open(file_save, "w"), indent=4)

if __name__ == "__main__":
    main()

# python src/prepare_data_sft/prepare_sft_llamafactory.py --file-json data/data_samples/AOK_VQA/baseline_gemini_combined.json --root-images data/aokvqa/processed/images/ --file-save data_llama_factory/aok_vqa/aok_vqa_sft.json

# python src/prepare_data_sft/prepare_sft_llamafactory.py --file-json data/data_samples/science_QA/baseline_gemini_combined.json --root-images data/science_QA/converted_data/images --file-save data_llama_factory/science_qa/science_qa_sft.json
