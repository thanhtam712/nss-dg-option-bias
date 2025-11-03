import sys
import json
import torch
import random
import jsonlines
import argparse
from pathlib import Path
from typing import List
from tqdm import tqdm

import re
import requests
from PIL import Image
from io import BytesIO

# from vllm import LLM, SamplingParams
# from transformers import AutoProcessor
# from qwen_vl_utils import process_vision_info

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.prompt_question import prompt_case1, prompt_case2, prompt_case3, prompt_case4, prompt_case5

from VTI.llava.eval.run_llava import eval_model
from VTI.llava.utils import disable_torch_init
from VTI.llava.model.builder import load_pretrained_model
from VTI.llava.conversation import conv_templates, SeparatorStyle
from VTI.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, get_model_name_from_path

def find_cache_dir():
    current = Path.cwd()
    
    for parent in [current] + list(current.parents):
        possible = parent / "Grad_Thesis" / "cache_dir"
        if possible.exists():
            return possible.resolve()

    raise FileNotFoundError("Không tìm thấy thư mục Grad_Thesis/cache_dir trong cây cha.")
cache_dir_path = find_cache_dir()

def parse_args():
    parser = argparse.ArgumentParser(description="Extract keyframes from videos.")
    parser.add_argument("--model", type=str, required=True, default="liuhaotian/llava-v1.5-7b", help="Model name or path.")
    parser.add_argument("--f", type=str, required=True, help="Input file json containing video IDs and questions.")
    parser.add_argument("--o-json", type=str, required=True, help="Output file json to save results.")
    parser.add_argument("--imgs", type=str, required=True, help="Directory containing images for the videos.")
    parser.add_argument("--videos", type=str, required=False, help="Directory containing videos.")
    parser.add_argument("--sub", type=str, required=False, help="Directory containing subtitles for the videos.")
    parser.add_argument("--max-model-len", type=int, default=256000, help="Maximum model length.")
    parser.add_argument("--gpu", type=float, default=0.95, help="GPU memory utilization for the model.")
    return parser.parse_args()

def load_model_and_processor(model_path: str, cache_dir: str = cache_dir_path):
    """
    Loads the model and processor with specified GPU memory utilization.
    """
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        cache_dir=cache_dir
    )

    return model, tokenizer, image_processor, context_len

def prompt_questions(question: str, answer_0: str, answer_1: str, answer_2: str, answer_3: str, answer_4: str, caption_generated: str = "", caption_reasoning: str = "", transcript_video: str = "", outline_information: str = "") -> List[str]:
    """
    Generates prompt questions based on the provided answers and context.
    """

    question_case1 = prompt_case1.format(
        question=question,
        answer_0=answer_0,
        answer_1=answer_1,
        answer_2=answer_2,
        answer_3=answer_3,
        answer_4=answer_4
    )
    
    question_case2 = prompt_case2.format(
        question=question,
        answer_0=answer_0,
        answer_1=answer_1,
        answer_2=answer_2,
        answer_3=answer_3,
        answer_4=answer_4,
        caption_generated=caption_generated
    )
    
    question_case3 = prompt_case3.format(
        question=question,
        answer_0=answer_0,
        answer_1=answer_1,
        answer_2=answer_2,
        answer_3=answer_3,
        answer_4=answer_4,
        caption_reasoning=caption_reasoning
    )
    
    question_case4 = prompt_case4.format(
        question=question,
        answer_0=answer_0,
        answer_1=answer_1,
        answer_2=answer_2,
        answer_3=answer_3,
        answer_4=answer_4,
        transcript=transcript_video
    )
    
    question_case5 = prompt_case5.format(
        question=question,
        answer_0=answer_0,
        answer_1=answer_1,
        answer_2=answer_2,
        answer_3=answer_3,
        answer_4=answer_4,
        information=outline_information
    )

    return [
        question_case1,
        question_case2,
        question_case3,
        question_case4,
        question_case5
    ]

def generate_caption(images: List[str], video_path: str, question: str, model=None, tokenizer=None, image_processor=None, context_len=None, model_path: str = ""):
    """
    Generates captions based on the provided images and question using the model.
    """
    
    prompt = "Generate caption based on the provided images and question below." + "\n" + question

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": images,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    outputs = eval_model(args, tokenizer, model, image_processor, context_len)
    print("Generated caption:", outputs)

    return outputs.strip()

def generate_outline_reasoning(images: List[str], video_path: str, question: str, reasoning: str, answer_outline: str, model=None, tokenizer=None, image_processor=None, context_len=None, model_path: str = ""):
    """
    Generates outline reasoning based on the provided images and question using the model.
    """
    
    prompt = "Generate an outline reasoning sentence based on the given question, groundtruth reasoning, and answer outline. Ensure the outline reasoning aligns with the groundtruth reasoning and supports the final answer structure, don't include any irrelevant information." + "\n" + question + "\n" + reasoning + "\n" + answer_outline

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": images,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    outputs = eval_model(args, tokenizer, model, image_processor, context_len)
    print("Generated caption:", outputs)

    return outputs.strip()

def generate_answer(images: List[str], video_path: str, question: str, model=None, tokenizer=None, image_processor=None, context_len=None, model_path: str = ""):
    """
    Generates answer based on the provided images and question using the model.
    """
    prompt = "Choose the **correct ID** answer for the question below, you must response with the **ID** answer, don't include text." + "\n" + str(question)

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": images,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    outputs = eval_model(args, tokenizer, model, image_processor, context_len)
    print("Generated caption:", outputs)

    return outputs.strip()

if __name__ == "__main__":
    args = parse_args()
    model_name = args.model
    f_json = str(args.f)
    o_json = str(args.o_json)
    images_dir = Path(args.imgs)
    max_model_len = args.max_model_len
    if args.videos:
        videos_dir = Path(args.videos)

    model, tokenizer, image_processor, context_len = load_model_and_processor(model_path=model_name)

    correct_answer = 0
    data = json.load(open(f_json, "r"))
    data_new = []

    dict_vid_id = {video_name.stem: video_name for video_name in Path("data/dataset").iterdir()}

    data_caption_vti = {}
    with jsonlines.open("VTI/results/caption_outline_openai_run_ori2_reasoning.jsonl", "r") as reader:
        for item in reader:
            if item["id"] not in data_caption_vti:
                data_caption_vti[item["id"]] = []
            data_caption_vti[item["id"]].append({
                "question": item["question"],
                "caption": item["caption"],
                "prompt_question": item["prompt_question"],
            })

    score_case_1, score_case_2, score_case_3, score_case_4, score_case_5 = 0, 0, 0, 0, 0
    total_answers = 0
    for item in tqdm(data, total=len(data), desc="Processing items"):
        video_id = item["video_id"]
        question = item["question"]
        caption_reasoning = item["reasoning"]
        answer_choices = item["correct_answer_id"]
        
        if video_id not in dict_vid_id or answer_choices == -1:
            continue
        
        images = list((images_dir / video_id).glob("*.jpg"))
        images = list(str(img) for img in images)
        if args.videos:
            video_path = str((videos_dir / video_id).glob("*.mp4"))
        else:
            video_path = None

        print(f"Processing {video_id}")
        total_answers += 1

        # Outline answer id
        # outline_answer_id = -1
        # while outline_answer_id == -1 or outline_answer_id == int(answer_choices):
        #     outline_answer_id = random.randint(0, 4)
        # item["outline_answer_id"] = outline_answer_id

        # Generate outline information
        # outline_information = generate_outline_reasoning(images, video_path, question, caption_reasoning, item["answer_choice_" + str(outline_answer_id)], model, tokenizer, image_processor, context_len, model_name)
        # outline_information = outline_information.replace("Outline Reasoning", "")

        # Generate caption
        caption_generated = generate_caption(images, video_path, question, model, tokenizer, image_processor, context_len, model_name)
        
        # Get outline information and answer id from const file
        # outline_information = item["outline_caption"]
        # get caption if question in data_caption_vti the same with question now
        for i in data_caption_vti[video_id]:
            if i["question"] == question:
                outline_information = i["caption"]
                item["outline_caption"] = outline_information
                break
        outline_answer_id = item["outline_answer_id"]
        # outline_information = item["outline_caption"]

        list_questions_prompt = prompt_questions(
            question=item["question"],
            answer_0=item["answer_choice_0"],
            answer_1=item["answer_choice_1"],
            answer_2=item["answer_choice_2"],
            answer_3=item["answer_choice_3"],
            answer_4=item["answer_choice_4"],
            caption_generated=caption_generated,
            caption_reasoning=caption_reasoning,
            transcript_video="",
            outline_information=outline_information
        )

        for idx, question_case in enumerate(list_questions_prompt):
            answer_model = generate_answer(images, video_path, question_case, model, tokenizer, image_processor, context_len, model_name)
            
            time = 0
            while answer_model.strip() not in ["0", "1", "2", "3", "4"] and time < 10:
                answer_model = generate_answer(images, video_path, question_case, model, tokenizer, image_processor, context_len, model_name)
                time += 1
            if time >= 10:
                continue
            
            if idx != 4 and answer_model.strip() in ["0", "1", "2", "3", "4"] and int(answer_model) == int(answer_choices):
                print("Correct answer!")
                if idx == 0:
                    score_case_1 += 1
                elif idx == 1:
                    score_case_2 += 1
                elif idx == 2:
                    score_case_3 += 1
                else:
                    score_case_4 += 1
            elif idx == 4 and answer_model.strip() in ["0", "1", "2", "3", "4"] and int(answer_model) == int(outline_answer_id):
                print("Correct outline answer!")
                score_case_5 += 1
            else:
                print("Incorrect answer.")

            item["answer_model_" + str(idx)] = answer_model

            print(f"Case {idx}")
            print("-"*20)
            print(question_case)
            print(f"Answer model: {answer_model} \nAnswer correct: {answer_choices}\n Answer outline: {outline_answer_id}")
            print("="*50)
        
        data_new.append(item)

    json.dump(data_new, open(o_json, "w"), indent=4)

    print(f"Score case 1: {score_case_1} / {total_answers}")
    print(f"Score case 2: {score_case_2} / {total_answers}")
    print(f"Score case 3: {score_case_3} / {total_answers}")
    print(f"Score case 4: {score_case_4} / {total_answers}")
    print(f"Score case 5: {score_case_5} / {total_answers}")


# python src/baseline_llava.py --model liuhaotian/llava-v1.5-7b --f save_caption_outline_Qwen.json --o-json baseline_llava.json --imgs data/keyframes_new --gpu 0.95
