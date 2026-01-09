import sys
import json
import torch
import random
import argparse
from tqdm import tqdm
from typing import List
from pathlib import Path
from natsort import natsorted

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

sys.path.append(str(Path(__file__).resolve().parent))
from prompt.prompt_question import prompt_user, prompt_user_no_caption

def parse_args():
    parser = argparse.ArgumentParser(description="Extract keyframes from videos.")
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
    parser.add_argument("--f", type=str, required=True, help="Input file json containing video IDs and questions.")
    parser.add_argument("--o-json", type=str, required=True, help="Output file json to save results.")
    parser.add_argument("--imgs", type=str, required=True, help="Directory containing images for the videos.")
    parser.add_argument("--videos", type=str, required=False, help="Directory containing videos.")
    parser.add_argument("--max-model-len", type=int, default=128000, help="Maximum model length.")
    parser.add_argument("--gpu", type=float, default=0.95, help="GPU memory utilization for the model.")
    return parser.parse_args()

def load_model_and_processor(model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", gpu_memory_utilization: float = 0.5, max_model_len: int = 128000, download_dir: str = "./cache_dir"):
    """
    Loads the model and processor with specified GPU memory utilization.
    """
    
    model = LLM(
        model=model_name,
        download_dir=download_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=256,
        stop_token_ids=[],
    )

    processor = AutoProcessor.from_pretrained(model_name, cache_dir="./cache_dir")

    return model, processor, sampling_params

def generate_answer(images: List[str], question: str, model=None, processor=None, sampling_params=None, vllm=True):
    """
    Generates answer based on the provided images and question using the model.
    """
    images_input = [{"type": "image", "image": str(image), "max_pixels": 602112} for image in images]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant tasked with selecting the CORRECT NUMBER ID based on the content of the provided images.",  
        },
        {
            "role": "user",
            "content": [
                *images_input,
                {"type": "text", "text": str(question) + "Instruction: Choose the **CORRECT NUMBER ID** answer for the question below, you must response with the **NUMBER ID** answer, don't include text."},
            ],
        }
    ]
    
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    mm_data = {}
    if image_inputs is not None:
        print("Have image inputs")
        mm_data["image"] = image_inputs
    else:
        print("No image inputs")
    
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    if vllm:
        outputs = model.generate(
            [llm_inputs], sampling_params=sampling_params, use_tqdm=False
        )
        generated_text = outputs[0].outputs[0].text.strip()
        
    else:
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.1, do_sample=True)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    return generated_text


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model
    f_json = str(args.f)
    o_json = str(args.o_json)
    images_dir = Path(args.imgs)
    gpu_memory_utilization = args.gpu
    max_model_len = args.max_model_len
    if args.videos:
        videos_dir = Path(args.videos)

    model, processor, sampling_params = load_model_and_processor(model_name=model_name, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)

    data = json.load(open(f_json, "r"))
    
    score_caption_correct, score_caption_completely_wrong, score_caption_partially_wrong, score_caption_redirected, score_redirect_caption_redirected, score_caption_irrelevance, score_no_caption = 0, 0, 0, 0, 0, 0, 0
    data_new = []
    
    for item in tqdm(data, total=len(data), desc="Processing items"):
        video_id = item["video_id"]
        question = item["question"]
        caption_reasoning = item["reasoning"]
        correct_answer_id = item["correct_answer_id"]
        redirected_answer_id = item["outline_answer_id"]
        
        images = list((images_dir / video_id).glob("*.jpg"))
        print(f"Processing {video_id}")

        caption_correct = item["reasoning"]
        caption_completely_wrong = item["caption_completely_wrong"]
        caption_partially_wrong = item["caption_partially_wrong"]
        caption_redirected = item["caption_redirected"]
        caption_irrelevant = item["caption_irrelevance"]
        no_caption = ""
        
        dict_captions = {
            "correct": caption_correct,
            "completely_wrong": caption_completely_wrong,
            "partially_wrong": caption_partially_wrong,
            "redirected": caption_redirected,
            "irrelevant": caption_irrelevant,
            "no_caption": no_caption
        }

        for caption_key, caption in dict_captions.items():
            if caption != "":
                prompt_question_caption = prompt_user(
                    question=question,
                    answer_0=item["answer_choice_0"],
                    answer_1=item["answer_choice_1"],
                    answer_2=item["answer_choice_2"],
                    answer_3=item["answer_choice_3"],
                    answer_4=item["answer_choice_4"],
                    caption=caption
                )
            else:
                prompt_question_caption = prompt_user_no_caption(
                    question=question,
                    answer_0=item["answer_choice_0"],
                    answer_1=item["answer_choice_1"],
                    answer_2=item["answer_choice_2"],
                    answer_3=item["answer_choice_3"],
                    answer_4=item["answer_choice_4"],
                )
            
            generated_answer = generate_answer(
                images=[str(img) for img in natsorted(images)],
                question=prompt_question_caption,
                model=model,
                processor=processor,
                sampling_params=sampling_params,
                vllm=True
            )
            
            while generated_answer.strip() not in ["0", "1", "2", "3", "4"]:
                print(f"Regenerating answer for video {video_id} due to invalid response: {generated_answer}")
                generated_answer = generate_answer(
                    images=[str(img) for img in natsorted(images)],
                    question=prompt_question_caption,
                    model=model,
                    processor=processor,
                    sampling_params=sampling_params,
                    vllm=True
                )
            
            print(f"Generated Answer: {generated_answer}, Correct Answer: {correct_answer_id}, Caption Type: {caption_key}")
            
            if caption_key == "correct" and generated_answer.strip() == str(correct_answer_id):
                score_caption_correct += 1
            elif caption_key == "completely_wrong" and generated_answer.strip() == str(correct_answer_id):
                score_caption_completely_wrong += 1
            elif caption_key == "partially_wrong" and generated_answer.strip() == str(correct_answer_id):
                score_caption_partially_wrong += 1
            elif caption_key == "redirected" and generated_answer.strip() == str(redirected_answer_id):
                score_caption_redirected += 1
            elif caption_key == "irrelevant" and generated_answer.strip() == str(correct_answer_id):
                score_caption_irrelevance += 1
            elif caption_key == "no_caption" and generated_answer.strip() == str(correct_answer_id):
                score_no_caption += 1
        
            if caption_key == "redirected" and generated_answer.strip() == str(redirected_answer_id):
                score_redirect_caption_redirected += 1
                
            item[f"generated_answer_{caption_key}"] = generated_answer.strip()
            
        data_new.append(item)

    print(f"Score with Caption Correct: {score_caption_correct}/{len(data)}")
    print(f"Score with Caption Completely Wrong: {score_caption_completely_wrong}/{len(data)}")
    print(f"Score with Caption Partially Wrong: {score_caption_partially_wrong}/{len(data)}")
    print(f"Score with Caption Redirected: {score_caption_redirected}/{len(data)}")
    print(f"Score with Redirected Caption Redirected: {score_redirect_caption_redirected}/{len(data)}")
    print(f"Score with Caption Irrelevance: {score_caption_irrelevance}/{len(data)}")
    print(f"Score with No Caption: {score_no_caption}/{len(data)}")
        
    json.dump(data_new, open(o_json, "w"), indent=4)


# python src/baseline.py --model OpenGVLab/InternVL3_5-2B --f data/combined_data.json --o-json final_score_internvl3_5_2b.json --imgs data/keyframes_new --sub data/subtitles/ --gpu 0.95
