import sys
import json
import torch
import random
import argparse
from natsort import natsorted
from pathlib import Path
from typing import List
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

sys.path.append(str(Path(__file__).resolve().parent))
from prompt_question import prompt_case1, prompt_case2, prompt_case3, prompt_case4, prompt_case5

def find_cache_dir():
    current = Path.cwd()
    
    for parent in [current] + list(current.parents):
        possible = parent / "Grad_Thesis" / "cache_dir"
        if possible.exists():
            return possible.resolve()

    raise FileNotFoundError("Không tìm thấy thư mục Grad_Thesis/cache_dir trong cây cha.")
# cache_dir_path = str(find_cache_dir())
cache_dir_path = "./cache_dir"

def parse_args():
    parser = argparse.ArgumentParser(description="Extract keyframes from videos.")
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
    parser.add_argument("--f", type=str, required=True, help="Input file json containing video IDs and questions.")
    parser.add_argument("--o-json", type=str, required=True, help="Output file json to save results.")
    parser.add_argument("--imgs", type=str, required=True, help="Directory containing images for the videos.")
    parser.add_argument("--videos", type=str, required=False, help="Directory containing videos.")
    parser.add_argument("--sub", type=str, required=True, help="Directory containing subtitles for the videos.")
    parser.add_argument("--max-model-len", type=int, default=256000, help="Maximum model length.")
    parser.add_argument("--gpu", type=float, default=0.95, help="GPU memory utilization for the model.")
    return parser.parse_args()

def load_model_and_processor(model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", gpu_memory_utilization: float = 0.5, max_model_len: int = 32768, download_dir: str = cache_dir_path):
    """
    Loads the model and processor with specified GPU memory utilization.
    """
    
    model = LLM(
        model=model_name,
        download_dir=download_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        # load_in_4bit=True,
        max_model_len=max_model_len,
        # flash_attention=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=256,
        stop_token_ids=[],
    )

    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir_path)

    return model, processor, sampling_params

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

def generate_caption(images: List[str], video_path: str, question: str, model=None, processor=None, sampling_params=None):
    """
    Generates captions based on the provided images and question using the model.
    """
    
    images_input = [{"type": "image", "image": str(image), "max_pixels": 360 * 240} for image in images]
    # video_input = [{"type": "video", "video": str(video_path), "max_pixels": 360 * 420, "video_fps": 30}]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that generate caption based on the provided images and question.",  
        },
        {
            "role": "user",
            "content": [
                *images_input,
                # *video_input,
                {"type": "text", "text": "Generate caption based on the provided images and question below." + "\n" + question},
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
        mm_data["image"] = image_inputs
    
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    outputs = model.generate(
        [llm_inputs], sampling_params=sampling_params, use_tqdm=False
    )
    
    print(outputs)
    
    generated_text = outputs[0].outputs[0].text

    # print(generated_text)
    # input("Press Enter to continue...")

    return generated_text.strip()

def generate_outline_reasoning(images: List[str], video_path: str, question: str, reasoning: str, answer_outline: str, model=None, processor=None, sampling_params=None):
    """
    Generates outline reasoning based on the provided images and question using the model.
    """
    
    images_input = [{"type": "image", "image": str(image), "max_pixels": 360 * 240} for image in images]
    # video_input = [{"type": "video", "video": str(video_path), "max_pixels": 360 * 420,"video_fps": 30}]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that generate outline reasoning based on the provided images, question and answers.",  
        },
        {
            "role": "user",
            "content": [
                *images_input,
                # *video_input,
                {"type": "text", "text": "Generate an outline reasoning sentence based on the given question, groundtruth reasoning, and answer outline. Ensure the outline reasoning aligns with the groundtruth reasoning and supports the final answer structure, don't include any irrelevant information." + "\n" + question + "\n" + reasoning + "\n" + answer_outline},
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
        mm_data["image"] = image_inputs
    
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    outputs = model.generate(
        [llm_inputs], sampling_params=sampling_params, use_tqdm=False
    )
    generated_text = outputs[0].outputs[0].text

    return generated_text.strip()


def generate_answer(images: List[str], video_path: str, question: str, model=None, processor=None, sampling_params=None, vllm=True):
    """
    Generates answer based on the provided images and question using the model.
    """
    
    # messages = [
    #     {
    #       "role": "system",
    #         "content": prompt_system,  
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             *images_input,
    #             {"type": "text", "text": question},
    #         ],
    #     }
    # ]
    
    images_input = [{"type": "image", "image": str(image), "max_pixels": 360 * 240} for image in images]
    # video_input = [{"type": "video", "video": str(video_path), "max_pixels": 360 * 420, "video_fps": 30}]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant tasked with selecting the correct answer ID based on the content of the provided images.",  
        },
        {
            "role": "user",
            "content": [
                # {
                #     "type": "video",
                #     "video": "test.mp4",
                #     "max_pixels": 360 * 420,
                #     "fps": 1.0,
                # },
                *images_input,
                # *video_input,
                {"type": "text", "text": "Choose the **correct ID** answer for the question below, you must response with the **ID** answer, don't include text." + "\n" + str(question)},
            ],
        }
    ]
    
    print(messages)

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
    # print(generated_text)
    # print("======================")

    return generated_text


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model
    f_json = str(args.f)
    o_json = str(args.o_json)
    images_dir = Path(args.imgs)
    sub_dir = Path(args.sub)
    gpu_memory_utilization = args.gpu
    max_model_len = args.max_model_len
    if args.videos:
        videos_dir = Path(args.videos)

    model, processor, sampling_params = load_model_and_processor(model_name=model_name, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)

    correct_answer = 0
    data = json.load(open(f_json, "r"))
    data_new = []

    dict_vid_id = {video_name.stem: video_name for video_name in Path("data/dataset").iterdir()}

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
        if args.videos:
            video_path = str((videos_dir / video_id).glob("*.mp4"))
        else:
            video_path = None
        # transcript_video = open(sub_dir / f"{video_id}.en.srt", "r").read()

        print(f"Processing {video_id}")
        total_answers += 1
        # TODO: Generate outline information and caption generated
        
        # outline_answer_id = -1
        # while outline_answer_id == -1 or outline_answer_id == int(answer_choices):
        #     outline_answer_id = random.randint(0, 4)
        # item["outline_answer_id"] = outline_answer_id
        # outline_information = generate_outline_reasoning(images, video_path, question, caption_reasoning, item["answer_choice_" + str(outline_answer_id)], model, processor, sampling_params)
        # outline_information = outline_information.replace("Outline Reasoning", "")
        outline_information = item["outline_caption"]
        outline_answer_id = item["outline_answer_id"]
        
        
        caption_generated = generate_caption(images, video_path, question, model, processor, sampling_params)

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
            answer_model = generate_answer(images, video_path, question_case, model, processor, sampling_params, True)
            
            time = 0
            while answer_model.strip() not in ["0", "1", "2", "3", "4"] and time < 10:
                answer_model = generate_answer(images, video_path, question_case, model, processor, sampling_params, True)
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
            # input("="*50)
        
        data_new.append(item)

    json.dump(data_new, open(o_json, "w"), indent=4)

    print(f"Score case 1: {score_case_1} / {total_answers}")
    print(f"Score case 2: {score_case_2} / {total_answers}")
    print(f"Score case 3: {score_case_3} / {total_answers}")
    print(f"Score case 4: {score_case_4} / {total_answers}")
    print(f"Score case 5: {score_case_5} / {total_answers}")


# python src/baseline.py --model OpenGVLab/InternVL3_5-2B --f data/combined_data.json --o-json final_score_internvl3_5_2b.json --imgs data/keyframes_new --sub data/subtitles/ --gpu 0.95
