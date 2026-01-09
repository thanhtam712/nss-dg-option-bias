import sys
import json
import random
import argparse
from urllib.parse import unquote
from natsort import natsorted
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.prompt.prompt_question import prompt_user, prompt_user_no_caption

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name or path",
    )
    parser.add_argument(
        "--imgs-root",
        type=str,
        required=True,
        help="Root directory containing images.",
    )
    parser.add_argument(
        "--json-file",
        type=str,
        required=True,
        help="Path to the input JSON file containing images.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save the output results JSON file.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=128000,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["vllm", "transformers", "lmdeploy"],
        default="vllm",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.001,
        help="Top-p sampling parameter."
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
        help="Repetition penalty."
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search."
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Whether to use cache."
    )
    parser.add_argument(
        "--output_attentions",
        action="store_true",
        help="Whether to output attentions."
    )
    parser.add_argument(
        "--return_dict_in_generate",
        action="store_true",
        help="Return dict in generate."
    )
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    if args.engine == "vllm":
        from src.models.vllm_generator import VLLM

        return VLLM(
            model_name=args.model_name,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            temperature=args.temperature,
            tensor_parallel_size=1,
        )
    elif args.engine == "lmdeploy":
        from src.models.lmdeploy_generator import LmDeployGenerator

        return LmDeployGenerator(
            model_name=args.model_name,
            session_len=args.max_model_len,
            cache_max_entry_count=args.gpu_memory_utilization,
            dtype=args.dtype,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    elif args.engine == "transformers":
        from src.models.transformers_generator import TransformersGenerator

        return TransformersGenerator(
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    else:
        raise ValueError(f"Unsupported engine: {args.engine}")
    
def check_find(answer: str) -> bool:
    answer = answer.strip().lower()
    if answer.find("0") != -1 or answer.find("1") != -1 or answer.find("2") != -1 or answer.find("3") != -1 or answer.find("4") != -1:
        return False
    return True

def main(args: argparse.Namespace):
    print(args)
    print("Loading data ...")
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    data = json.load(open(args.json_file, "r"))
    imgs_root = Path(args.imgs_root)
    data_wiki = open("data/wiki/test.txt", "r").readlines()
    
    results_correct, results_correct_redirected, result_redirect_redirected, results_completely_wrong, results_partially_wrong, results_irrelevant, results_no_caption = 0, 0 , 0, 0, 0, 0, 0
    results = []

    print("Loading model ...")
    model = load_model(args)

    for sample in tqdm(data[-1000:], desc="Evaluating samples", total=1000):
        if "video_id" in sample:
            video_id = sample["video_id"]
            imgs_url = (imgs_root / video_id).iterdir()
            imgs_url = [str(p) for p in imgs_url if p.suffix in [".jpg", ".png", ".jpeg"]]
            imgs_url = natsorted(imgs_url)
        else:
            imgs_url = [str(imgs_root / unquote(sample["image_id"]))]
            
        number_choices = len([sample[f"answer_choice_{i}"] for i in range(5) if f"answer_choice_{i}" in sample])
        option_answers = "\n".join([f"{i}: {sample[f'answer_choice_{i}']}" for i in range(number_choices)])
        
        question = sample["question"]
        caption_correct = sample["caption_correct"]
        caption_redirected = sample["caption_redirected"]
        caption_completely_wrong = sample["caption_completely_wrong"]
        caption_partially_wrong = sample["caption_partially_wrong"]
        caption_irrelevant = random.choice(data_wiki).strip()
        
        correct_answer_id = sample["correct_answer_id"]
        redirected_answer_id = sample["outline_answer_id"]

        # Caption correct
        time_loop = 0
        answer_caption_correct = model.chat(
            user_prompt="<image>" * len(imgs_url) + prompt_user.format(caption=caption_correct, options=option_answers, question=question),
            images=imgs_url,
        )
        answer_caption_correct = answer_caption_correct.replace("\n", " ").strip()
        while not check_find(answer=answer_caption_correct) and time_loop < 5:
            answer_caption_correct = model.chat(
                user_prompt="<image>" * len(imgs_url) + prompt_user.format(caption=caption_correct, options=option_answers, question=question),
                images=imgs_url,
            )
            answer_caption_correct = answer_caption_correct.replace("\n", " ").strip()
            time_loop += 1
        if answer_caption_correct.find(str(correct_answer_id)) != -1:
            results_correct += 1
            
        # Caption redirected
        time_loop = 0
        answer_caption_redirected = model.chat(
            user_prompt="<image>" * len(imgs_url) + prompt_user.format(caption=caption_redirected, options=option_answers, question=question),
            images=imgs_url,
        )
        answer_caption_redirected = answer_caption_redirected.replace("\n", " ").strip()
        while not check_find(answer=answer_caption_redirected) and time_loop < 5:
            answer_caption_redirected = model.chat(
                user_prompt="<image>" * len(imgs_url) + prompt_user.format(caption=caption_redirected, options=option_answers, question=question),
                images=imgs_url,
            )
            answer_caption_redirected = answer_caption_redirected.replace("\n", " ").strip()
            time_loop += 1
        
        
        if answer_caption_redirected.find(str(correct_answer_id)) != -1:
            results_correct_redirected += 1
        if answer_caption_redirected.find(str(redirected_answer_id)) != -1:
            result_redirect_redirected += 1
        
        # Caption completely wrong
        time_loop = 0
        answer_caption_completely_wrong = model.chat(
            user_prompt="<image>" * len(imgs_url) + prompt_user.format(caption=caption_completely_wrong, options=option_answers, question=question),
            images=imgs_url,
        )
        answer_caption_completely_wrong = answer_caption_completely_wrong.replace("\n", " ").strip()
        while not check_find(answer=answer_caption_completely_wrong) and time_loop < 5:
            answer_caption_completely_wrong = model.chat(
                user_prompt="<image>" * len(imgs_url) + prompt_user.format(caption=caption_completely_wrong, options=option_answers, question=question),
                images=imgs_url,
            )
            answer_caption_completely_wrong = answer_caption_completely_wrong.replace("\n", " ").strip()
            time_loop += 1
            
        if answer_caption_completely_wrong.find(str(correct_answer_id)) != -1:
            results_completely_wrong += 1
        
        # Caption partially wrong
        time_loop = 0
        answer_caption_partially_wrong = model.chat(    
            user_prompt="<image>" * len(imgs_url) + prompt_user.format(caption=caption_partially_wrong, options=option_answers, question=question),
            images=imgs_url,
        )
        answer_caption_partially_wrong = answer_caption_partially_wrong.replace("\n", " ").strip()
        while not check_find(answer=answer_caption_partially_wrong) and time_loop < 5:
            answer_caption_partially_wrong = model.chat(    
                user_prompt="<image>" * len(imgs_url) + prompt_user.format(caption=caption_partially_wrong, options=option_answers, question=question),
                images=imgs_url,
            )
            answer_caption_partially_wrong = answer_caption_partially_wrong.replace("\n", " ").strip()
            time_loop += 1
            
        if answer_caption_partially_wrong.find(str(correct_answer_id)) != -1:
            results_partially_wrong += 1
        
        # Caption irrelevant
        time_loop = 0
        answer_caption_irrelevant = model.chat(    
            user_prompt="<image>" * len(imgs_url) + prompt_user.format(caption=caption_irrelevant, options=option_answers, question=question),
            images=imgs_url,
        )
        answer_caption_irrelevant = answer_caption_irrelevant.replace("\n", " ").strip()
        while not check_find(answer=answer_caption_irrelevant) and time_loop < 5:
            answer_caption_irrelevant = model.chat(    
                user_prompt="<image>" * len(imgs_url) + prompt_user.format(caption=caption_irrelevant, options=option_answers, question=question),
                images=imgs_url,
            )
            answer_caption_irrelevant = answer_caption_irrelevant.replace("\n", " ").strip()
            time_loop += 1
        if answer_caption_irrelevant.find(str(correct_answer_id)) != -1:
            results_irrelevant += 1
            
        # No caption
        time_loop = 0
        answer_no_caption = model.chat(    
            user_prompt="<image>" * len(imgs_url) + prompt_user_no_caption.format(options=option_answers, question=question),
            images=imgs_url,
        )
        answer_no_caption = answer_no_caption.replace("\n", " ").strip()
        while not check_find(answer=answer_no_caption) and time_loop < 5:
            answer_no_caption = model.chat(    
                user_prompt="<image>" * len(imgs_url) + prompt_user_no_caption.format(options=option_answers, question=question),
                images=imgs_url,
            )
            answer_no_caption = answer_no_caption.replace("\n", " ").strip()
            time_loop += 1
        if answer_no_caption.find(str(correct_answer_id)) != -1:
            results_no_caption += 1
            
        result = {
            "answer_caption_correct": answer_caption_correct,
            "answer_caption_redirected": answer_caption_redirected,
            "answer_caption_completely_wrong": answer_caption_completely_wrong,
            "answer_caption_partially_wrong": answer_caption_partially_wrong,
            "answer_caption_irrelevant": answer_caption_irrelevant,
            "answer_no_caption": answer_no_caption,
        }
        result.update(sample.copy())
        results.append(result)
    
    print(f"Results with correct caption: {results_correct}/{len(data)} = {results_correct/len(data):.4f}")
    print(f"Results with redirected caption: {results_correct_redirected}/{len(data)} = {results_correct_redirected/len(data):.4f}")
    print(f"Results with redirected answer in redirected caption: {result_redirect_redirected}/{len(data)} = {result_redirect_redirected/len(data):.4f}")
    print(f"Results with completely wrong caption: {results_completely_wrong}/{len(data)} = {results_completely_wrong/len(data):.4f}")
    print(f"Results with partially wrong caption: {results_partially_wrong}/{len(data)} = {results_partially_wrong/len(data):.4f}")
    print(f"Results with irrelevant caption: {results_irrelevant}/{len(data)} = {results_irrelevant/len(data):.4f}")
    print(f"Results with no caption: {results_no_caption}/{len(data)} = {results_no_caption/len(data):.4f}")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)
        print(f"Results saved to {f.name}")

    with open((args.output_file).replace(".json", "_config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
        print(f"Configuration saved to {f.name}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
