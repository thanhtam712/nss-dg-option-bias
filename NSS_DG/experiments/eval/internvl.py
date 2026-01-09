import os
import torch
import json
import sys
import argparse
from tqdm import tqdm
from typing import List, Tuple, Optional
from pathlib import Path
from natsort import natsorted


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, PROJECT_ROOT)

from transformers import set_seed
from transformers import AutoTokenizer, AutoModel

from utils.llm_layers import add_nss_layers, remove_nss_layers
import builtins, functools
print = functools.partial(builtins.print, flush=True)

# Import InternVL utility functions
from utils.internvl_utils import (
    load_image_for_internvl,
    get_demos_internvl,
    obtain_visual_nss_internvl,
    obtain_textual_nss_internvl
)
from utils.internvl_dg import CaptionBiasReducerInternVL

# Template for question formatting with caption
reason_template = """
Question:
{question}

Answer options:
{options}

Here are some additional information which are text descriptions based on the image to assist you for answering the later question. Note, the information could be irrelevant, missing some information or inaccurate, please use it with caution: {information}
"""

# Template for no caption case
reason_template_no_caption = """
Question:
{question}

Answer options:
{options}
"""

cache_dir_path = "./../cache_dir"


def check_answer_id(result_str):
    """Extract answer ID from model output string."""
    for i in range(5):
        if str(i) in result_str:
            return i
    return -1


def eval_model(args):
    """
    Main evaluation function for InternVL with NSS intervention and Caption Bias Reduction.
    """
    model_path = os.path.expanduser(args.model_path)
    
    # ==================== STEP 1: Load Model ====================
    print("=" * 80)
    print("STEP 1: Loading InternVL model...")
    print("=" * 80)
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Prepare loading kwargs
    load_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    
    # Quantization options
    if args.load_in_4bit:
        print("Loading model in 4-bit precision")
        load_kwargs["load_in_4bit"] = True
    elif args.load_in_8bit:
        print("Loading model in 8-bit precision")
        load_kwargs["load_in_8bit"] = True
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        print("Using bfloat16 precision")
    
    # Offload folder
    if args.offload_folder:
        load_kwargs["offload_folder"] = args.offload_folder
        print(f"Offloading to: {args.offload_folder}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = AutoModel.from_pretrained(
        model_path,
        cache_dir=cache_dir_path,
        **load_kwargs
    )
    print("✓ Model loaded")
    print(model_path)
    print(model)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        device=device,
        cache_dir=cache_dir_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"✓ Set pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
    
    print("✓ Tokenizer loaded")
    
    model.eval()
    print("✓ Model set to evaluation mode")
    
    # ==================== STEP 2: Compute NSS Directions ====================
    print("\n" + "=" * 80)
    print("STEP 2: Computing NSS directions...")
    print("=" * 80)
    
    demo_inputs_images = None
    demo_inputs_text = None
    NSS_vision_stored = None
    NSS_text_stored = None
    effective_alpha_image_stored = None
    effective_alpha_text_stored = None
    
    if args.alpha_image != 0 or args.alpha_text != 0:
        print(f"Loading demos for NSS computation...")
        demo_inputs_images, demo_inputs_text = get_demos_internvl(
            args,
            model,
            tokenizer,
            image_folder=args.image_folder,
            sample_file=args.samples_file,
            num_demos=args.num_demos,
            num_trials=getattr(args, 'num_trials', 30),
            mask_ratio=getattr(args, 'mask_ratio', 0.5),
            patch_size=14,
            input_size=args.input_size,
            max_num=args.max_num_patches
        )
        print(f"✓ Successfully loaded demo samples")
    else:
        print("NSS disabled")
    
    torch.cuda.empty_cache()
    
    # Compute Visual NSS
    if args.alpha_image != 0 and demo_inputs_images is not None:
        print("\nComputing Visual NSS...")
        NSS_vision, _ = obtain_visual_nss_internvl(model, demo_inputs_images, rank=1)
        NSS_vision_stored = NSS_vision.to(device)
        effective_alpha_image_stored = args.alpha_image
        
        vision_model = model.vision_model if hasattr(model, 'vision_model') else model.vision_encoder
        add_nss_layers(vision_model, NSS_vision_stored, alpha=[effective_alpha_image_stored])
        print(f"✓ Added visual NSS with alpha={effective_alpha_image_stored:.3f}")
    
    # Compute Textual NSS
    if args.alpha_text != 0 and demo_inputs_text is not None:
        print("\nComputing Textual NSS...")
        
        # Temporarily remove visual NSS
        visual_nss_was_applied = False
        if NSS_vision_stored is not None:
            vision_model = model.vision_model if hasattr(model, 'vision_model') else model.vision_encoder
            remove_nss_layers(vision_model)
            visual_nss_was_applied = True
        
        NSS_text, _ = obtain_textual_nss_internvl(model, demo_inputs_text, rank=1)
        
        # Re-apply visual NSS
        if visual_nss_was_applied and NSS_vision_stored is not None:
            vision_model = model.vision_model if hasattr(model, 'vision_model') else model.vision_encoder
            add_nss_layers(vision_model, NSS_vision_stored, alpha=[effective_alpha_image_stored])
        
        NSS_text_stored = NSS_text.to(device)
        effective_alpha_text_stored = args.alpha_text
        
        language_model = model.language_model if hasattr(model, 'language_model') else model.llm
        add_nss_layers(language_model, NSS_text_stored, alpha=[effective_alpha_text_stored])
        print(f"✓ Added textual NSS with alpha={effective_alpha_text_stored:.3f}")
    
    torch.cuda.empty_cache()
    
    # ==================== STEP 3: Initialize Caption Bias Reducer ====================
    print("\n" + "=" * 80)
    print("STEP 3: Initializing Caption Bias Reducer...")
    print("=" * 80)
    
    bias_reducer = None
    if args.use_bias_reduction:
        # Parse layer ranges
        layer_ranges = []
        if hasattr(args, 'bias_layer_ranges') and args.bias_layer_ranges:
            try:
                for range_str in args.bias_layer_ranges.split(','):
                    start, end = map(int, range_str.strip().split('-'))
                    layer_ranges.append((start, end))
                print(f"Parsed layer ranges: {layer_ranges}")
            except Exception as e:
                print(f"Warning: Could not parse bias_layer_ranges: {e}")
                layer_ranges = []
        
        bias_reducer = CaptionBiasReducerInternVL(
            model=model,
            tokenizer=tokenizer,
            alpha=args.bias_alpha,
            top_k=args.bias_top_k,
            verbose=args.bias_verbose,
            layer_ranges=layer_ranges
        )
        print("✓ Caption Bias Reducer initialized:")
        print(f"  - Alpha: {args.bias_alpha}")
        print(f"  - Top-k: {args.bias_top_k}")
        print(f"  - Layer ranges: {layer_ranges if layer_ranges else 'all layers'}")
    else:
        print("Caption Bias Reduction disabled")
    
    # ==================== STEP 4: Prepare for Evaluation ====================
    print("\n" + "=" * 80)
    print("STEP 4: Starting evaluation...")
    print("=" * 80)
    
    # Prepare output files
    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    
    # Save configuration
    config_file = answer_file.replace('.jsonl', '_config.json').replace('.json', '_config.json')
    config_data = {
        "timestamp": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": args.model_path,
        "answer_file": args.answer_file,
        "data_path": args.data_path,
        "nss_config": {
            "alpha_image": args.alpha_image,
            "alpha_text": args.alpha_text,
            "effective_alpha_image": effective_alpha_image_stored if effective_alpha_image_stored else 0,
            "effective_alpha_text": effective_alpha_text_stored if effective_alpha_text_stored else 0,
            "nss_enabled": args.alpha_image != 0 or args.alpha_text != 0,
            "num_demos": args.num_demos,
        },
        "bias_reduction_config": {
            "enabled": args.use_bias_reduction,
            "alpha": args.bias_alpha,
            "top_k": args.bias_top_k,
            "verbose": args.bias_verbose,
            "layer_ranges": args.bias_layer_ranges if hasattr(args, 'bias_layer_ranges') and args.bias_layer_ranges else "all",
        },
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.1,
            "top_p": 0.001,
        },
        "image_config": {
            "input_size": args.input_size,
            "max_num_patches": args.max_num_patches,
            "max_frames": args.max_frames,
        },
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Configuration saved to: {config_file}\n")
    
    # Load evaluation samples
    print(f"Loading samples from: {args.data_path}")
    with open(args.data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"✓ Loaded {len(samples)} samples")
    
    # Initialize counters
    caculate_total = 0
    count1, count2, count3, count4, count5, count6 = 0, 0, 0, 0, 0, 0
    redirected_count1, redirected_count2, redirected_count3, redirected_count4 = 0, 0, 0, 0
    
    if Path(answer_file).exists():
        answer_file = answer_file.replace('.jsonl', '_new.jsonl').replace('.json', '_new.json')
        print(f"Answer file exists, using: {answer_file}")
    
    ans_file = open(answer_file, "w")
    
    # ==================== STEP 5: Evaluate Samples ====================
    samples_to_process = samples[-1000:]
    for sample_idx, sample in enumerate(tqdm(samples_to_process, desc="Evaluating")):
        try:
            if sample_idx % 10 == 0:
                print(f"\n[Progress] Sample {sample_idx}: {sample.get('image_id', f'sample_{sample_idx}')}")
            
            # Get image path

            if "video_id" in sample:
                if model_path.endswith("InternVL3_5-8B-Instruct"):
                    images_path = (Path(args.image_folder) / sample["video_id"]).glob("*.jpg")
                    list_images_path = natsorted([str(img) for img in images_path])[:5]
                else:
                    images_path = (Path(args.image_folder) / sample["video_id"]).glob("*.jpg")
                    list_images_path = natsorted([str(img) for img in images_path])
            else:
                list_images_path = [Path(args.image_folder) / sample["image_id"]]
            
            if not list_images_path:
                print(f"\nWarning: No images for {sample['image_id']}. Skipping.")
                continue
            
            # Load images
            pixel_values_list = []
            for image_path in list_images_path:
                pixel_values = load_image_for_internvl(
                    image_path,
                    input_size=args.input_size,
                    max_num=args.max_num_patches
                )
                pixel_values_list.append(pixel_values)
            
            if not pixel_values_list:
                print(f"\nWarning: Failed to load images for sample {sample_idx}. Skipping.")
                continue
            
            pixel_values = torch.cat(pixel_values_list, dim=0).to(device)
            
            # Format answer options
            number_choices = len([sample[f"answer_choice_{i}"] for i in range(5) if f"answer_choice_{i}" in sample])
            option_answers = "\n".join(str(idx) + ": " + sample[f"answer_choice_{idx}"] for idx in range(number_choices))
            
            image_tokens = '<image>\n' * len(list_images_path)
            instruction = "Choose the **CORRECT NUMBER ID** answer for the question below, you must response with the **NUMBER ID** answer, don't include text."
            
            # Evaluate 6 caption types
            result_list = []
            type_caption = ["caption_correct", "caption_completely_wrong", "caption_partially_wrong", 
                           "caption_redirected", "caption_irrelevance", "no_caption"]
            
            for index_res, res_type in enumerate(type_caption):
                # Prepare question template
                if res_type == "no_caption":
                    question_template = reason_template_no_caption.format(
                        question=sample["question"],
                        options=option_answers,
                    )
                else:
                    question_template = reason_template.format(
                        question=sample["question"],
                        options=option_answers,
                        information=sample.get(res_type, "")
                    )
                
                # Create prompt
                prompt = f"{image_tokens}{question_template}\n{instruction}"
                
                # Generate answer with bias reduction
                try:
                    # Prepare bias reducer if enabled and there's caption info
                    if bias_reducer and res_type != "no_caption":
                        # Tokenize to get input_ids for token identification
                        inputs = tokenizer(prompt, return_tensors="pt")
                        input_ids = inputs.input_ids.to(device)
                        
                        # Prepare bias reducer
                        bias_reducer.prepare_for_generation(input_ids, prompt)
                    
                    # Generate with model
                    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        response = model.chat(
                            tokenizer,
                            pixel_values,
                            prompt,
                            generation_config={
                                'max_new_tokens': args.max_new_tokens,
                                'temperature': 0.1,
                                'top_p': 0.001,
                                'num_beams': 1,
                                'do_sample': True,
                                'repetition_penalty': 1.05,
                            }
                        )
                    
                    # Cleanup bias reducer
                    if bias_reducer and res_type != "no_caption":
                        bias_reducer.cleanup_after_generation()
                    
                    output_text = response.strip()
                    output_id = check_answer_id(output_text)

                    # Retry logic for invalid outputs
                    max_retries = 1
                    if output_id == -1:
                        if "video_id" in sample:
                            print(f"⚠ Still invalid output for {sample['video_id']} after {max_retries} retries: '{output_text}'")
                        else:
                            print(f"⚠ Still invalid output for sample index {sample_idx} after {max_retries} retries: '{output_text}'")
                    
                    result_list.append(output_id)
                    
                except Exception as e:
                    print(f"\nError generating for {res_type}: {e}")
                    result_list.append(-1)
                    
                    # Cleanup on error
                    if bias_reducer:
                        try:
                            bias_reducer.cleanup_after_generation()
                        except:
                            pass
                
                # Memory cleanup
                torch.cuda.empty_cache()
            
            # Calculate results
            correct_answer = int(sample["correct_answer_id"])
            outline_answer = int(sample["outline_answer_id"])
            
            try: 
                result_caption_true = int(result_list[0])
            except Exception:
                print("Error parsing caption true result")
                result_caption_true = -1
            try:
                result_caption_completely_wrong = int(result_list[1])
            except Exception:
                print("Error parsing caption completely wrong result")
                result_caption_completely_wrong = -1
            try:
                result_caption_partially_wrong = int(result_list[2])
            except Exception:
                print("Error parsing caption partially wrong result")
                result_caption_partially_wrong = -1
            try:
                result_caption_redirected = int(result_list[3])
            except Exception:
                print("Error parsing caption redirected result")
                result_caption_redirected = -1
            try:
                result_caption_irrelevance = int(result_list[4])
            except Exception:
                print("Error parsing caption irrelevance result")
                result_caption_irrelevance = -1
            try:
                result_no_caption = int(result_list[5])
            except Exception:
                print("Error parsing no caption result")
                result_no_caption = -1

            caculate_total += 1
            
            # count1, count2, count3, count4, count5, count6: caption true, completely wrong, partially wrong, redirected, irrelevance, no caption 
            if result_caption_true == correct_answer:
                count1 += 1
            if result_caption_completely_wrong == correct_answer:
                count2 += 1
            if result_caption_partially_wrong == correct_answer:
                count3 += 1
            if result_caption_redirected == correct_answer:
                count4 += 1
            if result_caption_irrelevance == correct_answer:
                count5 += 1
            if result_no_caption == correct_answer:
                count6 += 1
            
            if result_caption_true == outline_answer:
                redirected_count1 += 1
            if result_caption_completely_wrong == outline_answer:
                redirected_count2 += 1
            if result_caption_partially_wrong == outline_answer:
                redirected_count3 += 1
            if result_caption_redirected == outline_answer:
                redirected_count4 += 1
                
            
            if "video_id" in sample:
                print(f"[Result] Video ID: {sample['video_id']}")
            else:
                print(f"[Result] Sample Index: {sample_idx}")
            print(f"count: {count1}, {count2}, {count3}, {count4}, {count5}, {count6} / {caculate_total}")
            print(f"redirected_count: {redirected_count1}, {redirected_count2}, {redirected_count3}, {redirected_count4} / {caculate_total}")

            # Save result
            result = {
                "id": sample["video_id"] if "video_id" in sample else sample.get("image_id", f"sample_{sample_idx}"),
                "question": sample["question"],
                "correct_answer_id": correct_answer,
                "outline_answer_id": outline_answer,
                "prediction_caption_true": result_caption_true,
                "prediction_caption_completely_wrong": result_caption_completely_wrong,
                "prediction_caption_partially_wrong": result_caption_partially_wrong,
                "prediction_caption_redirected": result_caption_redirected,
                "prediction_caption_irrelevance": result_caption_irrelevance,
                "prediction_no_caption": result_no_caption,
            }
            ans_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            ans_file.flush()
            
            # Aggressive memory cleanup every 10 samples
            if sample_idx % 10 == 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                if sample_idx % 50 == 0:
                    print(f"\n[Memory] CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    print(f"[Memory] CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\nError processing sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
                info = {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "sample_idx": sample_idx,
                    "input_name": sample.get("name", None),
                    "image_shape": getattr(sample.get("image", None), "shape", None),
                    "num_frames": sample.get("num_frames", None),
                    "cuda_allocated_MB": torch.cuda.memory_allocated() / 1024**2,
                    "cuda_reserved_MB": torch.cuda.memory_reserved() / 1024**2,
                    "error": str(e),
                }
                print(f"[OOM] Sample {sample_idx}: {info}")
                log_oom(info)

                # dọn dẹp
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise
    
    # ==================== STEP 6: Calculate and Save Results ====================
    print("\n" + "=" * 80)
    print("STEP 6: Evaluation complete - calculating results...")
    print("=" * 80)
    
    # Calculate accuracy metrics
    total_samples = len(samples)
    accuracy_caption_true = count1 / total_samples * 100
    accuracy_caption_completely_wrong = count2 / total_samples * 100
    accuracy_caption_partially_wrong = count3 / total_samples * 100
    accuracy_caption_redirected = count4 / total_samples * 100
    accuracy_caption_irrelevance = count5 / total_samples * 100
    accuracy_no_caption = count6 / total_samples * 100

    outline_rate_caption_true = redirected_count1 / total_samples * 100
    outline_rate_caption_completely_wrong = redirected_count2 / total_samples * 100
    outline_rate_caption_partially_wrong = redirected_count3 / total_samples * 100
    outline_rate_caption_redirected = redirected_count4 / total_samples * 100
    
    print("list: caption_true, caption_redirected, caption_completely_wrong, caption_partially_wrong, caption_irrelevance, no_caption")
    print(f"\nCaption True Accuracy: {accuracy_caption_true:.2f}% ({count1}/{total_samples})")
    print(f"Caption Completely Wrong Accuracy: {accuracy_caption_completely_wrong:.2f}% ({count2}/{total_samples})")
    print(f"Caption Partially Wrong Accuracy: {accuracy_caption_partially_wrong:.2f}% ({count3}/{total_samples})")
    print(f"Caption Redirected Accuracy: {accuracy_caption_redirected:.2f}% ({count4}/{total_samples})")
    print(f"Caption Irrelevance Accuracy: {accuracy_caption_irrelevance:.2f}% ({count5}/{total_samples})")
    print(f"No Caption Accuracy: {accuracy_no_caption:.2f}% ({count6}/{total_samples})")
    
    ans_file.close()
    print(f"\n✓ Saved answers to: {answer_file}")
    
    # Save accuracy results to config file
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        config_data["results"] = {
            "accuracy": {
                "correct_caption_true": count1,
                "correct_caption_completely_wrong": count2,
                "correct_caption_partially_wrong": count3,
                "correct_caption_redirected": count4,
                "correct_caption_irrelevance": count5,
                "correct_no_caption": count6,
                "total": total_samples,
            },
            "outline_rate": {
                "redirected_caption_true": redirected_count1,
                "redirected_caption_completely_wrong": redirected_count2,
                "redirected_caption_partially_wrong": redirected_count3,   
                "redirected_caption_redirected": redirected_count4,
                "total": total_samples,
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Updated config with results: {config_file}")
    except Exception as e:
        print(f"Warning: Could not update config file with results: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model-path", type=str, default="OpenGVLab/InternVL3_5-8B-Instruct", help="Path to the InternVL model")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--offload_folder", type=str, default=None)
    
    # Data arguments
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--answer-file", type=str, required=True)
    
    # NSS arguments
    parser.add_argument("--alpha_image", type=float, default=0.0)
    parser.add_argument("--alpha_text", type=float, default=0.0)
    parser.add_argument("--num_demos", type=int, default=50)
    parser.add_argument("--num_trials", type=int, default=4)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--samples-file", type=str, default=None)
    
    # Bias reduction arguments
    parser.add_argument("--use_bias_reduction", action="store_true", 
                       help="Enable caption bias reduction")
    parser.add_argument("--bias_alpha", type=float, default=0.1,
                       help="Strength of bias reduction (0.0-1.0)")
    parser.add_argument("--bias_top_k", type=int, default=10,
                       help="Number of top important image tokens")
    parser.add_argument("--bias_verbose", action="store_true",
                       help="Print debug info for bias reduction")
    parser.add_argument("--bias_layer_ranges", type=str, default=None,
                       help="Layer ranges for bias reduction, e.g., '0-10,20-30'")
    
    # Image arguments
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--max_num_patches", type=int, default=12)
    parser.add_argument("--max_frames", type=int, default=20)
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.001)
    parser.add_argument("--do_sample", action="store_true")
    
    # NSS Controller arguments
    parser.add_argument("--nss_strategy", type=str, default="always",
                       choices=["always", "confidence", "uncertainty", "adaptive", "selective"],
                       help="NSS application strategy")
    parser.add_argument("--nss_confidence_threshold", type=float, default=0.7,
                       help="Confidence threshold for NSS controller")
    parser.add_argument("--nss_uncertainty_threshold", type=float, default=0.3,
                       help="Uncertainty threshold for NSS controller")
    parser.add_argument("--nss_enable_dynamic_alpha", action='store_true', default=True,
                       help="Enable dynamic alpha adjustment in NSS controller")
    parser.add_argument("--nss_verbose", action='store_true', default=False,
                       help="Enable verbose output for NSS controller")

    # Other
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    eval_model(args)
