import argparse
import torch
import torch.nn.functional as F
import os
import json
import copy
import sys
import requests
import functools
import builtins
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from typing import List, Tuple, Optional
from pathlib import Path
from natsort import natsorted

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, PROJECT_ROOT)

from transformers import set_seed
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from utils.llm_layers import add_nss_layers, remove_nss_layers
from utils.qwen_utils import get_demos_qwen, obtain_visual_nss_qwen, obtain_textual_nss_qwen
from utils.nss_controller import NSSController
from utils.qwen_dg import CaptionBiasReducerQwen

# Flush print statements immediately
print = functools.partial(builtins.print, flush=True)

from datetime import datetime

log_file = "oom_log.txt"

def log_oom(sample_info: dict):
    """Ghi log chi tiết khi OOM"""
    with open(log_file, "a") as f:
        f.write(json.dumps(sample_info, ensure_ascii=False) + "\n")

# Template for question formatting
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


def load_image(image_file):
    """Load image from file path or URL."""
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def check_answer_id(result_str):
    for i in range(5):
        if f"{i}" in result_str and all(f"{j}" not in result_str for j in range(5) if j != i):
            return i
    return -1

def eval_model(args):
    """
    Main evaluation function combining NSS, Caption Bias Reduction, and NSS Controller.
    
    Flow:
    1. Load model and processor
    2. Compute NSS directions from demo samples (if enabled)
    3. Add NSS layers to model (static intervention)
    4. Initialize NSS Controller (intelligent NSS application)
    5. Initialize Caption Bias Reducer (dynamic intervention)
    6. For each sample:
        a. Prepare inputs with caption/reasoning information
        b. Apply Caption Bias Reducer hooks (if enabled)
        c. Generate answer with NSS, NSS Controller, and bias reduction active
        d. Remove hooks after generation
    7. Calculate and save results
    """
    model_path = os.path.expanduser(args.model_path)
    
    # ==================== STEP 1: Load Model ====================
    print("=" * 80)
    print("STEP 1: Loading model...")
    print("=" * 80)
    
    # Determine device - use cuda if available, respecting CUDA_VISIBLE_DEVICES
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")
    
    if "qwen" in model_path.lower():
        # Prepare loading kwargs based on memory optimization settings
        load_kwargs = {
            "device_map": "auto",  # Auto will use CUDA_VISIBLE_DEVICES
            "low_cpu_mem_usage": True,
        }
        
        # Quantization options (mutually exclusive)
        if args.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            print("Loading model in 4-bit precision...")
        elif args.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            print("Loading model in 8-bit precision...")
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
            print("Loading model in bfloat16 precision...")
        
        # Offload folder for CPU/disk offloading
        if args.offload_folder:
            load_kwargs["offload_folder"] = args.offload_folder
            print(f"Using offload folder: {args.offload_folder}")
        
        # Load Qwen model
        if "qwen2.5" in model_path.lower():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                cache_dir=cache_dir_path,
                **load_kwargs
            )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                cache_dir=cache_dir_path,
                **load_kwargs
            )
        print(f"model: {model_path}")
        print(model)
        print("✓ Loaded model")
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir_path)
        print("✓ Loaded tokenizer")
        image_processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir_path)
        print("✓ Loaded processor")
        context_len = 8192  # Default for Qwen models
    else:
        print("Error: Currently only Qwen models are supported in this evaluation script.")
    
    # ==================== STEP 2: Compute NSS Directions ====================
    print("\n" + "=" * 80)
    print("STEP 2: Computing NSS directions (static intervention)...")
    print("=" * 80)
    
    demo_inputs_images = None
    demo_inputs_text = None
    nss_vision_stored = None
    nss_text_stored = None
    effective_alpha_image_stored = None
    effective_alpha_text_stored = None
    
    if args.alpha_image != 0 or args.alpha_text != 0:
        print("Loading demos using Qwen-compatible method...")
        try:
            demo_inputs_images, demo_inputs_text = get_demos_qwen(
                args,
                processor=image_processor,
                samples_file=args.samples_file,
                image_folder=args.image_folder,
                cache_dir=cache_dir_path,
                num_demos=args.num_demos,
                num_trials=args.num_trials,
                min_pixels=200704,
                max_pixels=args.max_pixels
            )
            print(f"✓ Successfully loaded {len(demo_inputs_images)} demo samples for visual NSS")
            if demo_inputs_text is not None:
                print(f"✓ Successfully loaded {len(demo_inputs_text)} demo samples for textual NSS")
        except Exception as e:
            print(f"ERROR loading demos: {e}")
            print("NSS will be disabled.")
            args.alpha_image = 0
            args.alpha_text = 0
            demo_inputs_images = None
            demo_inputs_text = None
    else:
        print("NSS disabled (alpha_image=0, alpha_text=0)")
    
    torch.cuda.empty_cache()
    
    # Compute Visual NSS
    if args.alpha_image != 0 and demo_inputs_images is not None:
        print("\nComputing Visual NSS...")
        try:
            nss_vision, _ = obtain_visual_nss_qwen(
                model, args.model_path, cache_dir_path, demo_inputs_images, rank=1
            )
            
            # Validate and normalize
            if not torch.isfinite(nss_vision).all():
                print("Warning: Non-finite values in visual NSS. Cleaning...")
                nss_vision = torch.nan_to_num(nss_vision, nan=0.0, posinf=0.0, neginf=0.0)
            
            nss_norm = torch.norm(nss_vision)
            if nss_norm > 1e-6:
                nss_vision = nss_vision / nss_norm
                nss_vision = nss_vision * 0.1  # Scale down for safety
                print(f"Visual NSS normalized: norm={torch.norm(nss_vision).item():.6f}")
            else:
                print("Warning: NSS vision has near-zero norm, setting to zeros...")
                nss_vision = torch.zeros_like(nss_vision)
            
            nss_vision = torch.clamp(nss_vision, min=-0.5, max=0.5)
            
            # FIX: Mean-pool spatial tokens to get [n_layers, 1, hidden_dim]
            # Current shape: [33, 1024, 1280] -> Target: [33, 1, 1280]
            if nss_vision.dim() == 3:  # [n_layers, n_tokens, hidden_dim]
                nss_vision_pooled = nss_vision.mean(dim=1, keepdim=True)  # [33, 1, 1280]
                print(f"  Pooled visual NSS: {nss_vision.shape} -> {nss_vision_pooled.shape}")
                nss_vision = nss_vision_pooled
            
            # FIX: Check actual layer count in model.visual and adjust NSS shape if needed
            from utils.llm_layers import get_layers
            actual_visual_layers = get_layers(model.visual)
            n_actual_layers = len(actual_visual_layers)
            n_nss_layers = nss_vision.shape[0]
            
            print(f"  NSS layers: {n_nss_layers}, Actual model.visual layers: {n_actual_layers}")
            
            if n_nss_layers != n_actual_layers:
                print(f"  ⚠ Layer count mismatch! Interpolating NSS from {n_nss_layers} to {n_actual_layers} layers")
                # Interpolate along layer dimension: [n_nss_layers, 1, hidden_dim] -> [n_actual_layers, 1, hidden_dim]
                # Squeeze middle dim: [33, 1, 1280] -> [33, 1280]
                nss_vision_2d = nss_vision.squeeze(1)
                # Permute to [batch, channels, length] for interpolate: [33, 1280] -> [1, 1280, 33]
                nss_vision_2d = nss_vision_2d.unsqueeze(0).permute(0, 2, 1)
                # Interpolate: [1, 1280, 33] -> [1, 1280, 32]
                nss_vision_interpolated = F.interpolate(
                    nss_vision_2d,
                    size=n_actual_layers,
                    mode='linear',
                    align_corners=False
                )
                # Permute back and unsqueeze: [1, 1280, 32] -> [32, 1, 1280]
                nss_vision = nss_vision_interpolated.squeeze(0).permute(1, 0).unsqueeze(1)
                print(f"  Interpolated NSS shape: {nss_vision.shape}")
            
            # Store and add to model
            nss_vision_stored = nss_vision.cuda()
            effective_alpha_image_stored = args.alpha_image * 0.1
            
            add_nss_layers(model.visual, nss_vision_stored, alpha=[effective_alpha_image_stored])
            print(f"✓ Added visual NSS with effective_alpha={effective_alpha_image_stored:.3f}")
        except Exception as e:
            print(f"ERROR obtaining visual NSS: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing without visual NSS...")
    
    # Compute Textual NSS
    if args.alpha_text != 0 and demo_inputs_text is not None:
        print("\nComputing Textual NSS...")
        try:
            # Temporarily remove visual NSS to avoid interference
            visual_nss_was_applied = False
            if args.alpha_image != 0 and nss_vision_stored is not None:
                try:
                    remove_nss_layers(model.visual)
                    visual_nss_was_applied = True
                except Exception:
                    pass
            
            nss_text, _ = obtain_textual_nss_qwen(
                model, demo_inputs_text, rank=1
            )
            
            # Re-apply visual NSS if it was applied before
            if visual_nss_was_applied and nss_vision_stored is not None:
                add_nss_layers(model.visual, nss_vision_stored, alpha=[effective_alpha_image_stored])
                print("Re-applied visual NSS after textual NSS computation")
            
            textual_direction = nss_text
            
            # Validate and normalize
            if not torch.isfinite(textual_direction).all():
                print("Warning: Non-finite values in textual NSS. Cleaning...")
                textual_direction = torch.nan_to_num(textual_direction, nan=0.0, posinf=0.0, neginf=0.0)
            
            text_norm = torch.norm(textual_direction)
            if text_norm > 1e-6:
                textual_direction = textual_direction / text_norm
                textual_direction = textual_direction * 0.1
                print(f"Textual NSS normalized: norm={torch.norm(textual_direction).item():.6f}")
            else:
                print("Warning: NSS text has near-zero norm, setting to zeros...")
                textual_direction = torch.zeros_like(textual_direction)
            
            textual_direction = torch.clamp(textual_direction, min=-0.5, max=0.5)
            
            # Store and add to model
            nss_text_stored = textual_direction.cuda()
            effective_alpha_text_stored = args.alpha_text * 0.1
            
            # FIX: Add textual NSS to language_model, not the full model
            if hasattr(model, 'language_model'):
                add_nss_layers(model.language_model, nss_text_stored, alpha=[effective_alpha_text_stored])
                print(f"✓ Added textual NSS to model.language_model with effective_alpha={effective_alpha_text_stored:.3f}")
            elif hasattr(model, 'model'):
                add_nss_layers(model.model, nss_text_stored, alpha=[effective_alpha_text_stored])
                print(f"✓ Added textual NSS to model.model with effective_alpha={effective_alpha_text_stored:.3f}")
            else:
                add_nss_layers(model, nss_text_stored, alpha=[effective_alpha_text_stored])
                print(f"✓ Added textual NSS to model with effective_alpha={effective_alpha_text_stored:.3f}")
        except Exception as e:
            print(f"ERROR obtaining textual NSS: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing without textual NSS...")
    
    torch.cuda.empty_cache()
    
    # ==================== STEP 3: Initialize NSS Controller ====================
    print("\n" + "=" * 80)
    print("STEP 3: Initializing NSS Controller (intelligent NSS application)...")
    print("=" * 80)
    
    nss_controller = None
    if hasattr(args, 'nss_strategy') and args.nss_strategy != "always":
        nss_controller = NSSController(
            strategy=args.nss_strategy,
            confidence_threshold=getattr(args, 'nss_confidence_threshold', 0.7),
            uncertainty_threshold=getattr(args, 'nss_uncertainty_threshold', 0.3),
            alpha_image_base=args.alpha_image,
            alpha_text_base=args.alpha_text,
            enable_dynamic_alpha=getattr(args, 'nss_enable_dynamic_alpha', True),
            verbose=getattr(args, 'nss_verbose', False)
        )
        print(f"✓ NSS Controller initialized with strategy: {args.nss_strategy}")
    else:
        print("NSS Controller disabled (strategy='always')")
    
    # ==================== STEP 4: Initialize Caption Bias Reducer ====================
    print("\n" + "=" * 80)
    print("STEP 4: Initializing Caption Bias Reducer (dynamic intervention)...")
    print("=" * 80)
    
    bias_reducer = None
    if args.use_bias_reduction:
        # Parse layer ranges if specified
        layer_ranges = []
        if hasattr(args, 'bias_layer_ranges') and args.bias_layer_ranges:
            try:
                # Parse format like "0-10,20-27" or "19-33,18-35"
                for range_str in args.bias_layer_ranges.split(','):
                    start, end = map(int, range_str.strip().split('-'))
                    layer_ranges.append((start, end))
                print(f"Parsed layer ranges: {layer_ranges}")
            except Exception as e:
                print(f"Warning: Could not parse bias_layer_ranges '{args.bias_layer_ranges}': {e}")
                print("Applying bias reduction to all layers")
                layer_ranges = []
        
        bias_reducer = CaptionBiasReducerQwen(
            model=model,
            processor=image_processor,
            alpha=args.bias_alpha,
            top_k=args.bias_top_k,
            verbose=args.bias_verbose,
            layer_ranges=layer_ranges
        )
        print("✓ Caption Bias Reducer initialized:")
        print(f"  - Alpha: {args.bias_alpha}")
        print(f"  - Top-k: {args.bias_top_k}")
        print(f"  - Layer ranges: {layer_ranges if layer_ranges else 'all layers'}")
        print(f"  - Verbose: {args.bias_verbose}")
    else:
        print("Caption Bias Reduction disabled")
    
    # ==================== STEP 5: Prepare for Evaluation ====================
    print("\n" + "=" * 80)
    print("STEP 5: Starting evaluation...")
    print("=" * 80)
    
    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    
    # Save configuration to file
    config_file = answer_file.replace('.jsonl', '_config.json').replace('.json', '_config.json')
    config_data = {
        "timestamp": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": args.model_path,
        "model_base": args.model_base,
        "answer_file": args.answer_file,
        "data_path": args.data_path,
        "nss_config": {
            "alpha_image": args.alpha_image,
            "alpha_text": args.alpha_text,
            "effective_alpha_image": effective_alpha_image_stored if effective_alpha_image_stored else 0,
            "effective_alpha_text": effective_alpha_text_stored if effective_alpha_text_stored else 0,
            "nss_enabled": args.alpha_image != 0 or args.alpha_text != 0,
            "num_demos": args.num_demos,
            "mask_ratio": args.mask_ratio,
        },
        "nss_controller_config": {
            "enabled": nss_controller is not None,
            "strategy": args.nss_strategy if hasattr(args, 'nss_strategy') else "always",
            "confidence_threshold": args.nss_confidence_threshold if hasattr(args, 'nss_confidence_threshold') else None,
            "uncertainty_threshold": args.nss_uncertainty_threshold if hasattr(args, 'nss_uncertainty_threshold') else None,
            "enable_dynamic_alpha": args.nss_enable_dynamic_alpha if hasattr(args, 'nss_enable_dynamic_alpha') else None,
        },
        "bias_reduction_config": {
            "enabled": args.use_bias_reduction,
            "alpha": args.bias_alpha,
            "top_k": args.bias_top_k,
            "verbose": args.bias_verbose,
            "layer_ranges": args.bias_layer_ranges if hasattr(args, 'bias_layer_ranges') and args.bias_layer_ranges else "all",
        },
        "generation_config": {
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.1,
            "top_p": 0.001,
            "repetition_penalty": 1.05,
            "do_sample": True,
            "use_cache": True,
        },
        "image_config": {
            "max_pixels": args.max_pixels,
            "max_frames": args.max_frames,
        },
        "memory_optimization": {
            "load_in_8bit": args.load_in_8bit,
            "load_in_4bit": args.load_in_4bit,
            "offload_folder": args.offload_folder,
            "batch_size": args.batch_size,
        },
        "other_config": {
            "seed": args.seed,
            "conv_mode": args.conv_mode,
            "num_trials": args.num_trials,
        }
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Configuration saved to: {config_file}\n")
    
    with open(args.data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    # Initialize counters
    caculate_total = 0
    count1, count2, count3, count4, count5, count6 = 0, 0, 0, 0, 0, 0
    redirected_count1, redirected_count2, redirected_count3, redirected_count4 = 0, 0, 0, 0
    
    if Path(answer_file).exists():
        answer_file = answer_file.replace('.jsonl', '_new.jsonl').replace('.json', '_new.json')
        print(f"Answer file exists, using: {answer_file}")
    
    ans_file = open(answer_file, "w")
    
    # ==================== STEP 6: Evaluate Samples ====================
    # FIX: Load processor ONCE outside loop to prevent memory leak
    print("Loading processor for evaluation...")
    processor = AutoProcessor.from_pretrained(args.model_path, cache_dir=cache_dir_path)
    
    # Evaluate only last 1000 samples for quicker testing
    samples_to_process = samples[-1000:]
    for sample_idx, sample in enumerate(tqdm(samples_to_process, desc="Evaluating samples")):
        try:
            # Reduced debug prints - only show progress every 10 samples
            if sample_idx % 10 == 0:
                print(f"\n[Progress] Processing sample {sample_idx}: {sample.get('video_id', 'image_id')}")
            
            # Load images
            if "video_id" in sample:
                images_path = (Path(args.image_folder) / sample["video_id"]).glob("*.jpg")
                list_images_path = natsorted([str(img) for img in images_path])
            else:
                list_images_path = [Path(args.image_folder) / sample["image_id"]]
            
            if not list_images_path:
                print(f"\nWarning: No images found for {sample['video_id']}")
                continue

            # Use max_pixels directly to match baseline behavior
            actual_max_pixels = args.max_pixels
                
            images_input = [{"type": "image", "image": str(image), "max_pixels": actual_max_pixels} for image in list_images_path]
            
            result_list = []
            type_caption = ["caption_correct", "caption_completely_wrong", "caption_partially_wrong", "caption_redirected", "caption_irrelevance", "no_caption"]
            for index_res, res_type in enumerate(type_caption):
                
                number_choices = len([sample[f"answer_choice_{i}"] for i in range(5) if f"answer_choice_{i}" in sample])
                option_answers = "\n".join(str(idx) + ": " + sample[f"answer_choice_{idx}"] for idx in range(number_choices))

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
                
                # Create messages for outline
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant tasked with selecting the CORRECT NUMBER ID based on the content of the provided images.",
                    },
                    {
                        "role": "user",
                        "content": [
                            *images_input,
                            {"type": "text", "text": str(question_template) + "Instruction: Choose the **CORRECT NUMBER ID** answer for the question below, you must response with the **NUMBER ID** answer, don't include text."},
                        ],
                    }
                ]
                
                # Process inputs (processor already loaded outside loop)
                text_message = processor.apply_chat_template(
                    copy.deepcopy(messages), tokenize=False, add_generation_prompt=True
                )
                
                image_inputs_outline, video_inputs_outline = process_vision_info(messages)
                
                inputs_outline = processor(
                    text=[text_message],
                    images=image_inputs_outline,
                    videos=video_inputs_outline,
                    padding=True,
                    return_tensors="pt",
                )
                inputs_outline = inputs_outline.to("cuda")
                
                # ==================== Generate with Outline ====================
                try:
                    # Prepare bias reducer if enabled
                    if bias_reducer:
                        bias_reducer.prepare_for_generation(inputs_outline.input_ids, text_message)
                    
                    # Generation with timeout protection (using inference_mode)
                    with torch.inference_mode():
                        # Add max_time to prevent infinite generation
                        gen_output = model.generate(
                            **inputs_outline,
                            max_new_tokens=args.max_new_tokens,
                            max_time=30.0,  # Timeout after 30 seconds
                            temperature=0.1,
                            top_p=0.001,
                            repetition_penalty=1.05,
                            do_sample=True,
                            num_beams=1,
                            use_cache=True,
                            output_attentions=False,
                            return_dict_in_generate=True
                        )
                    
                    # Cleanup bias reducer
                    if bias_reducer:
                        bias_reducer.cleanup_after_generation()
                    
                    gen_ids = gen_output.sequences
                    trimmed_ids = [out[len(inp):] for inp, out in zip(inputs_outline.input_ids, gen_ids)]
                    output_text_raw = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0].replace("</think>", "").replace("<think>", "").strip()
                    output_text_gen = check_answer_id(output_text_raw)
                    
                    # Retry logic like baseline - regenerate if answer is invalid
                    retry_count = 0
                    max_retries = 3
                    while output_text_gen == -1 and retry_count < max_retries:
                        if retry_count == 0:  # Only print first retry
                            if "video_id" in sample:
                                print(f"⚠ Invalid output for {sample['video_id']}: '{output_text_gen}'. Retrying generation...")
                            else:
                                print(f"⚠ Invalid output for sample index {sample_idx}: '{output_text_gen}'. Retrying generation...")
                        retry_count += 1
                        gen_output = model.generate(
                            **inputs_outline,
                            max_new_tokens=args.max_new_tokens,
                            max_time=30.0,  # Timeout for retry too
                            temperature=0.1,
                            top_p=0.001,
                            repetition_penalty=1.05,
                            do_sample=True,
                            num_beams=1,
                            use_cache=True,
                            output_attentions=False,
                            return_dict_in_generate=True
                        )
                        gen_ids = gen_output.sequences
                        trimmed_ids = [out[len(inp):] for inp, out in zip(inputs_outline.input_ids, gen_ids)]
                        output_text_raw = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0].replace("</think>", "").replace("<think>", "").strip()
                        output_text_gen = check_answer_id(output_text_raw)
                        
                    # Only log if still invalid after retries
                    if output_text_gen == -1:
                        if "video_id" in sample:
                            print(f"⚠ Still invalid output for {sample['video_id']} after {max_retries} retries: '{output_text_gen}'")
                        else:
                            print(f"⚠ Still invalid output for sample index {sample_idx} after {max_retries} retries: '{output_text_gen}'")
                    # Clean up
                    del gen_output, gen_ids, trimmed_ids
                    
                except Exception as e:
                    print(f"\nError generating with outline: {e}")
                    output_text_gen = -1
                    if bias_reducer:
                        bias_reducer.cleanup_after_generation()

                # No need to call check_answer_id again - already an int
                result_list.append(output_text_gen)
                
                # Cleanup memory after each caption type
                del inputs_outline, text_message, messages
                torch.cuda.empty_cache()
                
            
            # ==================== Evaluate Results ====================
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
    
    # ==================== STEP 7: Calculate and Save Results ====================
    print("\n" + "=" * 80)
    print("STEP 7: Evaluation complete - calculating results...")
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
    
    # Print NSS Controller statistics if used
    if nss_controller is not None:
        print("\n" + "=" * 80)
        print("NSS Controller Statistics:")
        print("=" * 80)
        nss_controller.print_stats()
    
    # Cleanup
    remove_nss_layers(model)
    if hasattr(model, 'visual'):
        remove_nss_layers(model.visual)
    
    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Combined NSS and Caption Bias Reduction Evaluation")
    
    # Model and data arguments
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./../data/keyframes_full")
    parser.add_argument("--samples-file", type=str, default="./experiments/data/gemini_full_dataset.json", help="File containing demo samples for NSS computation")
    parser.add_argument("--data-path", type=str, default="./../data/data_samples/baseline_gemini_v2.json")
    parser.add_argument("--answer-file", type=str, default="./results/combined_nss_bias_reduction_enhanced.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    
    # NSS arguments
    parser.add_argument("--num_demos", type=int, default=30, help="Number of demo samples for NSS computation")
    parser.add_argument("--alpha_image", type=float, default=0.0, help="Alpha for visual NSS")
    parser.add_argument("--alpha_text", type=float, default=0.0, help="Alpha for textual NSS")
    parser.add_argument("--mask_ratio", type=float, default=0.99)
    
    # NSS Controller arguments - Intelligent NSS application
    parser.add_argument("--nss_strategy", type=str, default="always",
                       choices=["always", "confidence", "uncertainty", "adaptive", "selective"],
                       help="Strategy for NSS application")
    parser.add_argument("--nss_confidence_threshold", type=float, default=0.7,
                       help="Confidence threshold for confidence-based strategy")
    parser.add_argument("--nss_uncertainty_threshold", type=float, default=0.3,
                       help="Uncertainty threshold for uncertainty-based strategy")
    parser.add_argument("--nss_enable_dynamic_alpha", action='store_true', default=True,
                       help="Dynamically adjust alpha based on confidence/uncertainty")
    parser.add_argument("--nss_verbose", action='store_true', default=False,
                       help="Print NSS controller decision logs")
    
    # Caption Bias Reduction arguments
    parser.add_argument("--use_bias_reduction", action='store_true', default=False, 
                       help="Enable caption bias reduction")
    parser.add_argument("--bias_alpha", type=float, default=0.03, 
                       help="Alpha for bias reduction adjustment strength")
    parser.add_argument("--bias_top_k", type=int, default=10, 
                       help="Number of top important image tokens for bias reduction")
    parser.add_argument("--bias_verbose", action='store_true', 
                       help="Print bias reduction debug info")
    parser.add_argument("--bias_layer_ranges", type=str, default=None,
                       help="Layer ranges to apply bias reduction (e.g., '19-33,18-35'). If not specified, applies to all layers")
    
    # Generation arguments
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--use_cache", action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_trials", type=int, default=4, 
                       help="Number of samples to evaluate (None=all)")
    
    # Image/video arguments
    parser.add_argument("--max_pixels", type=int, default=602112)
    parser.add_argument("--max_frames", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    
    # Memory optimization arguments
    parser.add_argument("--load_in_8bit", action='store_true')
    parser.add_argument("--load_in_4bit", action='store_true')  
    parser.add_argument("--offload_folder", type=str, default=None)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    eval_model(args)
