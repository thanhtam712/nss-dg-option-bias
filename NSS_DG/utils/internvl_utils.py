import os
import sys
import json
import torch
import random
import requests
import numpy as np
import gc
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from typing import List, Tuple, Optional
from contextlib import nullcontext
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


def get_vision_model(model):
    """Get vision model from InternVL model with proper fallback."""
    if hasattr(model, 'vision_model'):
        return model.vision_model
    elif hasattr(model, 'vision_encoder'):
        return model.vision_encoder
    elif hasattr(model, 'vision_tower'):
        return model.vision_tower
    elif hasattr(model, 'visual'):
        return model.visual
    else:
        attrs = [attr for attr in dir(model) if not attr.startswith('_') and 'vision' in attr.lower()]
        raise AttributeError(
            f"Could not find vision model in InternVL. "
            f"Model type: {type(model).__name__}. "
            f"Available vision-related attrs: {attrs}"
        )


def get_language_model(model):
    """Get language model from InternVL model with proper fallback."""
    if hasattr(model, 'language_model'):
        return model.language_model
    elif hasattr(model, 'llm'):
        return model.llm
    elif hasattr(model, 'language'):
        return model.language
    elif hasattr(model, 'model'):
        return model.model
    else:
        attrs = [attr for attr in dir(model) if not attr.startswith('_') and ('language' in attr.lower() or 'llm' in attr.lower())]
        raise AttributeError(f"Could not find language model in InternVL. Model type: {type(model).__name__}. Available language-related attrs: {attrs}")


def load_image(image_file):
    """Load image from file path or URL."""
    if hasattr(image_file, '__fspath__') or hasattr(image_file, 'as_posix'):
        image_file_str = str(image_file)
    else:
        image_file_str = image_file
    
    if image_file_str.startswith('http') or image_file_str.startswith('https'):
        response = requests.get(image_file_str)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        del response
    else:
        image = Image.open(image_file_str).convert('RGB')
    return image


def build_transform(input_size):
    """Build image transform for InternVL"""
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio for dynamic preprocessing"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamic preprocessing for InternVL models"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    del resized_img
    if 'thumbnail_img' in locals():
        del thumbnail_img
    
    return processed_images


def mask_patches(tensor, indices, patch_size=14):
    """Replace specified patches with per-channel mean for InternVL."""
    original_shape = tensor.shape
    device = tensor.device

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    elif tensor.dim() != 3:
        raise ValueError(f"Expected 2D, 3D or 4D tensor, got {tensor.dim()}D")
    
    C, H, W = tensor.shape
    new_tensor = tensor.clone()

    mean_values = tensor.mean(dim=(1, 2), keepdim=True)

    patches_per_row = W // patch_size if patch_size > 0 else 0
    if patches_per_row == 0:
        del mean_values
        return tensor.reshape(original_shape)

    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(list(indices), dtype=torch.long, device=device)
    else:
        indices = indices.to(device)

    indices = indices.unique()

    for idx in indices:
        idx_val = idx.item()
        row = idx_val // patches_per_row
        col = idx_val % patches_per_row
        r_start = row * patch_size
        r_end = min(r_start + patch_size, H)
        c_start = col * patch_size
        c_end = min(c_start + patch_size, W)
        new_tensor[:, r_start:r_end, c_start:c_end] = mean_values

    del mean_values, indices
    return new_tensor


def process_hidden_states_to_vector(hidden_states_tuple, max_tokens_limit=512):
    """Convert tuple of hidden states to flattened vector with CPU pinning."""
    feature_dim = None
    max_tokens = 0
    
    for layer_hidden in hidden_states_tuple:
        if layer_hidden.dim() == 3:
            layer_hidden = layer_hidden.mean(dim=0)
            tokens, features = layer_hidden.shape
        elif layer_hidden.dim() == 2:
            tokens, features = layer_hidden.shape
        else:
            continue
        max_tokens = max(max_tokens, tokens)
        if feature_dim is None:
            feature_dim = features
        else:
            feature_dim = min(feature_dim, features)
    
    max_tokens = min(max_tokens, max_tokens_limit)
    
    normalized_layers = []
    for idx, layer_hidden in enumerate(hidden_states_tuple):
        tensor = layer_hidden.detach().float()
        
        if tensor.dim() == 3:
            tensor = tensor.mean(dim=0)
        
        if tensor.dim() == 2:
            tokens, features = tensor.shape
        elif tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
            tokens, features = tensor.shape
        else:
            continue
        
        if features > feature_dim:
            tensor = tensor[:, :feature_dim]
        
        if tokens != max_tokens:
            tensor = tensor.permute(1, 0).unsqueeze(0)
            tensor = F.interpolate(tensor, size=max_tokens, mode='linear', align_corners=False)
            tensor = tensor.squeeze(0).permute(1, 0)
        
        # Pin to CPU memory for faster transfers
        tensor = tensor.cpu().pin_memory() if torch.cuda.is_available() else tensor.cpu()
        normalized_layers.append(tensor)
        
        del layer_hidden, tensor
        if idx % 5 == 0:
            gc.collect()
    
    combined = torch.cat(normalized_layers, dim=0)
    del normalized_layers
    
    return combined


def load_image_for_internvl(image_file, input_size=448, max_num=12):
    """Load and preprocess image for InternVL"""
    image = load_image(image_file)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    
    del image, transform, images
    
    return pixel_values


def get_demos_internvl(args, model, tokenizer, image_folder, sample_file, num_demos=30, num_trials=30, 
                       mask_ratio=0.5, patch_size=14, input_size=448, max_num=6):
    """Get demonstration samples for NSS computation (Memory optimized for 60GB CPU + 24GB GPU)"""
    import torch
    import random
    
    random.seed(42)
    
    print(f"Loading {num_demos} demo samples from {sample_file}...")
    
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"Sample file not found: {sample_file}")
    
    data = json.load(open(sample_file, 'r'))
    print(f"Loaded {len(data)} samples from file")
    
    data_demos = random.sample(data, min(num_demos, len(data)))
    print(f"Selected {len(data_demos)} demos for NSS computation")
    
    inputs_images = []
    all_inputs_pos = []
    all_inputs_neg = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for demo_idx, demo_data in enumerate(data_demos):
        if demo_idx > 0 and demo_idx % 3 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            images_path = []
            
            if 'images' in demo_data:
                images_list = demo_data['images']
                if isinstance(images_list, list):
                    for img in images_list:
                        path_img = os.path.join(image_folder, demo_data.get("id", ""), img)
                        if not os.path.exists(path_img):
                            path_img = os.path.join(image_folder, img)
                        if os.path.exists(path_img):
                            images_path.append(path_img)
                elif isinstance(images_list, str):
                    images_path.append(os.path.join(image_folder, images_list))
            elif 'image_id' in demo_data:
                images_path.append(os.path.join(image_folder, demo_data['image_id']))
            
            if not images_path:
                print(f"Warning: Demo {demo_idx} missing valid image paths, skipping")
                continue
            
            print(f"Processing demo {demo_idx}: {len(images_path)} images")
            
            input_images = []
            
            for img_idx, image_path in enumerate(images_path):
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}, skipping")
                    continue
                
                pixel_values = load_image_for_internvl(image_path, input_size=input_size, max_num=max_num).to(device)
                
                if pixel_values is None or pixel_values.numel() == 0:
                    print(f"Warning: Failed to load image {img_idx} for demo {demo_idx}, skipping")
                    continue
                
                num_patches = pixel_values.shape[0]
                
                masked_trials = []
                MASKING_BATCH_SIZE = 3
                
                for batch_start in range(0, num_trials, MASKING_BATCH_SIZE):
                    batch_end = min(batch_start + MASKING_BATCH_SIZE, num_trials)
                    batch_trials_gpu = []
                    
                    for trial_idx in range(batch_start, batch_end):
                        num_to_mask = max(1, int(num_patches * mask_ratio))
                        mask_indices = random.sample(range(num_patches), num_to_mask)
                        
                        masked_pixel_values = pixel_values.clone()
                        for patch_idx in mask_indices:
                            masked_pixel_values[patch_idx] = mask_patches(pixel_values[patch_idx], indices=[0], patch_size=patch_size)
                        
                        batch_trials_gpu.append(masked_pixel_values)
                    
                    # Move to CPU and pin memory immediately
                    batch_trials_cpu = [t.cpu().pin_memory() if torch.cuda.is_available() else t.cpu() for t in batch_trials_gpu]
                    masked_trials.extend(batch_trials_cpu)
                    
                    del batch_trials_gpu, batch_trials_cpu
                    torch.cuda.empty_cache()
                
                input_images.append([masked_trials, pixel_values.cpu().pin_memory() if torch.cuda.is_available() else pixel_values.cpu()])
                
                del pixel_values, masked_trials
                torch.cuda.empty_cache()
                
                if (img_idx + 1) % 2 == 0:
                    gc.collect()
            
            inputs_images.append(input_images)
            del input_images
            torch.cuda.empty_cache()
            
            # ===== TEXT PROMPTS for TEXTUAL NSS =====
            question = demo_data.get("question", "")
            
            choices = []
            if 'answers_choice' in demo_data:
                choices = demo_data['answers_choice']
                if not isinstance(choices, list):
                    choices = [choices]
            else:
                for idx in range(5):
                    choice_key = f"answer_choice_{idx}"
                    if choice_key in demo_data:
                        choices.append(demo_data[choice_key])
            
            if len(choices) < 2:
                print(f"Warning: Demo {demo_idx} has less than 2 choices, skipping text prompts")
                torch.cuda.empty_cache()
                continue
            
            correct_id = demo_data.get("correct_id", demo_data.get("correct_answer_id", 0))
            hallucinated_id = demo_data.get("outline_id", demo_data.get("outline_answer_id", 1))
            
            if 'value' in demo_data and 'h_value' in demo_data:
                correct_answer = demo_data['value']
                hallucinated_answer = demo_data['h_value']
            else:
                if correct_id < 0 or correct_id >= len(choices):
                    print(f"Warning: Invalid correct_answer_id {correct_id} for demo {demo_idx}")
                    continue
                
                if hallucinated_id < 0 or hallucinated_id >= len(choices):
                    print(f"Warning: Invalid outline_answer_id {hallucinated_id} for demo {demo_idx}")
                    continue
                
                correct_answer = choices[correct_id]
                hallucinated_answer = choices[hallucinated_id]
            
            if len(choices) > 0:
                options_text = "\n".join([f"{i}: {choice}" for i, choice in enumerate(choices)])
            else:
                options_text = ""
            
            image_tokens = '<image>\n' * len(images_path)
            
            if options_text:
                text_content_pos = f"{question}\nInstruction: Choose the **CORRECT NUMBER ID** answer.\n\nAnswer options:\n{options_text}\n\nANSWER NUMBER ID: {correct_answer}"
                text_content_neg = f"{question}\nInstruction: Choose the **CORRECT NUMBER ID** answer.\n\nAnswer options:\n{options_text}\n\nANSWER NUMBER ID: {hallucinated_answer}"
            else:
                text_content_pos = f"{question}\nANSWER NUMBER ID: {correct_answer}"
                text_content_neg = f"{question}\nANSWER NUMBER ID: {hallucinated_answer}"
            
            prompt_pos = f"{image_tokens}{text_content_pos}"
            prompt_neg = f"{image_tokens}{text_content_neg}"
            
            inputs_pos = tokenizer(prompt_pos, return_tensors='pt', padding=True).to(device)
            inputs_neg = tokenizer(prompt_neg, return_tensors='pt', padding=True).to(device)
            
            all_inputs_pos.append(inputs_pos)
            all_inputs_neg.append(inputs_neg)
            
            if (demo_idx + 1) % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if (demo_idx + 1) % 10 == 0:
                print(f"  Processed {demo_idx + 1}/{len(data_demos)} demos")
                
        except Exception as e:
            print(f"Error processing demo {demo_idx}: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()
            continue
    
    if len(inputs_images) == 0:
        raise ValueError("No valid demo samples were loaded!")
    
    print(f"✓ Successfully processed {len(inputs_images)} demo samples")
    
    if len(all_inputs_pos) > 0 and len(all_inputs_neg) > 0:
        input_ids = list(zip(all_inputs_neg, all_inputs_pos))
    else:
        print("Warning: No valid text prompts for textual NSS")
        input_ids = None
    
    del data_demos, data, all_inputs_pos, all_inputs_neg
    torch.cuda.empty_cache()
    gc.collect()
    
    return inputs_images, input_ids


def obtain_visual_nss_internvl(model, demo_inputs, rank=1):
    """
    Obtain visual NSS direction for InternVL
    KEY FIX: Keep vision model on GPU in bfloat16 (required for flash attention)
    Move hidden states to CPU immediately after extraction
    """
    from utils.pca import PCA
    
    # ===== CRITICAL SETTINGS FOR 24GB GPU =====
    USE_GPU_VISION = True  # KEEP VISION ON GPU in bfloat16!
    USE_MIXED_PRECISION = True  # Use bfloat16 for vision model
    MAX_TOKENS_PER_LAYER = 256
    BATCH_SIZE_MASKED_TRIALS = 2
    
    print("Computing visual NSS direction (GPU bfloat16 MODE)...")
    print(f"  ✓ Vision model STAYS on GPU in bfloat16 (required for flash attention)")
    print(f"  ✓ Hidden states moved to CPU immediately after extraction")
    print(f"  Optimizations: gpu_vision={USE_GPU_VISION}, mixed_precision={USE_MIXED_PRECISION}")
    
    vision_model = get_vision_model(model)
    original_device = next(vision_model.parameters()).device
    device = original_device
    
    # Set to eval and enable output_hidden_states
    vision_model.eval()
    if hasattr(vision_model, 'output_hidden_states'):
        vision_model.output_hidden_states = True
    
    n_vision_layers = None
    all_demos_diffs = []
    
    # Use mixed precision context for vision model
    use_amp = USE_MIXED_PRECISION and device.type == 'cuda'
    
    with torch.no_grad():
        autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
        
        for demo_idx, demo_images in enumerate(tqdm(demo_inputs, desc="Processing demos for visual NSS")):
            if demo_idx > 0 and demo_idx % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            try:
                image_diffs = []
                
                if len(demo_images) == 0:
                    print(f"  ⚠️  Demo {demo_idx}: No images to process, skipping")
                    continue
                
                print(f"  Demo {demo_idx}: Processing {len(demo_images)} images...", end="")
                
                for img_idx, (masked_trials, original) in enumerate(demo_images):
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    original_gpu = original.to(device).float()
                    
                    with autocast_ctx:
                        try:
                            original_outputs = vision_model(original_gpu, return_dict=True, output_hidden_states=True)
                        except TypeError:
                            original_outputs = vision_model(original_gpu, return_dict=True)
                        
                        if hasattr(original_outputs, 'hidden_states') and original_outputs.hidden_states is not None:
                            h_original = original_outputs.hidden_states
                            if len(h_original) > 1:
                                h_original = h_original[1:]
                        elif hasattr(original_outputs, 'last_hidden_state'):
                            h_original = (original_outputs.last_hidden_state,)
                        else:
                            print(f"Warning: Could not extract hidden states for original image {img_idx}")
                            continue
                        
                        # CRITICAL: Move to CPU immediately while inside autocast context
                        h_original = tuple(h.detach().float().cpu().pin_memory() if torch.cuda.is_available() else h.detach().float().cpu() for h in h_original)
                        del original_outputs, original_gpu
                    
                    torch.cuda.empty_cache()
                    
                    h_masked_list = []
                    num_trials = len(masked_trials)
                    
                    for batch_start in range(0, num_trials, BATCH_SIZE_MASKED_TRIALS):
                        batch_end = min(batch_start + BATCH_SIZE_MASKED_TRIALS, num_trials)
                        batch_trials = masked_trials[batch_start:batch_end]
                        
                        for trial_idx, masked_pixel_values in enumerate(batch_trials):
                            masked_gpu = masked_pixel_values.to(device).float()
                            
                            if not torch.isfinite(masked_gpu).all():
                                print(f"Warning: Non-finite values in masked trial {batch_start + trial_idx}, skipping")
                                del masked_gpu
                                continue
                            
                            try:
                                with autocast_ctx:
                                    masked_outputs = vision_model(masked_gpu, return_dict=True, output_hidden_states=True)
                            except TypeError:
                                with autocast_ctx:
                                    masked_outputs = vision_model(masked_gpu, return_dict=True)
                            except Exception as e:
                                print(f"\nError processing masked trial {batch_start + trial_idx}: {e}")
                                del masked_gpu
                                continue
                            
                            if hasattr(masked_outputs, 'hidden_states') and masked_outputs.hidden_states is not None:
                                h_masked = masked_outputs.hidden_states
                                if len(h_masked) > 1:
                                    h_masked = h_masked[1:]
                            elif hasattr(masked_outputs, 'last_hidden_state'):
                                h_masked = (masked_outputs.last_hidden_state,)
                            else:
                                print(f"\nWarning: Could not extract hidden states for masked trial {batch_start + trial_idx}")
                                del masked_outputs, masked_gpu
                                continue
                            
                            # CRITICAL: Move to CPU immediately
                            h_masked = tuple(h.detach().float().cpu().pin_memory() if torch.cuda.is_available() else h.detach().float().cpu() for h in h_masked)
                            h_masked_list.append(h_masked)
                            
                            del masked_outputs, masked_gpu, h_masked
                            torch.cuda.empty_cache()
                        
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    if len(h_masked_list) == 0:
                        print(f"Warning: No valid masked trials for image {img_idx}")
                        continue
                    
                    # Average on CPU
                    h_masked_avg = tuple(
                        torch.stack([h[i] for h in h_masked_list]).mean(dim=0).float().cpu()
                        for i in range(len(h_masked_list[0]))
                    )
                    
                    del h_masked_list
                    torch.cuda.empty_cache()
                    
                    if n_vision_layers is None:
                        n_vision_layers = len(h_original)
                        print(f"✓ Detected {n_vision_layers} vision layers from hidden_states")
                    
                    if len(h_masked_avg) != n_vision_layers or len(h_original) != n_vision_layers:
                        print(f"Warning: Layer count mismatch")
                        continue
                    
                    # Convert to vectors on CPU
                    vec_masked = process_hidden_states_to_vector(h_masked_avg, max_tokens_limit=MAX_TOKENS_PER_LAYER)
                    vec_original = process_hidden_states_to_vector(h_original, max_tokens_limit=MAX_TOKENS_PER_LAYER)
                    
                    diff = vec_masked - vec_original
                    image_diffs.append(diff)
                    
                    del vec_masked, vec_original, h_masked_avg, h_original, diff
                
                if len(image_diffs) > 0:
                    demo_avg_diff = torch.stack(image_diffs).mean(dim=0)
                    all_demos_diffs.append(demo_avg_diff)
                    print(f"  ✓ Demo {demo_idx}: processed successfully")
                    del demo_avg_diff
                else:
                    print(f"  ✗ Demo {demo_idx}: No valid images")
                
                del image_diffs
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"\n[ERROR] Demo {demo_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()
                gc.collect()
                continue
    
    if len(all_demos_diffs) == 0:
        raise ValueError("No valid demo samples could be processed for visual NSS!")
    
    if n_vision_layers is None:
        raise ValueError("Could not determine n_vision_layers from hidden_states!")
    
    print(f"✓ Processed {len(all_demos_diffs)} demos successfully")
    
    # Find minimum dimensions
    min_dim = min(d.shape[0] for d in all_demos_diffs)
    
    feature_dims = []
    for d in all_demos_diffs:
        if d.dim() == 2:
            feature_dims.append(d.shape[1])
        elif d.dim() == 3:
            feature_dims.append(d.shape[1] * d.shape[2])
    
    feature_dim = min(feature_dims) if feature_dims else 768
    
    print(f"  Resizing to min dimension: {min_dim} tokens, {feature_dim} features")
    
    resized_diffs = []
    for idx, diff in enumerate(all_demos_diffs):
        if diff.dim() == 3:
            diff = diff.view(diff.shape[0], -1)
        elif diff.dim() != 2:
            del diff
            continue
        
        current_tokens, current_features = diff.shape
        
        if current_tokens != min_dim or current_features != feature_dim:
            diff_reshaped = diff.unsqueeze(0).unsqueeze(0)
            del diff
            
            diff_resized = F.interpolate(diff_reshaped, size=(min_dim, feature_dim), mode='bilinear', align_corners=False)
            del diff_reshaped
            
            resized_tensor = diff_resized.squeeze(0).squeeze(0)
            del diff_resized
            resized_diffs.append(resized_tensor)
            del resized_tensor
        else:
            resized_diffs.append(diff.clone())
            del diff
        
        if (idx + 1) % 3 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    del all_demos_diffs, feature_dims
    torch.cuda.empty_cache()
    gc.collect()
    
    # Stack with chunking for large datasets
    if len(resized_diffs) > 10:
        chunk_size = 5
        fit_data_chunks = []
        
        for i in range(0, len(resized_diffs), chunk_size):
            chunk = resized_diffs[i:i+chunk_size]
            chunk_stacked = torch.stack(chunk).float()
            chunk_flattened = chunk_stacked.reshape(len(chunk), -1)
            fit_data_chunks.append(chunk_flattened)
            
            del chunk, chunk_stacked
            torch.cuda.empty_cache()
        
        fit_data = torch.cat(fit_data_chunks, dim=0)
        del fit_data_chunks
    else:
        fit_data = torch.stack(resized_diffs).float()
        n_demos = fit_data.shape[0]
        fit_data = fit_data.reshape(n_demos, -1)
    
    del resized_diffs
    torch.cuda.empty_cache()
    gc.collect()
    
    # Normalize with numerical stability
    fit_data_normalized = fit_data.float()
    reading_direction_flat = fit_data.mean(0)
    
    if fit_data_normalized.shape[0] > 1 and fit_data_normalized.std() > 1e-6:
        mean_val = fit_data_normalized.mean(dim=0, keepdim=True)
        fit_data_normalized.sub_(mean_val)
        del mean_val
        
        std_val = fit_data_normalized.std(dim=0, keepdim=True)
        std_val = torch.clamp(std_val, min=1e-8)
        fit_data_normalized.div_(std_val)
        del std_val
    
    if not torch.isfinite(fit_data_normalized).all():
        fit_data_normalized = torch.nan_to_num(fit_data_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
    
    del fit_data
    torch.cuda.empty_cache()
    
    # PCA
    from utils.pca import PCA
    pca = PCA(n_components=rank).fit(fit_data_normalized)
    
    direction_flat = (pca.components_.sum(dim=0, keepdim=True) + pca.mean_).squeeze()
    
    del pca, fit_data_normalized
    torch.cuda.empty_cache()
    gc.collect()
    
    # Reshape
    direction_2d = direction_flat.view(min_dim, feature_dim)
    reading_direction_2d = reading_direction_flat.view(min_dim, feature_dim)
    
    tokens_per_layer = min_dim // n_vision_layers
    
    if tokens_per_layer == 0:
        raise ValueError(f"tokens_per_layer is 0: min_dim ({min_dim}) < n_vision_layers ({n_vision_layers})")
    
    if min_dim % n_vision_layers != 0:
        target_dim = tokens_per_layer * n_vision_layers
        
        direction_2d = direction_2d.unsqueeze(0).unsqueeze(0)
        direction_2d = F.interpolate(direction_2d, size=(target_dim, feature_dim), mode='nearest')
        direction_2d = direction_2d.squeeze(0).squeeze(0)
        
        reading_direction_2d = reading_direction_2d.unsqueeze(0).unsqueeze(0)
        reading_direction_2d = F.interpolate(reading_direction_2d, size=(target_dim, feature_dim), mode='nearest')
        reading_direction_2d = reading_direction_2d.squeeze(0).squeeze(0)
        
        min_dim = target_dim
        tokens_per_layer = min_dim // n_vision_layers
    
    direction_final = direction_2d.view(n_vision_layers, tokens_per_layer, feature_dim)
    reading_direction_final = reading_direction_2d.view(n_vision_layers, tokens_per_layer, feature_dim)
    
    del direction_2d, reading_direction_2d, direction_flat, reading_direction_flat
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"✓ NSS direction computed for {n_vision_layers} layers")
    print(f"  Final shape: {direction_final.shape} [n_layers, n_tokens, hidden_dim]")
    print(f"  Direction norm: {torch.norm(direction_final).item():.6f}")
    
    return direction_final, reading_direction_final


def obtain_textual_nss_internvl(model, demo_inputs, rank=1):
    """Obtain textual NSS direction for InternVL (hook-based approach)"""
    from utils.pca import PCA
    
    print(f"Computing textual NSS from {len(demo_inputs)} demos...")
    
    language_model = get_language_model(model)
    device = next(model.parameters()).device
    
    # Get layers
    if hasattr(language_model, 'layers'):
        layers = language_model.layers
    elif hasattr(language_model, 'model') and hasattr(language_model.model, 'layers'):
        layers = language_model.model.layers
    elif hasattr(language_model, 'transformer') and hasattr(language_model.transformer, 'h'):
        layers = language_model.transformer.h
    else:
        raise ValueError("Cannot hook into language model layers")
    
    n_layers = len(layers)
    print(f"Found {n_layers} language model layers")
    
    # Register hooks
    captured_hidden_states = {}
    hook_handles = []
    
    def make_hook_fn(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured_hidden_states[layer_idx] = output[0]
            else:
                captured_hidden_states[layer_idx] = output
        return hook
    
    for idx, layer in enumerate(layers):
        handle = layer.register_forward_hook(make_hook_fn(idx))
        hook_handles.append(handle)
    
    print(f"Registered hooks on {len(hook_handles)} layers")
    
    all_demos_diffs = []
    embed_tokens = language_model.get_input_embeddings()
    
    with torch.no_grad():
        for demo_idx, (inputs_neg, inputs_pos) in enumerate(tqdm(demo_inputs, desc="Processing text demos")):
            try:
                input_ids_neg = inputs_neg['input_ids'].to(device)
                input_ids_pos = inputs_pos['input_ids'].to(device)
                
                inputs_embeds_neg = embed_tokens(input_ids_neg)
                inputs_embeds_pos = embed_tokens(input_ids_pos)
                
                del input_ids_neg, input_ids_pos
                
                attention_mask_neg = inputs_neg.get('attention_mask', None)
                attention_mask_pos = inputs_pos.get('attention_mask', None)
                if attention_mask_neg is not None:
                    attention_mask_neg = attention_mask_neg.to(device)
                if attention_mask_pos is not None:
                    attention_mask_pos = attention_mask_pos.to(device)
                
                use_cuda = device.type == "cuda"
                ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda else nullcontext()
                
                with ctx:
                    captured_hidden_states.clear()
                    
                    kwargs_neg = {'inputs_embeds': inputs_embeds_neg, 'use_cache': False, 'output_hidden_states': False}
                    if attention_mask_neg is not None:
                        kwargs_neg['attention_mask'] = attention_mask_neg
                    
                    _ = language_model(**kwargs_neg)
                    
                    hidden_neg_all_layers = []
                    for layer_idx in range(n_layers):
                        if layer_idx in captured_hidden_states:
                            h = captured_hidden_states[layer_idx][:, -1, :].float().cpu().pin_memory() if torch.cuda.is_available() else captured_hidden_states[layer_idx][:, -1, :].float().cpu()
                            hidden_neg_all_layers.append(h)
                    
                    captured_hidden_states.clear()
                    
                    kwargs_pos = {'inputs_embeds': inputs_embeds_pos, 'use_cache': False, 'output_hidden_states': False}
                    if attention_mask_pos is not None:
                        kwargs_pos['attention_mask'] = attention_mask_pos
                    
                    _ = language_model(**kwargs_pos)
                    
                    hidden_pos_all_layers = []
                    for layer_idx in range(n_layers):
                        if layer_idx in captured_hidden_states:
                            h = captured_hidden_states[layer_idx][:, -1, :].float().cpu().pin_memory() if torch.cuda.is_available() else captured_hidden_states[layer_idx][:, -1, :].float().cpu()
                            hidden_pos_all_layers.append(h)
                    
                    captured_hidden_states.clear()
                
                if len(hidden_neg_all_layers) != n_layers or len(hidden_pos_all_layers) != n_layers:
                    print(f"Warning: Demo {demo_idx} incomplete")
                    continue
                
                layer_diffs = []
                for h_neg, h_pos in zip(hidden_neg_all_layers, hidden_pos_all_layers):
                    diff = (h_pos - h_neg).squeeze()
                    if not torch.isfinite(diff).all():
                        diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
                    layer_diffs.append(diff)
                
                demo_diff = torch.stack(layer_diffs)
                all_demos_diffs.append(demo_diff)
                
                del layer_diffs, hidden_neg_all_layers, hidden_pos_all_layers, demo_diff
                
                if (demo_idx + 1) % 2 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except Exception as e:
                print(f"Error processing demo {demo_idx}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                continue
    
    for handle in hook_handles:
        handle.remove()
    
    del hook_handles, captured_hidden_states
    torch.cuda.empty_cache()
    gc.collect()
    
    if len(all_demos_diffs) == 0:
        raise ValueError("Could not process any demo samples for textual NSS!")
    
    print(f"✓ Processed {len(all_demos_diffs)} demos successfully")
    
    all_diffs_stacked = torch.stack(all_demos_diffs)
    n_demos = all_diffs_stacked.shape[0]
    hidden_dim = all_diffs_stacked.shape[2]
    
    print(f"  All diffs shape: {all_diffs_stacked.shape}")
    
    del all_demos_diffs
    
    fit_data = all_diffs_stacked.reshape(n_demos, -1)
    del all_diffs_stacked
    
    fit_data_normalized = fit_data.float()
    if fit_data_normalized.std() > 1e-6:
        fit_data_normalized = (fit_data_normalized - fit_data_normalized.mean(dim=0)) / (fit_data_normalized.std(dim=0) + 1e-8)
    
    pca = PCA(n_components=rank).fit(fit_data_normalized)
    
    direction_flat = (pca.components_.sum(dim=0, keepdim=True) + pca.mean_).squeeze()
    reading_direction_flat = fit_data.mean(dim=0)
    
    del pca, fit_data, fit_data_normalized
    
    direction_per_layer = direction_flat.view(n_layers, hidden_dim)
    reading_direction_per_layer = reading_direction_flat.view(n_layers, hidden_dim)
    
    del direction_flat, reading_direction_flat
    
    direction = direction_per_layer.unsqueeze(1)
    reading_direction = reading_direction_per_layer.unsqueeze(1)
    
    del direction_per_layer, reading_direction_per_layer
    
    print(f"✓ Textual NSS direction computed for {n_layers} layers")
    print(f"  Final shape: {direction.shape}")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return direction, reading_direction
