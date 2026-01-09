
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import os
import json
import random
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from utils.pca import PCA
from torchvision import transforms


random.seed(42)

_QWEN_IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),  # Converts to [C, H, W] and scales to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Import helper functions from utils module if needed
try:
    from .hooked_forward import get_qwen_visual_hidden_states_from_tensor
except ImportError:
    print("Warning: Could not import get_qwen_visual_hidden_states_from_tensor")
    get_qwen_visual_hidden_states_from_tensor = None


def process_image_qwen(image_path, device=None):
    """
    Process a PIL Image and return a torch.Tensor for Qwen models.
    OPTIMIZED: Uses cached transform pipeline (no repeated object creation).
    CUDA-enabled: Processes directly on GPU for faster masking operations.
    
    Args:
        image_path: Path to image file
        device: Target device (default: cuda if available)
    
    Returns:
        pixel_values tensor, shape (C, H, W)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    image = Image.open(image_path).convert("RGB")
    
    # Resize to reasonable size matching Qwen/NSS pipeline (200k-600k pixels)
    # This drastically reduces patch count for faster masking
    w, h = image.size
    current_pixels = w * h
    
    # Target: ~200k-600k pixels (similar to min_pixels/max_pixels in Qwen)
    # Typical size: ~448x448 = 200k pixels, up to ~775x775 = 600k pixels
    target_pixels = 400000  # 400k pixels (balanced, ~632x632)
    
    if current_pixels > target_pixels * 1.2 or current_pixels < target_pixels * 0.5:
        # Calculate scale factor to achieve target pixels
        scale = (target_pixels / current_pixels) ** 0.5
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Ensure dimensions are divisible by patch_size (14) for clean patching
        new_w = (new_w // 14) * 14
        new_h = (new_h // 14) * 14
        
        # Minimum size to avoid too small images
        new_w = max(new_w, 224)
        new_h = max(new_h, 224)
        
        image = image.resize((new_w, new_h), Image.BILINEAR)
    
    # Use cached transform pipeline (FAST!) and move to device immediately
    pixel_values = _QWEN_IMAGE_TRANSFORM(image).to(device)
    
    return pixel_values
        

def mask_patches(tensor, indices, patch_size=14):
    """
    Replace the specified patches with the per-channel mean.
    """
    original_shape = tensor.shape
    device = tensor.device

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)          # (H, W) -> (1, H, W)
    elif tensor.dim() == 4:
        tensor = tensor.squeeze(0)            # (B, C, H, W) -> (C, H, W)
    elif tensor.dim() != 3:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got shape {original_shape}")
    
    C, H, W = tensor.shape
    new_tensor = tensor.clone()

    mean_values = tensor.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)

    patches_per_row = W // patch_size if patch_size > 0 else 0
    if patches_per_row == 0:
        return new_tensor

    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(list(indices), device=device)
    else:
        indices = indices.to(device)

    indices = indices.unique()

    for idx in indices:
        idx = int(idx.item())
        row = idx // patches_per_row
        col = idx % patches_per_row

        start_y = row * patch_size
        start_x = col * patch_size
        end_y = min(start_y + patch_size, H)
        end_x = min(start_x + patch_size, W)

        new_tensor[:, start_y:end_y, start_x:end_x] = mean_values

    return new_tensor


def average_tuples(tuples):
    """Average multiple tuples of tensors element-wise."""
    if not tuples:
        raise ValueError("The input list of tuples is empty.")
    n = len(tuples[0])
    if not all(len(t) == n for t in tuples):
        raise ValueError("All tuples must have the same length.")
    averaged_tensors = []
    for i in range(n):
        tensors_at_i = torch.stack([t[i].detach().cpu() for t in tuples])
        averaged_tensor = tensors_at_i.mean(dim=0)
        averaged_tensors.append(averaged_tensor)
    return tuple(averaged_tensors)


def process_hidden_states_to_vector(hidden_states_tuple, max_tokens_limit=1024):
    """
    Convert tuple of hidden states (per layer) to a single flattened vector.
    Handles layers with different token counts via interpolation.
    PRESERVES all spatial tokens - does NOT mean pool.
    
    OPTIMIZED: Handles dimension mismatches efficiently to prevent OOM.
    
    Args:
        hidden_states_tuple: Tuple of tensors, each shape (batch, tokens, features)
        max_tokens_limit: Maximum number of tokens to keep (to prevent OOM)
    
    Returns:
        Flattened vector (n_layers * tokens, features)
    """
    feature_dim = None
    max_tokens = 0
    
    for layer_hidden in hidden_states_tuple:
        if layer_hidden.dim() == 3:
            _, tokens, features = layer_hidden.shape
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
        tensor = layer_hidden.detach().cpu()
        
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        batch, tokens, features = tensor.shape
        
        if features > feature_dim:
            tensor = tensor[:, :, :feature_dim]
        
        if tokens != max_tokens:
            tensor = tensor.permute(0, 2, 1)
            tensor = F.interpolate(
                tensor,
                size=max_tokens,
                mode='linear',
                align_corners=False
            )
            tensor = tensor.permute(0, 2, 1)
        
        normalized_layers.append(tensor.squeeze(0))
        
        del tensor
    
    combined = torch.cat(normalized_layers, dim=0)
    
    del normalized_layers
    
    return combined


def get_visual_hiddenstates_single_image_qwen(model, image_data, processor):
    """
    Get hidden states for ONE image (with multiple masked trials).
    Adapted for Qwen models.
    
    Args:
        image_data: 
            - If list: [[masked_trial1, masked_trial2, ...], original_tensor]
            - If not: original_tensor
    
    Returns:
        Tuple of (masked_hidden_states, original_hidden_states)
        Each is a tuple of tensors WITHOUT embedding layer
    """
    with torch.no_grad():
        vision_model = model.visual
        model.visual.output_hidden_states = True
        
        device = next(model.visual.parameters()).device

        if isinstance(image_data, list):
            masked_trials = image_data[0]  # List of masked tensors
            original_tensor = image_data[1]
            
            # Get hidden states for all masked trials
            h_masked_list = []
            for trial_idx, masked_tensor in enumerate(masked_trials):
                # Move from CPU to CUDA (tensors are stored on CPU in get_demos_qwen)
                pixel_values = masked_tensor.unsqueeze(0).to(device)
                
                if get_qwen_visual_hidden_states_from_tensor is not None:
                    h_ = get_qwen_visual_hidden_states_from_tensor(
                        model, processor, pixel_values, return_dict=False
                    )
                else:
                    H, W = pixel_values.shape[-2], pixel_values.shape[-1]
                    patch_size_model = 14
                    gh = H // patch_size_model
                    gw = W // patch_size_model
                    grid_thw = torch.tensor([[1, gh, gw]], device=pixel_values.device)
                    
                    h_ = vision_model(
                        pixel_values=pixel_values,
                        grid_thw=grid_thw,
                        output_hidden_states=True,
                        return_dict=True
                    ).hidden_states
                
                if len(h_) > 1:
                    h_ = h_[1:]
                
                h_masked_list.append(h_)
                
                del pixel_values
            
            h_masked = average_tuples(h_masked_list)
            del h_masked_list
        else:
            original_tensor = image_data
            h_masked = None
        
        pixel_values = original_tensor.unsqueeze(0).to(device)
        
        if get_qwen_visual_hidden_states_from_tensor is not None:
            h_original = get_qwen_visual_hidden_states_from_tensor(
                model, processor, pixel_values, return_dict=False
            )
        else:
            H, W = pixel_values.shape[-2], pixel_values.shape[-1]
            patch_size_model = 14
            gh = H // patch_size_model
            gw = W // patch_size_model
            grid_thw = torch.tensor([[1, gh, gw]], device=pixel_values.device)
            
            h_original = vision_model(
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                output_hidden_states=True,
                return_dict=True
            ).hidden_states
        
        if len(h_original) > 1:
            h_original = h_original[1:]
        
        model.visual.output_hidden_states = False
        
        if h_masked is None:
            return (h_original, h_original)
        
        return (h_masked, h_original)


def get_demos_qwen(args, processor, samples_file, image_folder, cache_dir="./../cache_dir", num_demos=10, num_trials=30, mask_ratio=0.5, patch_size=14, min_pixels=200704, max_pixels=602112):
    """
    Get demo samples for NSS computation - Qwen compatible version.
    
    UPDATED: Thêm masking logic và text prompts.
    
    Returns:
        inputs_images: List of masked/original image tensors for VISUAL NSS
        input_ids: List of (negative, positive) text prompts for TEXTUAL NSS
    """
    data = json.load(open(samples_file, 'r'))
    data_demos = random.sample(data, min(num_demos, len(data)))
    print(f"Selected {len(data_demos)} demos for NSS computation")
    
    inputs_images = []
    
    all_inputs_pos = []
    all_inputs_neg = []
    
    for demo_idx, demo_data in enumerate(data_demos):
        images_path = []
        
        for img in demo_data['images']:
            path_img = os.path.join(image_folder, demo_data["id"], img)
            if not os.path.exists(path_img):
                path_img = os.path.join(image_folder, img)
                if not os.path.exists(path_img):
                    print(f"Warning: Image not found: {path_img}")
            images_path.append(path_img)
        
        input_images = []
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for img_idx, image_path in enumerate(images_path):
            image_tensor = process_image_qwen(image_path, device=device)
            
            # Calculate number of patches once (before loop)
            H, W = image_tensor.shape[-2:]
            n_h = H // patch_size
            n_w = W // patch_size
            num_patches = n_h * n_w
            num_mask = int(mask_ratio * num_patches)
            
            # Create masked versions (multiple trials) - all on GPU first
            image_tensor_cd_all_trials_gpu = []
            for t in range(num_trials):
                # Fast random mask indices using torch.randint
                mask_index = torch.randint(0, num_patches, (num_mask,), device=device)
                
                # Apply masking (stays on GPU)
                image_tensor_cd = mask_patches(image_tensor, mask_index, patch_size=patch_size)
                image_tensor_cd_all_trials_gpu.append(image_tensor_cd)
            
            image_tensor_cd_all_trials = [t.cpu() for t in image_tensor_cd_all_trials_gpu]
            
            input_images.append([image_tensor_cd_all_trials, image_tensor.cpu()])
            
            del image_tensor_cd_all_trials_gpu, image_tensor_cd_all_trials, image_tensor
        
        inputs_images.append(input_images)
        del input_images
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        text_content_pos = demo_data["question"] + "Instruction: Choose the **CORRECT NUMBER ID** answer for the question below, you must response with the **NUMBER ID** answer, don't include text.\n" + "ANSWER NUMBER ID:" + demo_data["value"]
        text_content_neg = demo_data["question"] + "Instruction: Choose the **CORRECT NUMBER ID** answer for the question below, you must response with the **NUMBER ID** answer, don't include text.\n" + "ANSWER NUMBER ID:" + demo_data["h_value"]
        
        images_input = [{"type": "image", "image": str(image), "max_pixels": args.max_pixels} for image in images_path]

        messages_pos_text = [
            {
                "role": "system",
                "content": "You are a helpful assistant tasked with selecting the CORRECT NUMBER ID based on the content of the provided images."
            },
            {
                "role": "user",
                "content": [
                    *images_input,
                    {"type": "text", "text": text_content_pos}
                ]
            }
        ]
        
        messages_neg_text = [
            {
                "role": "system",
                "content": "You are a helpful assistant tasked with selecting the CORRECT NUMBER ID based on the content of the provided images."
            },
            {
                "role": "user", 
                "content": [
                    *images_input,
                    {"type": "text", "text": text_content_neg}
                ]
            }
        ]
        
        try:
            from qwen_vl_utils import process_vision_info
            
            text_pos = processor.apply_chat_template(
                messages_pos_text, tokenize=False, add_generation_prompt=False
            )
            image_inputs_pos, _ = process_vision_info(messages_pos_text)
            
            inputs_pos = processor(
                text=[text_pos],
                images=image_inputs_pos,
                padding=True,
                return_tensors="pt"
            )
            
            text_neg = processor.apply_chat_template(
                messages_neg_text, tokenize=False, add_generation_prompt=False
            )
            image_inputs_neg, _ = process_vision_info(messages_neg_text)
            
            inputs_neg = processor(
                text=[text_neg],
                images=image_inputs_neg,
                padding=True,
                return_tensors="pt"
            )
            
            del image_inputs_pos, image_inputs_neg
            
        except Exception as e:
            print(f"Warning: Error processing demo {demo_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        all_inputs_pos.append(inputs_pos)
        all_inputs_neg.append(inputs_neg)
        
        if (demo_idx + 1) % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if len(inputs_images) == 0:
        raise ValueError("No valid demo samples could be processed!")
    
    print(f"✓ Successfully processed {len(inputs_images)} demo samples")
    print(f"  - Masked images for VISUAL NSS: {len(inputs_images)} demos")
    print(f"  - Text prompts for TEXTUAL NSS: {len(all_inputs_pos)} demos")
    
    if len(all_inputs_pos) > 0 and len(all_inputs_neg) > 0:
        input_ids = [(all_inputs_neg[i], all_inputs_pos[i]) for i in range(len(all_inputs_pos))]
    else:
        print("⚠ Warning: No text prompts created, textual NSS will not be available")
        print("   Please ensure your data has 'value' and 'h_value' fields for each demo")
        input_ids = None
    
    return inputs_images, input_ids


def obtain_visual_nss_qwen(model, model_path, cache_dir, image_tensor, rank=1):
    """
    Compute visual NSS direction from demo samples - Qwen compatible version.
    
    UPDATED STRATEGY (matching utils.py):
    - Uses MASKED vs ORIGINAL images (not positive vs negative text)
    - PRESERVES all spatial tokens instead of mean pooling
    - Concatenates all layers: [L, S, H] -> [L*S, H]
    - Keeps full spatial information for better NSS computation
    - MEMORY OPTIMIZED: Processes demos in smaller batches to prevent OOM
    
    Args:
        model: Qwen2_5_VLForConditionalGeneration model or Qwen3VLForConditionalGeneration
        image_tensor: List of demos, each demo is list of images
            Structure: [
                # Demo 0
                [
                    [[masked_trial1, ...], original],  # Image 1
                    [[masked_trial1, ...], original],  # Image 2
                ],
                # Demo 1
                [...]
            ]
        rank: PCA rank (default 1)
    
    Returns:
        direction: NSS direction tensor [n_layers, 1, hidden_dim]
        reading_direction: Mean difference direction [n_layers, 1, hidden_dim]
    """

    
    print("Computing visual NSS direction (using MASKED vs ORIGINAL images)...")
    
    processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
    
    vision_model = model.visual
    
    n_vision_layers = None
    
    all_demos_diffs = []
    
    with torch.no_grad():
        for demo_idx, demo_images in enumerate(image_tensor):
            try:
                image_diffs = []
                for image_data in demo_images:
                    h_masked, h_original = get_visual_hiddenstates_single_image_qwen(
                        model, image_data, processor
                    )
                    
                    if n_vision_layers is None:
                        n_vision_layers = len(h_masked)
                        print(f"✓ Detected {n_vision_layers} vision layers from hidden_states_tuple")
                        print(f"  (embedding layer already removed)")
                    
                    if len(h_masked) != n_vision_layers or len(h_original) != n_vision_layers:
                        print(f"Warning: Layer count mismatch - h_masked:{len(h_masked)}, h_original:{len(h_original)}, expected:{n_vision_layers}")
                        continue
                    
                    # Convert to vectors (concatenate all layers)
                    vec_masked = process_hidden_states_to_vector(h_masked)
                    vec_original = process_hidden_states_to_vector(h_original)
                    
                    # Compute difference: masked - original (matching original NSS implementation)
                    diff = (vec_masked - vec_original).float().cpu()
                    # diff = (vec_original - vec_masked).float().cpu()
                    image_diffs.append(diff)
                    
                    del vec_masked, vec_original, h_masked, h_original
                
                demo_avg_diff = torch.stack(image_diffs).mean(dim=0)
                all_demos_diffs.append(demo_avg_diff)
                
                del image_diffs
                
                if demo_idx == 0:
                    print(f"  Demo {demo_idx}: diff shape = {demo_avg_diff.shape}")
                
                if (demo_idx + 1) % 2 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    
            except Exception as e:
                print(f"Error processing demo {demo_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if len(all_demos_diffs) == 0:
        raise ValueError("No valid demo samples could be processed for visual NSS!")
    
    if n_vision_layers is None:
        raise ValueError("Could not determine n_vision_layers from hidden_states_tuple!")
    
    print(f"✓ Processed {len(all_demos_diffs)} demos successfully")
    print(f"✓ Using n_vision_layers = {n_vision_layers} (from hidden_states_tuple)")
    
    if len(all_demos_diffs) < 5:
        print(f"⚠ Warning: Only {len(all_demos_diffs)} demos processed. NSS quality may be reduced.")
        print("   Consider reducing num_trials or image resolution if OOM occurred.")
    
    min_dim = min(d.shape[0] for d in all_demos_diffs)
    feature_dim = all_demos_diffs[0].shape[-1]
    
    print(f"  Resizing to min dimension: {min_dim} tokens, {feature_dim} features")
    
    resized_diffs = []
    for diff in all_demos_diffs:
        if diff.shape[0] != min_dim:
            diff_reshaped = diff.unsqueeze(0).unsqueeze(0)  # [1, 1, tokens, features]
            diff_resized = F.interpolate(
                diff_reshaped,
                size=(min_dim, feature_dim),
                mode='bilinear',
                align_corners=False
            )
            resized_diffs.append(diff_resized.squeeze(0).squeeze(0))
        else:
            resized_diffs.append(diff)
    
    del all_demos_diffs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        fit_data = torch.stack(resized_diffs)
        n_demos = fit_data.shape[0]
        fit_data_flat = fit_data.reshape(n_demos, -1)
        print(f"  PCA input shape: {fit_data_flat.shape}")
    except Exception as e:
        print(f"ERROR stacking/flattening diffs: {e}")
        raise
    
    try:
        fit_data_normalized = fit_data_flat.float()
        if fit_data_normalized.std() > 1e-6:
            fit_data_normalized = (fit_data_normalized - fit_data_normalized.mean(dim=0)) / (fit_data_normalized.std(dim=0) + 1e-8)
        print(f"  Normalized data: mean={fit_data_normalized.mean():.6f}, std={fit_data_normalized.std():.6f}")
    except Exception as e:
        print(f"ERROR normalizing data: {e}")
        raise
    
    try:
        pca = PCA(n_components=rank).to(fit_data_flat.device).fit(fit_data_normalized)
        print(f"  PCA components shape: {pca.components_.shape}")
    except Exception as e:
        print(f"ERROR in PCA: {e}")
        raise
    
    del fit_data_normalized, resized_diffs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        direction_flat = (pca.components_.sum(dim=0, keepdim=True) + pca.mean_).squeeze()
        reading_direction_flat = fit_data_flat.mean(0)
        print(f"  Direction flat shape: {direction_flat.shape}, reading direction flat shape: {reading_direction_flat.shape}")
    except Exception as e:
        print(f"ERROR computing direction: {e}")
        raise
    
    try:
        direction_2d = direction_flat.view(min_dim, feature_dim)
        reading_direction_2d = reading_direction_flat.view(min_dim, feature_dim)
        print(f"  Direction 2D shape: {direction_2d.shape}")
    except Exception as e:
        print(f"ERROR reshaping to 2D - direction_flat: {direction_flat.shape}, min_dim: {min_dim}, feature_dim: {feature_dim}")
        print(f"  Total elements: {direction_flat.numel()}, expected: {min_dim * feature_dim}")
        raise
    
    # Reshape to match model layers: [tokens, features] -> [n_layers, tokens_per_layer, features]
    tokens_per_layer = min_dim // n_vision_layers
    print(f"  Tokens per layer: {tokens_per_layer} (min_dim={min_dim}, n_vision_layers={n_vision_layers})")
    
    # Validation: tokens_per_layer must be > 0
    if tokens_per_layer == 0:
        print(f"ERROR: tokens_per_layer is 0! min_dim={min_dim}, n_vision_layers={n_vision_layers}")
        print(f"  This means min_dim < n_vision_layers. Increase image resolution or reduce num_demos.")
        raise ValueError(f"tokens_per_layer is 0: min_dim ({min_dim}) < n_vision_layers ({n_vision_layers})")
    
    if min_dim % n_vision_layers != 0:
        # Resize to be divisible by n_layers
        target_dim = tokens_per_layer * n_vision_layers
        print(f"  Adjusting dimension from {min_dim} to {target_dim} for {n_vision_layers} layers")
        
        direction_2d = direction_2d.unsqueeze(0).unsqueeze(0)
        direction_2d = F.interpolate(
            direction_2d,
            size=(target_dim, feature_dim),
            mode='bilinear',
            align_corners=False
        )
        direction_2d = direction_2d.squeeze(0).squeeze(0)
        
        reading_direction_2d = reading_direction_2d.unsqueeze(0).unsqueeze(0)
        reading_direction_2d = F.interpolate(
            reading_direction_2d,
            size=(target_dim, feature_dim),
            mode='bilinear',
            align_corners=False
        )
        reading_direction_2d = reading_direction_2d.squeeze(0).squeeze(0)
        
        min_dim = target_dim
        tokens_per_layer = min_dim // n_vision_layers
    
    # Reshape: [total_tokens, features] -> [n_layers, tokens_per_layer, features]
    try:
        print(f"  Final reshape: {direction_2d.shape} -> [{n_vision_layers}, {tokens_per_layer}, {feature_dim}]")
        direction_layered = direction_2d.view(n_vision_layers, tokens_per_layer, feature_dim)
        reading_direction_layered = reading_direction_2d.view(n_vision_layers, tokens_per_layer, feature_dim)
    except Exception as e:
        print(f"ERROR in final reshape: {e}")
        print(f"  direction_2d shape: {direction_2d.shape}")
        print(f"  Expected: [{n_vision_layers}, {tokens_per_layer}, {feature_dim}]")
        print(f"  Total elements: direction_2d={direction_2d.numel()}, expected={n_vision_layers * tokens_per_layer * feature_dim}")
        raise
    
    # Free intermediate tensors
    del direction_2d, reading_direction_2d, direction_flat, reading_direction_flat
    del fit_data, fit_data_flat, pca
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Keep full spatial information like original NSS (no averaging across tokens)
    # Final format: [n_layers, tokens_per_layer, features]
    direction_final = direction_layered
    reading_direction_final = reading_direction_layered
    
    print(f"✓ NSS direction computed for {n_vision_layers} layers")
    print(f"  Final shape: {direction_final.shape} [n_layers, n_tokens, hidden_dim]")
    print(f"  Direction norm: {torch.norm(direction_final).item():.6f}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return direction_final, reading_direction_final


def obtain_textual_nss_qwen(model, demo_inputs, rank=1):
    """
    Compute textual NSS direction from demo samples - Qwen compatible version.
    
    UPDATED STRATEGY (matching utils.py):
    - Captures hidden states from ALL layers (not just last layer)
    - Computes diff for EACH layer separately
    - PCA on flattened [n_demos, n_layers * hidden_dim]
    - Reshapes back to [n_layers, hidden_dim] for layer-specific directions
    
    Args:
        model: Qwen2_5_VLForConditionalGeneration model
        demo_inputs: List of (negative, positive) input tuples from get_demos_qwen()
        rank: PCA rank (default 1)
    
    Returns:
        direction: NSS direction tensor [n_layers, 1, hidden_dim]
        reading_direction: Mean difference direction [n_layers, 1, hidden_dim]
    """
    from utils.pca import PCA
    
    print("Computing textual NSS direction (capturing ALL layers)...")
    
    device = next(model.parameters()).device
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        print(f"Found layers at model.model.layers: {len(layers)} layers")
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
        layers = model.language_model.layers
        print(f"Found layers at model.language_model.layers: {len(layers)} layers")
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
        print(f"Found layers at model.transformer.h: {len(layers)} layers")
    elif hasattr(model, 'layers'):
        layers = model.layers
        print(f"Found layers at model.layers: {len(layers)} layers")
    else:
        print("Warning: Could not find language model layers")
        print(f"Model type: {type(model)}")
        raise ValueError("Cannot hook into language model layers")
    
    n_layers = len(layers)
    
    # Register hooks on ALL layers to capture hidden states
    captured_hidden_states = {}
    hook_handles = []
    
    def make_hook_fn(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured_hidden_states[layer_idx] = output[0]
            else:
                captured_hidden_states[layer_idx] = output
        return hook
    
    # Register hook for each layer
    for idx, layer in enumerate(layers):
        handle = layer.register_forward_hook(make_hook_fn(idx))
        hook_handles.append(handle)
    
    print(f"Registered hooks on {len(hook_handles)} layers")
    
    all_demos_diffs = []
    
    with torch.no_grad():
        for demo_idx, (inputs_neg, inputs_pos) in enumerate(demo_inputs):
            try:
                # Move to device
                inputs_neg_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                    for k, v in inputs_neg.items()}
                inputs_pos_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                    for k, v in inputs_pos.items()}
                
                # with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                use_cuda = device.type == "cuda"
                ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda else nullcontext()
                with ctx:
                    # Forward pass for negative sample - capture all layers
                    captured_hidden_states.clear()
                    _ = model(**inputs_neg_device, use_cache=False, output_hidden_states=False)
                    
                    # Extract last token from each layer
                    hidden_neg_all_layers = []
                    for layer_idx in range(n_layers):
                        if layer_idx in captured_hidden_states:
                            h = captured_hidden_states[layer_idx][:, -1, :].float().cpu()
                            hidden_neg_all_layers.append(h)
                        else:
                            print(f"Warning: Layer {layer_idx} not captured for negative sample")
                    
                    # Forward pass for positive sample - capture all layers
                    captured_hidden_states.clear()
                    _ = model(**inputs_pos_device, use_cache=False, output_hidden_states=False)
                    
                    # Extract last token from each layer
                    hidden_pos_all_layers = []
                    for layer_idx in range(n_layers):
                        if layer_idx in captured_hidden_states:
                            h = captured_hidden_states[layer_idx][:, -1, :].float().cpu()
                            hidden_pos_all_layers.append(h)
                        else:
                            print(f"Warning: Layer {layer_idx} not captured for positive sample")
                
                # Compute difference for EACH layer
                if len(hidden_neg_all_layers) != n_layers or len(hidden_pos_all_layers) != n_layers:
                    print(f"Warning: Demo {demo_idx} incomplete - neg:{len(hidden_neg_all_layers)}, pos:{len(hidden_pos_all_layers)}, expected:{n_layers}")
                    continue
                
                layer_diffs = []
                for h_neg, h_pos in zip(hidden_neg_all_layers, hidden_pos_all_layers):
                    diff = (h_pos - h_neg).squeeze()  # Shape: [hidden_dim]
                    
                    # Validate
                    if not torch.isfinite(diff).all():
                        diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    layer_diffs.append(diff)
                
                # Stack: List of [hidden_dim] -> [n_layers, hidden_dim]
                demo_diff = torch.stack(layer_diffs)
                all_demos_diffs.append(demo_diff)
                
                del layer_diffs, hidden_neg_all_layers, hidden_pos_all_layers
                
                if demo_idx == 0:
                    print(f"  Demo {demo_idx}: diff shape = {demo_diff.shape}")
                
                if (demo_idx + 1) % 2 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                
            except Exception as e:
                print(f"Error processing demo {demo_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    for handle in hook_handles:
        handle.remove()
    
    if len(all_demos_diffs) == 0:
        raise ValueError("Could not process any demo samples for textual NSS!")
    
    print(f"✓ Processed {len(all_demos_diffs)} demos successfully")
    
    # Stack all demos: [n_demos, n_layers, hidden_dim]
    all_diffs_stacked = torch.stack(all_demos_diffs)
    n_demos = all_diffs_stacked.shape[0]
    hidden_dim = all_diffs_stacked.shape[2]
    
    print(f"  All diffs shape: {all_diffs_stacked.shape} [n_demos, n_layers, hidden_dim]")
    
    # Flatten: [n_demos, n_layers, hidden_dim] -> [n_demos, n_layers * hidden_dim]
    fit_data = all_diffs_stacked.reshape(n_demos, -1)
    
    print(f"  PCA input shape: {fit_data.shape}")
    
    # Normalize
    fit_data_normalized = fit_data.float()
    if fit_data_normalized.std() > 1e-6:
        fit_data_normalized = (fit_data_normalized - fit_data_normalized.mean(dim=0)) / (fit_data_normalized.std(dim=0) + 1e-8)
    
    # PCA
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data_normalized)
    
    # Get direction
    direction_flat = (pca.components_.sum(dim=0, keepdim=True) + pca.mean_).squeeze()
    reading_direction_flat = fit_data.mean(dim=0)
    
    # Reshape back: [n_layers * hidden_dim] -> [n_layers, hidden_dim]
    direction_per_layer = direction_flat.view(n_layers, hidden_dim)
    reading_direction_per_layer = reading_direction_flat.view(n_layers, hidden_dim)
    
    # Reshape to expected format: [n_layers, 1, hidden_dim]
    direction = direction_per_layer.unsqueeze(1)
    reading_direction = reading_direction_per_layer.unsqueeze(1)
    
    print(f"✓ Textual NSS direction computed for {n_layers} layers")
    print(f"  Final shape: {direction.shape}, norm={torch.norm(direction).item():.6f}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return direction, reading_direction
