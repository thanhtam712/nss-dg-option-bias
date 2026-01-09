import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional


class HiddenStatesHook:
    def __init__(self):
        self.hidden_states = []
        self.hooks = []
    
    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            self.hidden_states.append(output[0].detach().cpu())
        else:
            self.hidden_states.append(output.detach().cpu())
    
    def register_hooks(self, model):
        if hasattr(model, 'patch_embed'):
            hook = model.patch_embed.register_forward_hook(self.hook_fn)
            self.hooks.append(hook)
        
        if hasattr(model, 'blocks'):
            for block in model.blocks:
                hook = block.register_forward_hook(self.hook_fn)
                self.hooks.append(hook)
        
        if hasattr(model, 'merger'):
            hook = model.merger.register_forward_hook(self.hook_fn)
            self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear(self):
        self.hidden_states = []
    
    def get_hidden_states(self):
        return tuple(self.hidden_states)


def get_qwen_visual_hidden_states_from_tensor(
    model,
    processor,
    images_tensor: torch.Tensor,
    return_dict: bool = False
):
    from torchvision import transforms
    to_pil = transforms.ToPILImage()
    images = []

    if isinstance(images_tensor, list):
        for img_tensor in images_tensor:
            if not isinstance(img_tensor, torch.Tensor):
                img_tensor = img_tensor.unsqueeze(0).cuda()

            # Handle tensors that might have a batch dim of 1, e.g. (1, C, H, W)
            if img_tensor.dim() == 4:
                img_tensor = img_tensor.squeeze(0)
            images.append(to_pil(img_tensor))
    elif hasattr(images_tensor, 'shape'):
        # Input is a single tensor, e.g., (B, C, H, W) or (C, H, W)
        if images_tensor.dim() == 4: # Batch of images
            images = [to_pil(img) for img in images_tensor]
        elif images_tensor.dim() == 3: # Single image
            images = [to_pil(images_tensor)]
        else:
            raise ValueError(f"Unsupported tensor dimension: {images_tensor.dim()}")
    else:
        raise TypeError(f"Unsupported input type for images_tensor: {type(images_tensor)}")

    # Process the converted PIL images
    inputs = processor(text="", images=images, return_tensors="pt")

    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    
    if len(images) == 1:
        return get_qwen_visual_hidden_states_from_processor_output(
            model, inputs, return_dict=return_dict
        )
    else:
        all_hidden_states = []
        for idx in range(len(images)):
            h = get_qwen_visual_hidden_states(
                model,
                inputs['pixel_values'][idx:idx+1],
                inputs['image_grid_thw'][idx:idx+1],
                return_dict=return_dict
            )
            all_hidden_states.append(h)
        return all_hidden_states


def get_qwen_visual_hidden_states(
    model, 
    pixel_values: torch.Tensor,
    grid_thw: Optional[torch.Tensor] = None,
    return_dict: bool = False
):
    try:
        if hasattr(model, 'visual'):
            vision_model = model.visual
        elif hasattr(model, 'model') and hasattr(model.model, 'visual'):
            vision_model = model.model.visual
        else:
            vision_model = model.model.vision_tower.vision_tower.vision_model
    except:
        raise AttributeError("Cannot find vision model in the provided model structure")
    
    hook_manager = HiddenStatesHook()
    hook_manager.register_hooks(vision_model)
    
    try:
        with torch.no_grad():
            if not pixel_values.is_cuda and next(vision_model.parameters()).is_cuda:
                pixel_values = pixel_values.cuda()
            if not grid_thw.is_cuda and next(vision_model.parameters()).is_cuda:
                grid_thw = grid_thw.cuda()
            
            # Forward pass
            _ = vision_model(pixel_values, grid_thw=grid_thw)
            
            hidden_states = hook_manager.get_hidden_states()
    finally:
        hook_manager.remove_hooks()
    
    if return_dict:
        hidden_states_dict = {
            f'layer_{i}': h for i, h in enumerate(hidden_states)
        }
        return hidden_states_dict
    
    return hidden_states


def get_qwen_visual_hidden_states_from_processor_output(
    model,
    processor_output: Dict,
    image_idx: int = 0,
    return_dict: bool = False
):
    pixel_values = processor_output.get('pixel_values')
    grid_thw = processor_output.get('image_grid_thw')
    
    if pixel_values is None:
        raise ValueError("No pixel_values found in processor output")
    
    if pixel_values.dim() > 3 and pixel_values.shape[0] > 1:
        pixel_values = pixel_values[image_idx:image_idx+1]
        if grid_thw is not None:
            grid_thw = grid_thw[image_idx:image_idx+1]
    
    return get_qwen_visual_hidden_states(
        model, 
        pixel_values, 
        grid_thw=grid_thw,
        return_dict=return_dict
    )


def extract_hidden_states_batch(
    model, 
    pixel_values_list: List[torch.Tensor],
    grid_thw_list: Optional[List[torch.Tensor]] = None
):
    all_results = []
    
    if grid_thw_list is None:
        grid_thw_list = [None] * len(pixel_values_list)
    
    for pixel_values, grid_thw in zip(pixel_values_list, grid_thw_list):
        hidden_states = get_qwen_visual_hidden_states(
            model, 
            pixel_values,
            grid_thw=grid_thw
        )
        all_results.append(hidden_states)
    
    return all_results
