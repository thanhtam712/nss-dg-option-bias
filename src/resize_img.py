from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image, ImageFilter
import torch
import numpy as np

# Load a pre-trained ViT model and feature extractor
model_name = 'google/vit-large-patch32-384'
processor = ViTImageProcessor.from_pretrained(model_name, do_rescale=False)
model = ViTForImageClassification.from_pretrained(model_name, attn_implementation='eager')

# Run an image through the pipline
inputs = processor(images="dogandcat.png", return_tensors="pt")
outputs = model(**inputs, output_attentions=True)

# Getting the attentions
attentions = np.load('attentions.npy', allow_pickle=True)

def attention_rollout(attentions):
    # Initialize rollout with identity matrix
    rollout = torch.eye(attentions[0].size(-1)).to(attentions[0].device)

    # Multiply attention maps layer by layer
    for attention in attentions:
        attention_heads_fused = attention.mean(dim=1) # Average attention across heads
        attention_heads_fused += torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device) # A + I
        attention_heads_fused /= attention_heads_fused.sum(dim=-1, keepdim=True) # Normalizing A
        rollout = torch.matmul(rollout, attention_heads_fused) # Multiplication

    return rollout

rollout = attention_rollout(attentions)
num_of_patches = rollout.size(-1) - 1

cls_attention = rollout[0, 1:, 0]  # Get attention values from [CLS] token to all patches
cls_attention = 1 - cls_attention.reshape(int(np.sqrt(num_of_patches)), int(np.sqrt(num_of_patches)))

# Normalize the attention map for better visualization
cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min())

img_size = inputs['pixel_values'].shape[-1]

# Resize and blur the attention map
cls_attention_resized = Image.fromarray((cls_attention * 255).astype(np.uint8)).resize((img_size, img_size), resample=Image.BICUBIC)
cls_attention_resized = cls_attention_resized.filter(ImageFilter.GaussianBlur(radius=2))

# Convert the attention map to RGBA
cls_attention_colored = np.array(cls_attention_resized.convert("L"))
cls_attention_colored = np.stack([cls_attention_colored]*3 + [cls_attention_colored], axis=-1)

# Adjust the alpha channel to control brightness
cls_attention_colored_img = Image.fromarray(cls_attention_colored, mode="RGBA")
cls_attention_colored_img.putalpha(100)  # Adjust alpha for blending (lower value for darker overlay)
