import math
import torch
import argparse
import os
import re
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import numpy as np
from typing import List, Tuple, Optional

model_path = 'Qwen/Qwen2.5-VL-3B-Instruct'

class CaptionBiasReducer:
    """Reduces caption bias by amplifying image-relevant tokens during inference"""
    
    def __init__(self, model, processor, alpha: float = 0.1, top_k: int = 10):
        self.model = model
        self.processor = processor
        self.alpha = alpha
        self.top_k = top_k
        self.hooks = []
        self.caption_token_indices = None
        self.question_token_indices = None
        self.image_token_indices = None
        
    def identify_token_segments(self, input_ids: torch.Tensor, text: str) -> Tuple[List[int], List[int], List[int]]:
        """
        Identify which tokens belong to image, question, and caption
        Returns: (image_indices, question_indices, caption_indices)
        """
        tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Find vision token boundaries
        vision_start_idx = None
        vision_end_idx = None
        caption_start_idx = None
        
        for idx, token in enumerate(tokens):
            token_str = str(token)
            
            # Find vision start
            if '<|vision_start|>' in token_str:
                vision_start_idx = idx
            
            # Find vision end
            if '<|vision_end|>' in token_str:
                vision_end_idx = idx
            
            # Find caption start (look for "caption" token)
            if vision_end_idx is not None and caption_start_idx is None:
                if 'caption' in token_str.lower():
                    caption_start_idx = idx
        
        # Extract indices
        image_indices = []
        question_indices = []
        caption_indices = []
        
        if vision_start_idx is not None and vision_end_idx is not None:
            # Image tokens are between <|vision_start|> and <|vision_end|> (inclusive)
            image_indices = list(range(vision_start_idx, vision_end_idx + 1))
            
            # Question starts after vision_end
            question_start = vision_end_idx + 1
            
            if caption_start_idx is not None:
                # Question ends at caption start
                question_indices = list(range(question_start, caption_start_idx))
                # Caption is from caption_start to end
                caption_indices = list(range(caption_start_idx, len(tokens)))
            else:
                # No caption found, all remaining tokens are question
                question_indices = list(range(question_start, len(tokens)))
        else:
            # Fallback: if we can't find vision markers
            print("Warning: Could not find vision tokens, using fallback")
            # Try to find any image_pad tokens
            for idx, token in enumerate(tokens):
                if '<|image_pad|>' in str(token):
                    image_indices.append(idx)
            
            if image_indices:
                question_start = max(image_indices) + 1
            else:
                question_start = 0
            
            # Find caption
            for idx in range(question_start, len(tokens)):
                if 'caption' in str(tokens[idx]).lower():
                    caption_start_idx = idx
                    break
            
            if caption_start_idx:
                question_indices = list(range(question_start, caption_start_idx))
                caption_indices = list(range(caption_start_idx, len(tokens)))
            else:
                question_indices = list(range(question_start, len(tokens)))
        
        return image_indices, question_indices, caption_indices
    
    def compute_image_token_importance(self, hidden_states: torch.Tensor, 
                                      question_indices: List[int],
                                      image_indices: List[int]) -> List[int]:
        """
        Compute which image tokens are most important for answering the question
        Uses attention-based importance scoring
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            question_indices: indices of question tokens
            image_indices: indices of image tokens
            
        Returns:
            List of top-k most important image token indices
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Filter indices to only valid ones within current sequence length
        valid_image_indices = [idx for idx in image_indices if idx < seq_len]
        valid_question_indices = [idx for idx in question_indices if idx < seq_len]
        
        if len(valid_question_indices) == 0 or len(valid_image_indices) == 0:
            return valid_image_indices[:self.top_k] if len(valid_image_indices) >= self.top_k else valid_image_indices
        
        # Extract question and image representations
        try:
            question_hidden = hidden_states[:, valid_question_indices, :]  # [batch, q_len, hidden]
            image_hidden = hidden_states[:, valid_image_indices, :]  # [batch, img_len, hidden]
        except Exception as e:
            print(f"Warning: Error accessing hidden states: {e}")
            return valid_image_indices[:self.top_k] if len(valid_image_indices) >= self.top_k else valid_image_indices
        
        # Compute similarity between question tokens and image tokens
        # Use mean of question tokens as query
        question_repr = question_hidden.mean(dim=1, keepdim=True)  # [batch, 1, hidden]
        
        # Compute cosine similarity
        question_norm = torch.nn.functional.normalize(question_repr, p=2, dim=-1)
        image_norm = torch.nn.functional.normalize(image_hidden, p=2, dim=-1)
        
        # Similarity scores: [batch, img_len]
        similarity = torch.matmul(question_norm, image_norm.transpose(1, 2)).squeeze(1)
        
        # Get top-k image token indices
        top_k = min(self.top_k, len(valid_image_indices))
        _, top_indices = torch.topk(similarity, k=top_k, dim=-1)
        
        # Convert to actual indices in the sequence
        top_image_indices = [valid_image_indices[idx.item()] for idx in top_indices[0]]
        
        return top_image_indices
    
    def apply_bias_reduction(self, hidden_states: torch.Tensor,
                            top_image_indices: List[int],
                            caption_indices: List[int],
                            question_indices: List[int]) -> torch.Tensor:
        """
        Apply the bias reduction by updating text tokens with image information
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            top_image_indices: indices of important image tokens
            caption_indices: indices of caption tokens
            question_indices: indices of question tokens
            
        Returns:
            Updated hidden_states
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Filter valid indices within current sequence length
        valid_caption_indices = [idx for idx in caption_indices if idx < seq_len]
        valid_question_indices = [idx for idx in question_indices if idx < seq_len]
        valid_top_image_indices = [idx for idx in top_image_indices if idx < seq_len]
        
        if len(valid_top_image_indices) == 0 or len(valid_caption_indices) == 0:
            return hidden_states
        
        # Get last caption token
        last_caption_idx = valid_caption_indices[-1] if valid_caption_indices else None
        if last_caption_idx is None or last_caption_idx >= seq_len:
            return hidden_states
            
        try:
            last_caption_token = hidden_states[:, last_caption_idx, :]  # [batch, hidden]
            
            # Compute difference vectors for top image tokens
            differences = []
            for img_idx in valid_top_image_indices:
                if img_idx < seq_len:
                    img_token = hidden_states[:, img_idx, :]  # [batch, hidden]
                    diff = img_token - last_caption_token
                    differences.append(diff)
            
            if len(differences) == 0:
                return hidden_states
            
            # Compute mean adjustment vector
            v = torch.stack(differences, dim=0).mean(dim=0)  # [batch, hidden]
            
            # Update both question and caption tokens (only valid ones)
            text_indices = valid_question_indices + valid_caption_indices
            for idx in text_indices:
                if idx < seq_len:
                    hidden_states[:, idx, :] = hidden_states[:, idx, :] + self.alpha * v
                    
        except Exception as e:
            print(f"Warning: Error in bias reduction: {e}")
            return hidden_states
        
        return hidden_states
    
    def create_hook(self, layer_idx: int):
        """Create a forward hook for a specific layer"""
        def hook_fn(module, input, output):
            # output is typically a tuple (hidden_states, ...)
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            # Apply bias reduction
            if self.image_token_indices and self.caption_token_indices:
                # Compute important image tokens
                top_image_indices = self.compute_image_token_importance(
                    hidden_states,
                    self.question_token_indices,
                    self.image_token_indices
                )
                
                # Apply bias reduction
                hidden_states = self.apply_bias_reduction(
                    hidden_states,
                    top_image_indices,
                    self.caption_token_indices,
                    self.question_token_indices
                )
                
                # Return modified output
                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                else:
                    return hidden_states
            
            return output
        
        return hook_fn
    
    def register_hooks(self):
        """Register hooks on all transformer layers"""
        # For Qwen2.5-VL, layers are in model.model.layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            layers = self.model.layers
        
        for idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(self.create_hook(idx))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_with_bias_reduction(self, inputs, input_ids, text, **generate_kwargs):
        """
        Generate with bias reduction applied
        
        Args:
            inputs: processed inputs for the model
            input_ids: input token ids
            text: original text for parsing
            **generate_kwargs: additional arguments for generation
        """
        # Identify token segments
        self.image_token_indices, self.question_token_indices, self.caption_token_indices = \
            self.identify_token_segments(input_ids, text)
        
        print(f"Token segments identified:")
        print(f"  Image tokens: {len(self.image_token_indices)} tokens")
        print(f"  Question tokens: {len(self.question_token_indices)} tokens")
        print(f"indicate question: {self.question_token_indices}")
        print(f"  Caption tokens: {len(self.caption_token_indices)} tokens")
        
        # Register hooks
        self.register_hooks()
        
        try:
            # Generate with hooks active
            with torch.no_grad():
                output = self.model.generate(**inputs, **generate_kwargs)
            return output
        finally:
            # Always remove hooks after generation
            self.remove_hooks()


def load_model(model_id):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, padding_side='left', use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda:0"
    ).to("cuda:0").eval()
    return model, processor


def get_response_with_bias_reduction(model, processor, image_path, question, 
                                     alpha=0.1, top_k=10, final_answer_tokens=1024):
    """Generate response with caption bias reduction"""
    
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": f"{question} You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE in <answer> </answer> tags, Let's think more."}
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Create bias reducer
    bias_reducer = CaptionBiasReducer(model, processor, alpha=alpha, top_k=top_k)
    
    # Generate with bias reduction
    gen_output = bias_reducer.generate_with_bias_reduction(
        inputs=inputs,
        input_ids=inputs.input_ids,
        text=text,
        max_new_tokens=final_answer_tokens,
        do_sample=True,
        output_attentions=True,
        return_dict_in_generate=True
    )
    
    gen_ids = gen_output.sequences
    trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0].strip()

    print("\nGenerated Answer (with bias reduction):")
    print(output_text)
    return output_text


def process_single_sample(model, processor, image_path, question, use_bias_reduction=True, alpha=0.1, top_k=10):
    """Process a single sample with optional bias reduction"""
    
    if use_bias_reduction:
        answer = get_response_with_bias_reduction(
            model, processor, image_path, question, 
            alpha=alpha, top_k=top_k
        )
    else:
        # Original generation without bias reduction
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"{question} You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE in <answer> </answer> tags, Let's think more."}
            ]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            gen_output = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                output_attentions=True,
                return_dict_in_generate=True
            )
            gen_ids = gen_output.sequences
            trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
            output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0].strip()

        print("\nGenerated Answer (baseline):")
        print(output_text)
        answer = output_text
    
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='/mlcv3/WorkingSpace/Personal/baotg/TTam/Grad_Thesis/src/tvh/verson7_1/OIP.webp')
    parser.add_argument('--question', type=str, default='What color is the dog in the image? caption: a person with a black dog near a river.')
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--use_bias_reduction', action='store_true', default=True, help='Use caption bias reduction')
    parser.add_argument('--alpha', type=float, default=0.05, help='Strength of bias reduction (0.0 to 1.0)')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top image tokens to use')
    parser.add_argument('--compare', action='store_true', help='Compare with and without bias reduction')
    args = parser.parse_args()

    model, processor = load_model(args.model_id)
    
    if args.compare:
        print("="*80)
        print("BASELINE (without bias reduction)")
        print("="*80)
        process_single_sample(model, processor, args.image_path, args.question, 
                            use_bias_reduction=False)
        
        print("\n" + "="*80)
        print(f"WITH BIAS REDUCTION (alpha={args.alpha}, top_k={args.top_k})")
        print("="*80)
        process_single_sample(model, processor, args.image_path, args.question,
                            use_bias_reduction=True, alpha=args.alpha, top_k=args.top_k)
    else:
        process_single_sample(model, processor, args.image_path, args.question,
                            use_bias_reduction=args.use_bias_reduction,
                            alpha=args.alpha, top_k=args.top_k)