import math
import torch
import argparse
import os
import re
from pathlib import Path
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import numpy as np
from typing import List, Tuple, Optional
import json
import functools, builtins
print = functools.partial(builtins.print, flush=True)

model_path = 'Qwen/Qwen2.5-VL-3B-Instruct'

class CaptionBiasReducer:
    """Reduces caption bias by amplifying image-relevant tokens during inference"""
    
    def __init__(self, model, processor, alpha: float = 0.1, top_k: int = 10, layer_ranges: Optional[List[int]] = None):
        self.model = model
        self.processor = processor
        self.alpha = alpha
        self.top_k = top_k
        self.hooks = []
        self.caption_token_indices = None
        self.question_token_indices = None
        self.image_token_indices = None
        self.layer_ranges = layer_ranges or []
    
    def _is_in_ranges(self, idx: int) -> bool:
        for r in self.layer_ranges:
            start, end = r
            if start <= idx <= end:
                return True
        return False

        
    def identify_token_segments(self, input_ids: torch.Tensor, text: str) -> Tuple[List[int], List[int], List[int]]:
        """
        Identify which tokens belong to image, question, and caption
        Returns: (image_indices, question_indices, caption_indices)
        """
        tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids[0])
        print(tokens)
        # Find vision token boundaries
        vision_start_idx = None
        vision_end_idx = None
        caption_start_idx = None
        instruct_idx = None
        check_first_vision_start = True
        
        for idx, token in enumerate(tokens):
            token_str = str(token)
            
            # Find vision start
            if '<|vision_start|>' in token_str and check_first_vision_start:
                vision_start_idx = idx
                check_first_vision_start = False
                # print('check 1')
            
            # Find vision end
            if '<|vision_end|>' in token_str:
                vision_end_idx = idx
                # print('check 2')
            
            # Find caption start (look for "caption" token)
            if vision_end_idx is not None and caption_start_idx is None:
                if 'caption' in token_str.lower():
                    caption_start_idx = idx
                    # print('check 3')

            # if instruct at last of prompt
            if 'Instruction' in token_str.lower() or 'instruction' in token_str.lower():
                instruct_idx = idx
                # print('check 4')
            
    
        
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
                if instruct_idx is not None:
                    caption_indices = list(range(caption_start_idx, instruct_idx))
                    print(f"Found 'Instruction' token at index {instruct_idx}, caption ends there {tokens[caption_indices[-1]]}.")
                else:
                    caption_indices = list(range(caption_start_idx, len(tokens)))
                    print(f"end at the last token {tokens[-1]}.")
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
                if instruct_idx is not None:
                    caption_indices = list(range(caption_start_idx, instruct_idx))
                else:       
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
            # print(f"Applying bias reduction to tokens: {text_indices}")
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
            if self.layer_ranges and self._is_in_ranges(idx):
                # print(f"Registering hook on layer {idx} within specified ranges.")
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
        # print(self.image_token_indices)
        # print(self.question_token_indices)
        # print(self.caption_token_indices)
        # print(f"total tokens: {len(input_ids[0])}")
        # print(f"Token segments identified:")
        # print(f"  Image tokens: {len(self.image_token_indices)} tokens")
        # print(f"  Question tokens: {len(self.question_token_indices)} tokens")
        # print(f"indicate question: {self.question_token_indices}")
        # print(f"  Caption tokens: {len(self.caption_token_indices)} tokens")
        
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
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, padding_side='left', use_fast=True, max_pixels=512*512)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda:0",
        cache_dir="./cache_dir"
    ).eval()
    return model, processor


def get_response_with_bias_reduction(model, processor, image_path, question, 
                                     alpha=0.1, top_k=10, final_answer_tokens=1024, layer_ranges=None):
    """Generate response with caption bias reduction"""
    
    messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant tasked with selecting the correct answer ID based on the content of the provided images."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img
                    } for img in image_path
                ] + [
                {
                    "type": "text", 
                    "text": f"{question} Instruction: Choose the **correct ID** answer for the question below, you must response with the **ID** answer, don't include text."
                }
                ]
            }
        ]
    # messages = [
    #     {"role": "system", "content": "Instruction: Think step by step. Final answer MUST be only the single word 'Yes' or 'No'."},
    #     {"role": "user", "content": [
    #         {"type": "image", "image": image_path},
    #         {"type": "text", "text": question}
    #     ]}
    # ]


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
    bias_reducer = CaptionBiasReducer(model, processor, alpha=alpha, top_k=top_k, layer_ranges=layer_ranges)
    
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

    return output_text


def process_single_sample(model, processor, image_path, question, use_bias_reduction=True, alpha=0.1, top_k=10, layer_ranges=None):
    """Process a single sample with optional bias reduction"""
    
    if use_bias_reduction:
        answer = get_response_with_bias_reduction(
            model, processor, image_path, question, 
            alpha=alpha, top_k=top_k, layer_ranges=layer_ranges
        )
    else:
        # Original generation without bias reduction
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant tasked with selecting the correct answer ID based on the content of the provided images."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img
                    } for img in image_path
                ] + [
                {
                    "type": "text", 
                    "text": f"{question} Instruction: Choose the **correct ID** answer for the question below, you must response with the **ID** answer, don't include text."
                }
                ]
            }
        ]
        # messages = [
        #     {"role": "system", "content": "Instruction: Think step by step. Final answer MUST be only the single word 'Yes' or 'No'."},
        #     {"role": "user", "content": [
        #         {"type": "image", "image": image_path},
        #         {"type": "text", "text": question}
        #     ]}
        # ]


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

        answer = output_text
    
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_image_path', type=str, default='/mlcv3/WorkingSpace/Personal/baotg/TTam/Grad_Thesis/data/keyframes_new')
    parser.add_argument('--data_path', type=str, default='/mlcv3/WorkingSpace/Personal/baotg/TTam/Grad_Thesis/data/baseline_caption_outline_openai.json')
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--use_bias_reduction', action='store_true', default=True, help='Use caption bias reduction')
    parser.add_argument('--alpha', type=float, default=0.05, help='Strength of bias reduction (0.0 to 1.0)')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top image tokens to use')
    parser.add_argument('--compare', action='store_true', default=True, help='Compare with and without bias reduction')
    args = parser.parse_args()

    print("Loading model...")
    print(f"Model ID: {args.model_id}")
    print(f"Using caption bias reduction: {args.use_bias_reduction}, alpha: {args.alpha}, top_k: {args.top_k}")
    model, processor = load_model(args.model_id)
    
    layer_ranges = [[19, 33]]  # Example: apply hooks only on layers 0-3 and 8-11
    topk_list = [20]
    alpha_values = [0.03]
    data = json.load(open(args.data_path, 'r'))

    for layer_range in layer_ranges:
        for top_k in topk_list:
            for alpha in alpha_values:
                print("="*100)
                print(f"Top-k: {top_k}, Alpha: {alpha}, Layer Ranges: {layer_range}")
                print("="*100)

                number_of_samples = 0
                score_baseline = 0
                score_follow_caption_outline_base = 0
                score_bias_reduction = 0
                score_follow_caption_outline = 0

                for item in data:
                    video_id = item["video_id"]
                    question = item["question"]
                    caption_reasoning = item["reasoning"]
                    answer_choices = item["correct_answer_id"]
                    outline_answer_id = item["outline_answer_id"]
                    outline_information = item["outline_caption"]
                    
                    number_of_samples += 1
                    image_dir = Path(args.folder_image_path)
                    images = [str(img) for img in (image_dir / video_id).glob("*.jpg")]

                    text = f"""
Question: {item["question"]}
Answer options:
0: {item['answer_choice_0']}
1: {item['answer_choice_1']}
2: {item['answer_choice_2']}       
3: {item['answer_choice_3']}
4: {item['answer_choice_4']}
Caption: {outline_information}"""
                    if args.compare:
                        print("="*80)
                        try:
                            result_baseline = process_single_sample(model, processor, images, text, 
                                                use_bias_reduction=False)
                            # print(f"Baseline Result: {result_baseline}")
                            
                            result_recution = process_single_sample(model, processor, images, text,
                                                use_bias_reduction=True, alpha=alpha, top_k=top_k, layer_ranges=[layer_range])
                            # print(f"Bias Reduction Result: {result_recution}")
                        except Exception as e:
                            print(f"Error processing sample {video_id}: {e}")
                            continue

                        # base line result
                        try:
                            if int(result_baseline) == int(answer_choices):
                                score_baseline += 1
                            else:
                                if int(result_baseline) == int(outline_answer_id):
                                    score_follow_caption_outline_base += 1

                            # method result
                            if int(result_recution) == int(answer_choices):
                                score_bias_reduction += 1
                            else:
                                if int(result_recution) == int(outline_answer_id):
                                    score_follow_caption_outline += 1
                        except:
                            print("Error in parsing answer IDs.")
                        print("-"*20)    
                        print(f" Baseline Accuracy {score_baseline}/{number_of_samples} |||| {score_follow_caption_outline_base}/{number_of_samples}: {score_follow_caption_outline_base/number_of_samples:.4f}")
                        print(f" Bias Reduction Accuracy {score_bias_reduction}/{number_of_samples}|||| {score_follow_caption_outline}/{number_of_samples}: {score_follow_caption_outline/number_of_samples:.4f}")
                    else:
                        process_single_sample(model, processor, args.image_path, args.question,
                                            use_bias_reduction=args.use_bias_reduction,
                                            alpha=args.alpha, top_k=args.top_k)
                    # result_baseline = process_single_sample(model, processor, images, text, 
                    #                             use_bias_reduction=False)
                    # if int(result_baseline) == int(answer_choices):
                    #     score_baseline += 1
                    # else:
                    #     if int(result_baseline) == int(outline_answer_id):
                    #         score_follow_caption_outline_base += 1
                    # print(f" Baseline Accuracy {score_baseline}/{number_of_samples} |||| {score_follow_caption_outline_base}/{number_of_samples}: {score_follow_caption_outline_base/number_of_samples:.4f}")
            
# python src/tvh/version8/inference_dataset.py --compare > src/tvh/version8/log_version8.txt 2>&1
