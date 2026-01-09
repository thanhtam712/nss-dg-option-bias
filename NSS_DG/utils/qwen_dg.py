from typing import List, Tuple, Optional
import torch

class CaptionBiasReducerQwen:
    """
    Enhanced Caption Bias Reducer with question_end_idx detection and layer range support.
    
    This class implements a dynamic bias reduction mechanism that:
    1. Identifies token segments (image, question, caption)
    2. Detects question end using '?' token for better importance computation
    3. Computes importance of image tokens relative to question
    4. Adjusts text token representations based on image-caption differences
    5. Applies bias reduction only to specified layer ranges for efficiency
    """
    
    def __init__(self, model, processor, alpha: float = 0.1, top_k: int = 10, 
                 verbose: bool = False, layer_ranges: Optional[List[Tuple[int, int]]] = None):
        """
        Initialize the Enhanced Caption Bias Reducer.
        
        Args:
            model: The VLM model
            processor: The model's processor
            alpha: Strength of bias reduction adjustment (0.0 to 1.0)
            top_k: Number of top important image tokens to consider
            verbose: Print debug information
            layer_ranges: List of (start, end) tuples specifying which layer ranges to apply bias reduction.
                         If None or empty list, applies to all layers. 
                         Example: [(19, 33), (18, 35)] applies to layers 19-33 and 18-35
        """
        self.model = model
        self.processor = processor
        self.alpha = alpha
        self.top_k = top_k
        self.verbose = verbose
        self.layer_ranges = layer_ranges or []  # Empty list means apply to all layers
        self.hooks = []
        self.caption_token_indices = None
        self.question_token_indices = None
        self.image_token_indices = None
        self.question_end_idx = None 
    
    def _is_in_ranges(self, idx: int) -> bool:
        """
        Check if a layer index is within the specified layer ranges.
        
        Args:
            idx: Layer index to check
            
        Returns:
            True if idx is in any of the layer_ranges, or if layer_ranges is empty (apply to all)
        """
        # If no layer_ranges specified, apply to all layers
        if not self.layer_ranges:
            return True
        
        # Check if idx is in any of the specified ranges
        for start, end in self.layer_ranges:
            if start <= idx <= end:
                return True
        return False
        
    def identify_token_segments(self, input_ids: torch.Tensor, text: str) -> Tuple[List[int], List[int], List[int], Optional[int]]:
        """
        Identify which tokens belong to image, question, and caption segments.
        Enhanced version that also detects question_end_idx using '?' token.
        
        Args:
            input_ids: Token IDs from tokenizer
            text: Original text for debugging
            
        Returns:
            Tuple of (image_indices, question_indices, caption_indices, question_end_idx)
        """
        # Convert tokens only once and store in a list to avoid repeated conversions
        tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        if self.verbose:
            print(f"\n[Token Identification] Total tokens: {len(tokens)}")
        
        # Find vision token boundaries and question end
        vision_start_idx = None
        vision_end_idx = None
        caption_start_idx = None
        instruct_idx = None
        check_first_vision_start = True
        question_end_idx = None
        debug_token_question = True
        
        # Single pass through tokens to find all markers
        for idx, token in enumerate(tokens):
            token_str = str(token)
            if '<|vision_start|>' in token_str and check_first_vision_start:
                vision_start_idx = idx
                check_first_vision_start = False
            if '<|vision_end|>' in token_str:
                vision_end_idx = idx

            # NEW: Detect question end using '?' token
            if '?' in token_str and question_end_idx is None:
                question_end_idx = idx
                debug_token_question = False
            
            if vision_end_idx is not None and caption_start_idx is None and question_end_idx is None:
                if debug_token_question:
                    # Fallback: look for 'Question' token if '?' not found
                    if 'answer' in token_str.lower() and 'option' in str(tokens[idx+1]).lower():
                        question_end_idx = idx - 1 
                        debug_token_question = False
                        print('check 3')

            if vision_end_idx is not None and caption_start_idx is None:
                # if 'caption' in token_str.lower():
                if 'Here' in token_str and 'are' in str(tokens[idx+1]).lower():
                    caption_start_idx = idx
                    

            if 'instruction' in token_str.lower():
                instruct_idx = idx
                print("check 4")
                
        
        # Clean up tokens list immediately after use
        # del tokens
        
        # Extract indices
        image_indices = []
        question_indices = []
        caption_indices = []
        
        if vision_start_idx is not None and vision_end_idx is not None:
            image_indices = list(range(vision_start_idx, vision_end_idx + 1))
            question_start = vision_end_idx + 1
            if caption_start_idx is not None:
                question_indices = list(range(question_start, question_end_idx + 1 if question_end_idx is not None else caption_start_idx))
                if instruct_idx is not None:
                    caption_indices = list(range(caption_start_idx, instruct_idx))
                    print(f"Found 'Instruction' token at index {instruct_idx}, caption ends there {tokens[caption_indices[-1]]}.")
                    print(f"token question query end idx: {question_end_idx}, token: {tokens[question_end_idx]}")
                else:
                    caption_indices = list(range(caption_start_idx, len(input_ids[0])))
            else:
                question_indices = list(range(question_start, len(input_ids[0])))
        else:
            # Fallback: search for image pad tokens
            for idx in range(len(input_ids[0])):
                token_str = str(self.processor.tokenizer.convert_ids_to_tokens([input_ids[0][idx]])[0])
                if '<|image_pad|>' in token_str:
                    image_indices.append(idx)
            
            question_start = max(image_indices) + 1 if image_indices else 0
            
            # Search for caption start
            for idx in range(question_start, len(input_ids[0])):
                token_str = str(self.processor.tokenizer.convert_ids_to_tokens([input_ids[0][idx]])[0])
                if 'caption' in token_str.lower():
                    caption_start_idx = idx
                    break
            
            if caption_start_idx:
                question_indices = list(range(question_start, caption_start_idx))
                if instruct_idx is not None:
                    caption_indices = list(range(caption_start_idx, instruct_idx))
                else:
                    caption_indices = list(range(caption_start_idx, len(input_ids[0])))
            else:
                question_indices = list(range(question_start, len(input_ids[0])))
        
        if self.verbose:
            print(f"[Token Segments] Image: {len(image_indices)}, Question: {len(question_indices)}, "
                  f"Caption: {len(caption_indices)}, Question end idx: {question_end_idx}")
        
        return image_indices, question_indices, caption_indices, question_end_idx

    def compute_image_token_importance(self, hidden_states: torch.Tensor, 
                                      question_indices: List[int],
                                      image_indices: List[int],
                                      question_end_idx: Optional[int]) -> List[int]:
        """
        Compute which image tokens are most important for answering the question.
        Enhanced version that uses question_end_idx as query token if available.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            question_indices: Indices of question tokens
            image_indices: Indices of image tokens
            question_end_idx: Index of the '?' token (end of question)
            
        Returns:
            List of top-k most important image token indices
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Filter indices to only valid ones within current sequence length
        valid_image_indices = [idx for idx in image_indices if idx < seq_len]
        valid_question_indices = [idx for idx in question_indices if idx < seq_len]
        
        if len(valid_question_indices) == 0 or len(valid_image_indices) == 0:
            return valid_image_indices[:self.top_k] if len(valid_image_indices) >= self.top_k else valid_image_indices
        
        try:
            # NEW: Use question_end_idx as query if available and valid
            if question_end_idx is not None and question_end_idx < seq_len and False:
                question_repr = hidden_states[:, question_end_idx:question_end_idx+1, :]  # [B, 1, H]
                if self.verbose:
                    print(f"[Importance] Using question end token at index {question_end_idx}")
            else:
                # Fallback: use mean of all question tokens
                question_hidden = hidden_states[:, valid_question_indices, :]
                question_repr = question_hidden.mean(dim=1, keepdim=True)  # [B, 1, H]
                if self.verbose:
                    print(f"[Importance] Using mean of {len(valid_question_indices)} question tokens")
            
            image_hidden = hidden_states[:, valid_image_indices, :]  # [B, img_len, H]
            
            # Compute cosine similarity (normalize then compute similarity)
            question_norm = torch.nn.functional.normalize(question_repr, p=2, dim=-1)
            image_norm = torch.nn.functional.normalize(image_hidden, p=2, dim=-1)
            
            # Similarity scores: [B, img_len]
            similarity = torch.matmul(question_norm, image_norm.transpose(1, 2)).squeeze(1)
            
            # Get top-k image token indices
            top_k = min(self.top_k, len(valid_image_indices))
            top_vals, top_indices = torch.topk(similarity, k=top_k, dim=-1)
            
            # Convert to actual indices in the sequence (use tolist() once instead of item() in loop)
            top_indices_list = top_indices[0].tolist()
            top_image_indices = [valid_image_indices[idx] for idx in top_indices_list]
            
            # Clean up intermediate tensors
            del question_repr, image_hidden, question_norm, image_norm, similarity, top_vals, top_indices, top_indices_list
            
            return top_image_indices
            
        except Exception as e:
            print(f"Warning: Error in compute_image_token_importance: {e}")
            return valid_image_indices[:self.top_k] if len(valid_image_indices) >= self.top_k else valid_image_indices
    
    def apply_bias_reduction(self, hidden_states: torch.Tensor,
                            top_image_indices: List[int],
                            caption_indices: List[int],
                            question_indices: List[int]) -> torch.Tensor:
        """
        Apply bias reduction by updating text tokens with image information.
        
        The core idea: compute the difference between important image tokens and 
        caption tokens, then add this difference to text tokens to make them more
        image-grounded and less caption-biased.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim] (detached clone)
            top_image_indices: Indices of important image tokens
            caption_indices: Indices of caption tokens
            question_indices: Indices of question tokens
            
        Returns:
            Updated hidden_states with bias reduction applied
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Filter valid indices within current sequence length
        valid_caption_indices = [idx for idx in caption_indices if idx < seq_len]
        valid_question_indices = [idx for idx in question_indices if idx < seq_len]
        valid_top_image_indices = [idx for idx in top_image_indices if idx < seq_len]
        
        if len(valid_top_image_indices) == 0 or len(valid_caption_indices) == 0:
            return hidden_states
        
        # Get last caption token as reference
        last_caption_idx = valid_caption_indices[-1] if valid_caption_indices else None
        if last_caption_idx is None or last_caption_idx >= seq_len:
            return hidden_states
            
        try:
            last_caption_token = hidden_states[:, last_caption_idx, :]  # [B, H]
            
            # Compute difference between each top image token and last caption token
            diffs = []
            for img_idx in valid_top_image_indices:
                if img_idx < seq_len:
                    img_token = hidden_states[:, img_idx, :]
                    diffs.append(img_token - last_caption_token)
            
            if len(diffs) == 0:
                return hidden_states
            
            # Average the differences and normalize
            v = torch.stack(diffs, dim=0).mean(dim=0)  # [B, H]
            # v = torch.nn.functional.normalize(v, p=2, dim=-1)
            
            # Apply bias reduction to all text tokens (question + caption)
            text_indices = valid_question_indices + valid_caption_indices
            for idx in text_indices:
                if idx < seq_len:
                    # Safe assignment on hidden_states (it's a detached clone)
                    hidden_states[:, idx, :] = hidden_states[:, idx, :] + (self.alpha * 0.1 * v)
            
            # Clean up intermediate tensors
            
            del last_caption_token, diffs, v
                    
        except Exception as e:
            print(f"Warning: Error in apply_bias_reduction: {e}")
            return hidden_states
        
        return hidden_states
    
    def create_hook(self, layer_idx: int):
        """Create a forward hook for a specific layer."""
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            # Quick exit if no segments identified
            if not (self.image_token_indices and self.caption_token_indices):
                return output
            
            with torch.no_grad():
                # Work on detached clone to avoid modifying internal cached tensors
                try:
                    hs = hidden_states.detach().clone()
                except Exception:
                    hs = hidden_states.detach()
                
                # Compute importance and apply bias reduction
                top_image_indices = self.compute_image_token_importance(
                    hs,
                    self.question_token_indices,
                    self.image_token_indices,
                    self.question_end_idx  # NEW: Pass question_end_idx
                )
                
                hs = self.apply_bias_reduction(
                    hs,
                    top_image_indices,
                    self.caption_token_indices,
                    self.question_token_indices
                )
                
                # Free temporaries
                try:
                    del top_image_indices
                except Exception:
                    pass
            
            if isinstance(output, tuple):
                result = (hs,) + tuple(output[1:])
            else:
                result = hs
            
            return result
        
        return hook_fn
    
    def register_hooks(self):
        """Register hooks on specified transformer layers or all layers."""
        layers = None
        
        # Try different locations for layers based on model architecture
        # For Qwen2.5-VL, layers are in model.model.layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
            if self.verbose:
                print(f"[Hooks] Found layers in model.model.layers ({len(layers)} layers)")
        # For Qwen2.5-VL-ForConditionalGeneration, check language_model
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model') and hasattr(self.model.model.language_model, 'layers'):
            layers = self.model.model.language_model.layers
            if self.verbose:
                print(f"[Hooks] Found layers in model.model.language_model.layers ({len(layers)} layers)")
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
            if self.verbose:
                print(f"[Hooks] Found layers in model.layers ({len(layers)} layers)")
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
            if self.verbose:
                print(f"[Hooks] Found layers in model.transformer.h ({len(layers)} layers)")
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            layers = self.model.encoder.layer
            if self.verbose:
                print(f"[Hooks] Found layers in model.encoder.layer ({len(layers)} layers)")
        else:
            # Print debug info to help diagnose
            print("[Hooks] Could not find model layers. Checking model structure...")
            if hasattr(self.model, 'model'):
                print(f"[Hooks] model.model exists, attributes: {dir(self.model.model)[:10]}")
                if hasattr(self.model.model, 'language_model'):
                    print(f"[Hooks] model.model.language_model exists, attributes: {dir(self.model.model.language_model)[:10]}")
            raise ValueError("Could not find model layers. Please check model architecture.")
        
        # Register hooks only on layers in specified ranges
        registered_layers = []
        for idx, layer in enumerate(layers):
            if self._is_in_ranges(idx):
                hook = layer.register_forward_hook(self.create_hook(idx))
                self.hooks.append(hook)
                registered_layers.append(idx)
        
        if self.verbose:
            if self.layer_ranges:
                print(f"[Hooks] Registered on {len(registered_layers)} layers: {registered_layers}")
                print(f"[Hooks] Layer ranges: {self.layer_ranges}")
            else:
                print(f"[Hooks] Registered on all {len(registered_layers)} layers")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self.hooks = []
        
        if self.verbose:
            print("[Hooks] Removed all hooks")
    
    def prepare_for_generation(self, input_ids: torch.Tensor, text: str):
        """
        Prepare for generation by identifying token segments and registering hooks.
        
        Args:
            input_ids: Input token IDs
            text: Original text
        """
        # Identify token segments (includes question_end_idx)
        self.image_token_indices, self.question_token_indices, self.caption_token_indices, self.question_end_idx = \
            self.identify_token_segments(input_ids, text)
        # print(self.image_token_indices)
        # print(self.question_token_indices)
        # print(self.caption_token_indices)
        # print(f"Question end token index: {self.question_end_idx}")
        
        # Register hooks
        self.register_hooks()
    
    def cleanup_after_generation(self):
        """Cleanup after generation by removing hooks."""
        self.remove_hooks()
        # Clear token indices
        self.image_token_indices = None
        self.question_token_indices = None
        self.caption_token_indices = None
        self.question_end_idx = None