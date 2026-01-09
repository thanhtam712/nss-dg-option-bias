import torch
from typing import List, Tuple, Optional

class CaptionBiasReducerInternVL:
    """
    Caption Bias Reducer adapted for InternVL models.
    Uses hooks to detect actual visual token positions in hidden states.
    """
    
    def __init__(self, model, tokenizer, alpha: float = 0.1, top_k: int = 10, 
                 verbose: bool = False, layer_ranges: Optional[List[Tuple[int, int]]] = None):
        """
        Initialize the Caption Bias Reducer for InternVL.
        
        Args:
            model: The InternVL model
            tokenizer: The model's tokenizer
            alpha: Strength of bias reduction adjustment (0.0 to 1.0)
            top_k: Number of top important image tokens to consider
            verbose: Print debug information
            layer_ranges: List of (start, end) tuples specifying which layer ranges to apply bias reduction.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.top_k = top_k
        self.verbose = verbose
        self.layer_ranges = layer_ranges or []
        
        # Hooks
        self.hooks = []
        self.detection_hook = None
        
        # Token segment info (will be detected from hidden states)
        self.image_token_indices = None
        self.question_token_indices = None
        self.caption_token_indices = None
        self.question_end_idx = None
        
        # Detection state
        self.sequence_info_detected = False
        self.actual_seq_len = None
        self.original_token_len = None
        self.num_images = None
    
    def _is_in_ranges(self, idx: int) -> bool:
        """Check if a layer index is within the specified layer ranges."""
        if not self.layer_ranges:
            return True
        
        for start, end in self.layer_ranges:
            if start <= idx <= end:
                return True
        return False
    
    def detect_sequence_structure(self, input_ids: torch.Tensor, prompt: str):
        """
        Set up detection to find actual visual token positions.
        This prepares metadata needed for the detection hook.
        
        Args:
            input_ids: Original input token IDs [batch, seq_len]
            prompt: Original text prompt
        """
        self.original_token_len = input_ids.shape[1]
        self.num_images = prompt.count('<image>')
        
        # Find <image> marker positions in original input_ids
        # Try multiple ways to find the marker
        image_marker_ids = self.tokenizer.encode('<image>', add_special_tokens=False)
        
        input_ids_list = input_ids[0].tolist()
        self.image_marker_positions = []
        pattern_len = len(image_marker_ids)
        
        # Method 1: Direct pattern matching
        i = 0
        while i < len(input_ids_list) - pattern_len + 1:
            if input_ids_list[i:i+pattern_len] == image_marker_ids:
                self.image_marker_positions.append(i)
                i += pattern_len
            else:
                i += 1
        
        # Method 2: If no matches found, try finding individual <image> token ID
        # Some models use a single special token for <image>
        if len(self.image_marker_positions) == 0:
            if hasattr(self.tokenizer, 'img_context_token_id'):
                img_token_id = self.tokenizer.img_context_token_id
            elif '<image>' in self.tokenizer.get_vocab():
                img_token_id = self.tokenizer.convert_tokens_to_ids('<image>')
            else:
                img_token_id = None
            
            if img_token_id is not None and img_token_id in input_ids_list:
                for idx, token_id in enumerate(input_ids_list):
                    if token_id == img_token_id:
                        self.image_marker_positions.append(idx)
        
        # Method 3: If still no matches, search for the pattern more flexibly
        if len(self.image_marker_positions) == 0:
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            for idx in range(len(tokens) - 2):
                token_str = str(tokens[idx])
                next_token = str(tokens[idx + 1])
                next_next_token = str(tokens[idx + 2]) if idx + 2 < len(tokens) else ""
                
                # Check for '<' 'image' '>' pattern
                if ('<' in token_str and 'image' in next_token.lower() and '>' in next_next_token):
                    # Mark the start of the pattern
                    self.image_marker_positions.append(idx)
                    if self.verbose:
                        print(f"  Found <image> pattern at idx {idx}: '{token_str}' '{next_token}' '{next_next_token}'")
        
        # Tokenize to find text boundaries
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        if self.verbose:
            print(f"\n[Detection Setup]")
            print(f"  Original token length: {self.original_token_len}")
            print(f"  Number of images: {self.num_images}")
            print(f"  Image marker positions: {self.image_marker_positions}")
            print(f"  Image marker IDs: {image_marker_ids}")
            print(f"  First 20 tokens: {tokens[:20]}")
            print(f"  First 20 token IDs: {input_ids_list[:20]}")
        
        # If we still couldn't find markers but num_images > 0, assume they're at the start
        if len(self.image_marker_positions) == 0 and self.num_images > 0:
            print(f"  [Warning] Could not find image markers, assuming they start at position 0")
            # For InternVL, images typically come first
            # We'll detect the actual positions in the hook based on sequence length change
            self.image_marker_positions = [0]  # Placeholder
        
        # Find text segment boundaries in ORIGINAL tokenization
        self.original_question_end_idx = None
        self.original_caption_start_idx = None
        self.original_instruction_idx = None
        
        for idx, token in enumerate(tokens):
            token_str = str(token)
            
            # Find question end
            if '?' in token_str and self.original_question_end_idx is None:
                self.original_question_end_idx = idx
                if self.verbose:
                    print(f"  Original question end at idx {idx}")

            # trackback '?' not found
            if self.original_question_end_idx is None and 'Answer' in token_str:
                self.original_question_end_idx = idx - 1
                if self.verbose:
                    print(f"  Original question end (fallback) at idx {idx}")
            
            # Find caption start
            if self.original_caption_start_idx is None:
                if 'Here' in token_str and idx + 2 < len(tokens):
                    next_tokens = str(tokens[idx+1]) + str(tokens[idx+2])
                    if 'are' in next_tokens.lower() and 'some' in next_tokens.lower():
                        self.original_caption_start_idx = idx
                        if self.verbose:
                            print(f"  Original caption start at idx {idx}")
            
            # Find instruction start
            if 'Choose' in token_str or 'Instruction' in token_str:
                self.original_instruction_idx = idx
                if self.verbose:
                    print(f"  Original instruction start at idx {idx}")
        
        self.sequence_info_detected = False
    
    def create_detection_hook(self):
        """
        Create a one-time hook to detect the actual sequence structure
        after visual tokens have been injected.
        """
        def detection_hook_fn(module, input, output):
            if self.sequence_info_detected:
                return output
            
            hidden_states = output[0] if isinstance(output, tuple) else output
            self.actual_seq_len = hidden_states.shape[1]
            
            if self.verbose:
                print(f"\n[Detection Hook Triggered]")
                print(f"  Actual sequence length: {self.actual_seq_len}")
                print(f"  Original token length: {self.original_token_len}")
            
            # Calculate visual token info
            marker_len = len(self.tokenizer.encode('<image>', add_special_tokens=False))
            
            # Handle case where we couldn't find markers
            if len(self.image_marker_positions) == 0:
                if self.num_images > 0:
                    # Estimate: visual tokens are the difference in sequence length
                    total_visual_tokens = self.actual_seq_len - self.original_token_len
                    num_visual_tokens_per_image = total_visual_tokens // self.num_images
                    
                    if self.verbose:
                        print(f"  [Estimation] Total expansion: {total_visual_tokens} tokens")
                        print(f"  [Estimation] Visual tokens per image: {num_visual_tokens_per_image}")
                    
                    # Assume images are at the beginning
                    self.image_token_indices = list(range(0, total_visual_tokens))
                    
                    # Text starts after all visual tokens
                    text_start = total_visual_tokens
                    
                    # Map text boundaries (they shift by total_visual_tokens - original_marker_space)
                    def map_pos_simple(orig_pos):
                        if orig_pos is None:
                            return None
                        # All text positions shift by the visual expansion amount
                        return orig_pos + total_visual_tokens
                    
                    question_end_idx = map_pos_simple(self.original_question_end_idx)
                    caption_start_idx = map_pos_simple(self.original_caption_start_idx)
                    instruction_idx = map_pos_simple(self.original_instruction_idx)
                    
                    # Determine ranges
                    if caption_start_idx is not None:
                        self.question_token_indices = list(range(text_start, question_end_idx))
                        caption_end = instruction_idx if instruction_idx is not None else self.actual_seq_len
                        self.caption_token_indices = list(range(caption_start_idx, caption_end))
                    else:
                        question_end = instruction_idx if instruction_idx is not None else self.actual_seq_len
                        self.question_token_indices = list(range(text_start, question_end))
                        self.caption_token_indices = []
                    
                    self.question_end_idx = question_end_idx
                    
                else:
                    # No images at all
                    self.image_token_indices = []
                    self.question_token_indices = []
                    self.caption_token_indices = []
                    self.question_end_idx = None
            else:
                # Normal case: we found the markers
                total_marker_tokens = self.num_images * marker_len
                total_visual_tokens = self.actual_seq_len - self.original_token_len + total_marker_tokens
                
                if self.num_images > 0:
                    num_visual_tokens_per_image = total_visual_tokens // self.num_images
                else:
                    num_visual_tokens_per_image = 0
                
                if self.verbose:
                    print(f"  Visual tokens per image: {num_visual_tokens_per_image}")
                    print(f"  Total visual tokens: {total_visual_tokens}")
                
                # Map original positions to actual positions
                self.image_token_indices = []
                offset = 0
                
                for marker_pos in self.image_marker_positions:
                    # Account for previous expansions
                    actual_start = marker_pos + offset
                    actual_end = actual_start + num_visual_tokens_per_image
                    
                    # Visual tokens occupy this range
                    visual_range = list(range(actual_start, actual_end))
                    self.image_token_indices.extend(visual_range)
                    
                    if self.verbose:
                        print(f"  Image marker at {marker_pos} → visual tokens [{actual_start}:{actual_end}]")
                    
                    # Update offset for next marker
                    offset += (num_visual_tokens_per_image - marker_len)
                
                # Map text segment positions
                def map_position(original_pos):
                    """Map original token position to actual position after visual expansion."""
                    if original_pos is None:
                        return None
                    
                    mapped_pos = original_pos
                    for marker_pos in self.image_marker_positions:
                        if original_pos > marker_pos:
                            # This text position comes after this image marker
                            mapped_pos += (num_visual_tokens_per_image - marker_len)
                    
                    return mapped_pos
                
                # Map all text boundaries
                question_end_idx = map_position(self.original_question_end_idx)
                caption_start_idx = map_position(self.original_caption_start_idx)
                instruction_idx = map_position(self.original_instruction_idx)
                
                # Determine question and caption ranges
                self.question_token_indices = []
                self.caption_token_indices = []
                
                if self.image_token_indices:
                    # Question starts after last visual token
                    question_start = max(self.image_token_indices) + 1
                    
                    if caption_start_idx is not None:
                        # Question ends at caption start
                        self.question_token_indices = list(range(question_start, question_end_idx))
                        
                        # Caption ends at instruction or sequence end
                        caption_end = instruction_idx if instruction_idx is not None else self.actual_seq_len
                        self.caption_token_indices = list(range(caption_start_idx, caption_end))
                    else:
                        # No caption - question goes to instruction or end
                        question_end = instruction_idx if instruction_idx is not None else self.actual_seq_len
                        self.question_token_indices = list(range(question_start, question_end))
                
                self.question_end_idx = question_end_idx
            
            if self.verbose:
                print(f"\n[Detected Segments]")
                print(f"  Image tokens: {len(self.image_token_indices)} tokens")
                if self.image_token_indices:
                    print(f"    Ranges: {self.image_token_indices[:5]}...{self.image_token_indices[-5:] if len(self.image_token_indices) > 10 else ''}")
                print(f"  Question tokens: {len(self.question_token_indices)} tokens")
                if self.question_token_indices:
                    print(f"    Range: [{min(self.question_token_indices)}, {max(self.question_token_indices)}]")
                print(f"  Caption tokens: {len(self.caption_token_indices)} tokens")
                if self.caption_token_indices:
                    print(f"    Range: [{min(self.caption_token_indices)}, {max(self.caption_token_indices)}]")
                print(f"  Question end idx: {self.question_end_idx}")
            
            self.sequence_info_detected = True
            
            return output
        
        return detection_hook_fn
    
    def compute_image_token_importance(self, hidden_states: torch.Tensor, 
                                      question_indices: List[int],
                                      image_indices: List[int],
                                      question_end_idx: Optional[int]) -> List[int]:
        """
        Compute which image tokens are most important for answering the question.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Filter valid indices
        valid_image_indices = [idx for idx in image_indices if idx < seq_len]
        valid_question_indices = [idx for idx in question_indices if idx < seq_len]
        
        if len(valid_question_indices) == 0 or len(valid_image_indices) == 0:
            return valid_image_indices[:self.top_k] if len(valid_image_indices) >= self.top_k else valid_image_indices
        
        try:
            # Use mean of question tokens as query representation
            question_hidden = hidden_states[:, valid_question_indices, :]
            question_repr = question_hidden.mean(dim=1, keepdim=True)  # [B, 1, H]
            
            image_hidden = hidden_states[:, valid_image_indices, :]  # [B, img_len, H]
            
            # Compute cosine similarity
            question_norm = torch.nn.functional.normalize(question_repr, p=2, dim=-1)
            image_norm = torch.nn.functional.normalize(image_hidden, p=2, dim=-1)
            
            similarity = torch.matmul(question_norm, image_norm.transpose(1, 2)).squeeze(1)  # [B, img_len]
            
            # Get top-k
            top_k = min(self.top_k, len(valid_image_indices))
            top_vals, top_indices = torch.topk(similarity, k=top_k, dim=-1)
            
            top_indices_list = top_indices[0].tolist()
            top_image_indices = [valid_image_indices[idx] for idx in top_indices_list]
            
            # Cleanup
            del question_repr, image_hidden, question_norm, image_norm, similarity, top_vals, top_indices
            
            return top_image_indices
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Error in compute_image_token_importance: {e}")
            return valid_image_indices[:self.top_k] if len(valid_image_indices) >= self.top_k else valid_image_indices
    
    def apply_bias_reduction(self, hidden_states: torch.Tensor,
                            top_image_indices: List[int],
                            caption_indices: List[int],
                            question_indices: List[int]) -> torch.Tensor:
        """
        Apply bias reduction by updating text tokens with image information.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Filter valid indices
        valid_caption_indices = [idx for idx in caption_indices if idx < seq_len]
        valid_question_indices = [idx for idx in question_indices if idx < seq_len]
        valid_top_image_indices = [idx for idx in top_image_indices if idx < seq_len]
        
        if len(valid_top_image_indices) == 0 or len(valid_caption_indices) == 0:
            return hidden_states
        
        # Get last caption token as reference
        last_caption_idx = valid_caption_indices[-1]
        if last_caption_idx >= seq_len:
            return hidden_states
        
        try:
            last_caption_token = hidden_states[:, last_caption_idx, :]  # [B, H]
            
            # Compute difference between top image tokens and caption
            diffs = []
            for img_idx in valid_top_image_indices:
                if img_idx < seq_len:
                    img_token = hidden_states[:, img_idx, :]
                    diffs.append(img_token - last_caption_token)
            
            if len(diffs) == 0:
                return hidden_states
            
            # Average the differences
            v = torch.stack(diffs, dim=0).mean(dim=0)  # [B, H]
            
            # Apply bias reduction to all text tokens
            text_indices = valid_question_indices + valid_caption_indices
            check_add = True
            for idx in text_indices:
                if idx < seq_len:
                    hidden_states[:, idx, :] = hidden_states[:, idx, :] + (self.alpha * 0.1 * v)
                    if v is not None:
                        check_add = False
            
            if check_add and self.verbose:
                print("  [Warning] No text tokens were updated in apply_bias_reduction")
            
            # Cleanup
            del last_caption_token, diffs, v
                    
        except Exception as e:
            if self.verbose:
                print(f"Warning: Error in apply_bias_reduction: {e}")
            return hidden_states
        
        return hidden_states
    
    def create_bias_reduction_hook(self, layer_idx: int):
        """Create a forward hook for bias reduction on a specific layer."""
        def hook_fn(module, input, output):
            # Only apply if sequence info has been detected
            if not self.sequence_info_detected:
                return output
            
            # Quick exit if no segments identified
            if not (self.image_token_indices and self.caption_token_indices):
                return output
            
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            with torch.no_grad():
                # Work on detached clone
                try:
                    hs = hidden_states.detach().clone()
                except Exception:
                    hs = hidden_states.detach()
                
                # Compute importance and apply bias reduction
                top_image_indices = self.compute_image_token_importance(
                    hs,
                    self.question_token_indices,
                    self.image_token_indices,
                    self.question_end_idx
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
    
    def register_detection_hook(self):
        """Register detection hook on the first language model layer."""
        layers = self._get_language_layers()
        
        if layers is None or len(layers) == 0:
            raise ValueError("Could not find language model layers for InternVL")
        
        # Register detection hook on first layer
        first_layer = layers[0]
        self.detection_hook = first_layer.register_forward_hook(self.create_detection_hook())
        
        if self.verbose:
            print(f"[Hooks] Registered detection hook on layer 0")
    
    def register_bias_reduction_hooks(self):
        """Register bias reduction hooks on specified layers."""
        layers = self._get_language_layers()
        
        if layers is None:
            raise ValueError("Could not find language model layers for InternVL")
        
        # Register hooks on specified layers
        registered_layers = []
        for idx, layer in enumerate(layers):
            if self._is_in_ranges(idx):
                hook = layer.register_forward_hook(self.create_bias_reduction_hook(idx))
                self.hooks.append(hook)
                registered_layers.append(idx)
        
        if self.verbose:
            if self.layer_ranges:
                print(f"[Hooks] Registered bias reduction on {len(registered_layers)} layers: {registered_layers}")
                print(f"[Hooks] Layer ranges: {self.layer_ranges}")
            else:
                print(f"[Hooks] Registered bias reduction on all {len(registered_layers)} layers")
    
    def _get_language_layers(self):
        """Get language model layers from InternVL model."""
        layers = None
        
        # InternVL architecture: model.language_model.model.layers
        if hasattr(self.model, 'language_model'):
            if hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'layers'):
                layers = self.model.language_model.model.layers
            elif hasattr(self.model.language_model, 'layers'):
                layers = self.model.language_model.layers
        
        # Alternative: model.llm
        if layers is None and hasattr(self.model, 'llm'):
            if hasattr(self.model.llm, 'model') and hasattr(self.model.llm.model, 'layers'):
                layers = self.model.llm.model.layers
            elif hasattr(self.model.llm, 'layers'):
                layers = self.model.llm.layers
        
        return layers
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        # Remove detection hook
        if self.detection_hook is not None:
            try:
                self.detection_hook.remove()
            except Exception:
                pass
            self.detection_hook = None
        
        # Remove bias reduction hooks
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self.hooks = []
        
        if self.verbose:
            print("[Hooks] Removed all hooks")
    
    def prepare_for_generation(self, input_ids: torch.Tensor, prompt: str):
        """
        Prepare for generation by setting up detection and registering hooks.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            prompt: Original prompt text
        """
        # Reset detection state
        self.sequence_info_detected = False
        self.image_token_indices = None
        self.question_token_indices = None
        self.caption_token_indices = None
        self.question_end_idx = None
        
        # Set up detection metadata
        self.detect_sequence_structure(input_ids, prompt)
        
        # Register detection hook (will trigger on first forward pass)
        self.register_detection_hook()
        
        # Register bias reduction hooks (will activate after detection)
        self.register_bias_reduction_hooks()
        
        if self.verbose:
            print("\n[Preparation Complete]")
            print("  Detection hook: Ready")
            print("  Bias reduction hooks: Ready")
            print("  Waiting for forward pass to detect sequence structure...")
    
    def cleanup_after_generation(self):
        """Cleanup after generation by removing hooks and clearing state."""
        self.remove_hooks()
        
        # Clear all state
        self.sequence_info_detected = False
        self.image_token_indices = None
        self.question_token_indices = None
        self.caption_token_indices = None
        self.question_end_idx = None
        self.actual_seq_len = None
        self.original_token_len = None
        self.num_images = None
        
        if self.verbose:
            print("[Cleanup] All state cleared")
