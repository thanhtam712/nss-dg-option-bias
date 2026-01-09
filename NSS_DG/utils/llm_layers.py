import torch
from torch.nn import functional as F
from torch import nn
from transformers import PreTrainedModel
from torch import Tensor
import numpy as np


class NSSLayer(nn.Module):

    def __init__(self, nss_direction, lam):
        super(NSSLayer, self).__init__()
        self.nss_direction = nss_direction
        self.lam = lam

    def forward(self, x):
        if self.nss_direction is not None:
            # Store original dtype to preserve it
            original_dtype = x.dtype
            
            # Ensure nss_direction is on the same device as x
            if self.nss_direction[0].device != x.device:
                self.nss_direction = [d.to(x.device) for d in self.nss_direction]
            
            # Compute in float32 for numerical stability
            norm = torch.norm(x.float(),dim=-1).unsqueeze(-1)            
            y = torch.zeros_like(x.float())  # Initialize y with same shape as x
            
            # Handle both 2D and 3D inputs
            # x can be [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]
            if x.dim() == 2:
                # 2D case: [seq_len, hidden_dim]
                seq_len, hidden_dim = x.shape
                batch_size = None
                is_2d = True
            elif x.dim() == 3:
                # 3D case: [batch_size, seq_len, hidden_dim]
                batch_size, seq_len, hidden_dim = x.shape
                is_2d = False
            else:
                print(f"Warning: Unexpected input shape: {x.shape}")
                return x.to(original_dtype)
            
            for i in range(len(self.nss_direction)):
                # Ensure direction is on correct device
                direction = self.nss_direction[i].to(x.device)
                
                # Handle different direction shapes
                if direction.dim() == 1:
                    # Shape: [hidden_dim] -> broadcast to all tokens
                    direction_normalized = F.normalize(direction, dim=-1)
                    if is_2d:
                        # For 2D input: [hidden_dim] -> [1, hidden_dim] -> [seq_len, hidden_dim]
                        direction_expanded = direction_normalized.unsqueeze(0).expand(seq_len, -1)
                    else:
                        # For 3D input: [hidden_dim] -> [1, 1, hidden_dim] -> [batch_size, seq_len, hidden_dim]
                        direction_expanded = direction_normalized.view(1, 1, -1).expand(batch_size, seq_len, -1)
                    
                elif direction.dim() == 2:
                    # Shape: [n_tokens, hidden_dim]
                    n_tokens, dir_hidden = direction.shape
                    
                    # Check if hidden_dim matches
                    if dir_hidden != hidden_dim:
                        print(f"Warning: nss direction hidden_dim {dir_hidden} != model hidden_dim {hidden_dim}")
                        continue
                    
                    # Interpolate tokens if needed
                    if n_tokens != seq_len:
                        # Interpolate: [n_tokens, hidden_dim] -> [seq_len, hidden_dim]
                        direction = F.interpolate(
                            direction.permute(1, 0).unsqueeze(0),  # [n_tokens, hidden] -> [1, hidden, n_tokens]
                            size=seq_len,
                            mode='linear',
                            align_corners=False
                        ).squeeze(0).permute(1, 0)  # [1, hidden, seq_len] -> [seq_len, hidden]
                    
                    # Normalize and expand
                    direction_normalized = F.normalize(direction, dim=-1)
                    if is_2d:
                        # For 2D input: already [seq_len, hidden_dim]
                        direction_expanded = direction_normalized
                    else:
                        # For 3D input: [seq_len, hidden_dim] -> [batch_size, seq_len, hidden_dim]
                        direction_expanded = direction_normalized.unsqueeze(0).expand(batch_size, -1, -1)
                    
                elif direction.dim() == 3:
                    # Shape: [batch_dir, n_tokens, hidden_dim]
                    # Take first batch or average
                    direction = direction[0] if direction.shape[0] == 1 else direction.mean(dim=0)
                    
                    n_tokens, dir_hidden = direction.shape
                    
                    if dir_hidden != hidden_dim:
                        print(f"Warning: nss direction hidden_dim {dir_hidden} != model hidden_dim {hidden_dim}")
                        continue
                    
                    # Interpolate tokens if needed
                    if n_tokens != seq_len:
                        direction = F.interpolate(
                            direction.permute(1, 0).unsqueeze(0),
                            size=seq_len,
                            mode='linear',
                            align_corners=False
                        ).squeeze(0).permute(1, 0)
                    
                    # Normalize and expand
                    direction_normalized = F.normalize(direction, dim=-1)
                    if is_2d:
                        # For 2D input: [seq_len, hidden_dim]
                        direction_expanded = direction_normalized
                    else:
                        # For 3D input: [seq_len, hidden_dim] -> [batch_size, seq_len, hidden_dim]
                        direction_expanded = direction_normalized.unsqueeze(0).expand(batch_size, -1, -1)
                else:
                    print(f"Warning: Unexpected nss direction shape: {direction.shape}")
                    continue
                
                lambda_sim = 1.0
                y += self.lam[i] * lambda_sim * direction_expanded
                
            y = y / len(self.nss_direction)
            x = F.normalize(F.normalize(x.float(),dim=-1) +  0.1 * y, dim=-1) * norm
            
            # Clean any non-finite values before returning
            if not torch.isfinite(x).all():
                print(f"Warning: NSSLayer produced non-finite values! Cleaning...")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # CRITICAL: Convert back to ORIGINAL dtype (bfloat16 for Qwen, float16 for others)
            # DO NOT use .half() as it always converts to float16!
            return x.to(original_dtype)
        else:
            return x


class nss_MLP(nn.Module):
    def __init__(self, original_mlp, nss_direction, alpha):
        super(nss_MLP, self).__init__()
        self.original_mlp = original_mlp
        # Ensure nss_direction is a list for NSSLayer
        if not isinstance(nss_direction, list):
            nss_direction = [nss_direction]
        self.nss_layer = NSSLayer(nss_direction, alpha)

    def forward(self, x):
        x = self.original_mlp(x)
        x = self.nss_layer(x)
        return x


def get_nested_attr(obj, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def set_nested_attr(obj, attr_path, value):
    attrs = attr_path.split(".")
    parent = get_nested_attr(obj, ".".join(attrs[:-1]))
    setattr(parent, attrs[-1], value)


def find_longest_modulelist(model, path=""):
    """
    Recursively find the longest nn.ModuleList in a PyTorch model.
    Args:
        model: PyTorch model.
        path: Current path in the model (used for recursion).
    Returns:
        Tuple with path and length of the longest nn.ModuleList found.
    """
    longest_path = path
    longest_len = 0

    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = f"{path}.{name}" if path else name

        # Recursively check the child's children
        child_path, child_len = find_longest_modulelist(child, f"{path}.{name}" if path else name)
        if child_len > longest_len:
            longest_len = child_len
            longest_path = child_path

    return longest_path, longest_len


def find_module(block, keywords):
    """
    Try to find a module in a transformer block.
    Args:
        block: Transformer block (nn.Module).
        keywords: List of possible module names (str).
    Returns:
        The found module if found, else None.
    """
    for name, module in block.named_modules():
        if any(keyword in name for keyword in keywords):
            return module
    submodule_names = [name for name, _ in block.named_modules()]
    raise ValueError(f"Could not find keywords {keywords} in: {submodule_names}")


def get_embedding_layer(model: PreTrainedModel):
    # model_type = model.__class__.__name__
    # if model_type == "LlamaForCausalLM":
    #     return model.model.embed_tokens
    # elif model_type == "RWForCausalLM":
    #     return model.transformer.word_embeddings

    keywords = ["emb", "wte"]
    return find_module(model, keywords)

def get_layers_path(model: PreTrainedModel):
    longest_path, longest_len = find_longest_modulelist(model)
    return longest_path


def get_layers(model: PreTrainedModel):
    longest_path = get_layers_path(model)
    return get_nested_attr(model, longest_path)

def get_mlp_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    mlp_layers = [find_module(layer, mlp_keywords) for layer in layers]
    return mlp_layers

def add_nss_layers(model: PreTrainedModel, nss_drections: Tensor, alpha: list):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    assert len(nss_drections) == len(layers)
    for i, layer in enumerate(layers):
        original_mlp = find_module(layer, mlp_keywords)
        nss_mlp = nss_MLP(original_mlp, nss_drections[i], alpha)
        setattr(layer, mlp_keywords[0], nss_mlp)

def remove_nss_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"] 
    for i, layer in enumerate(layers):
        nss_mlp = find_module(layer, mlp_keywords)
        if isinstance(nss_mlp, nss_MLP):
            setattr(layer, mlp_keywords[0], nss_mlp.original_mlp)
