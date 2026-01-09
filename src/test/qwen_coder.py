import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
import re
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGenerationMask

# --- STEP 0: SETUP ---
# Load both models and their corresponding tokenizers/processors

# Model 1: The Vision-Language model to "see" the image
print("Loading Vision-Language model (Qwen-VL)...")
vl_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
vl_processor = AutoProcessor.from_pretrained(vl_model_name, trust_remote_code=True, cache_dir="./cache_dir")
vl_model = Qwen2_5_VLForConditionalGenerationMask.from_pretrained(
    vl_model_name,
    trust_remote_code=True,
    device_map="auto",
    cache_dir="./cache_dir",
    torch_dtype=torch.float16
)

# Model 2: The Coder model for text-based masked token prediction
print("Loading Coder model (Qwen-Coder)...")
coder_model_name = "unsloth/Qwen2.5-Coder-1.5B-Instruct"
coder_tokenizer = AutoTokenizer.from_pretrained(coder_model_name, cache_dir="./cache_dir")
coder_model = AutoModelForCausalLM.from_pretrained(
    coder_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    cache_dir="./cache_dir",
).to("cuda")


# --- STEP 1: GENERATE IMAGE DESCRIPTION WITH VL MODEL ---
print("\n--- Step 1: Generating image description with Qwen-VL ---")
image_path = "dogandcat.png"
question_text = "This is a picture of a tree"

# Prepare input for the VL model to describe the image
description_messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]
vl_text_prompt = vl_processor.apply_chat_template(description_messages, tokenize=False, add_generation_prompt=True)
vl_image_inputs, vl_video_inputs = process_vision_info(description_messages)
vl_inputs = vl_processor(
    text=[vl_text_prompt], images=vl_image_inputs, videos=vl_video_inputs, padding=True, return_tensors="pt"
).to("cuda")

# Generate the description
with torch.no_grad():
    generated_ids = vl_model.generate(**vl_inputs, max_new_tokens=128, do_sample=False)
    generated_text = vl_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Extract only the assistant's response
    image_description = generated_text.split("assistant\n")[-1].strip()

print(f"Generated Image Context: \"{image_description}\"")


# --- STEP 2: MASKED TOKEN PREDICTION WITH CODER MODEL ---
print("\n--- Step 2: Predicting masked tokens with Qwen-Coder ---")

# Get the mask token for the Coder model
if coder_tokenizer.mask_token is None:
    # Fallback if no specific mask token exists
    mask_token = "[MASK]"
    coder_tokenizer.add_special_tokens({'mask_token': mask_token})
else:
    mask_token = coder_tokenizer.mask_token

mask_token_id = coder_tokenizer.convert_tokens_to_ids(mask_token)

# Tokenize the question part separately to iterate over it
question_tokens = coder_tokenizer.tokenize(question_text)
question_token_ids = coder_tokenizer.convert_tokens_to_ids(question_tokens)
token_embeddings = coder_model.get_input_embeddings().weight

# Build the comprehensive list of special tokens to filter out
special_token_ids = set(coder_tokenizer.all_special_ids)
vocab = coder_tokenizer.get_vocab()
for token, token_id in vocab.items():
    if re.match(r"<.*>", token):
        special_token_ids.add(token_id)
special_token_ids = list(special_token_ids)


for i, (token_str, token_id) in enumerate(zip(question_tokens, question_token_ids)):
    
    # Construct the masked question
    masked_question_tokens = question_tokens[:i] + [mask_token] + question_tokens[i+1:]
    masked_question = coder_tokenizer.convert_tokens_to_string(masked_question_tokens)

    # Create the full prompt for the Coder model
    coder_prompt = f"Context: {image_description}\n\nBased on the context, complete the following sentence:\n{masked_question}"
    
    inputs = coder_tokenizer(coder_prompt, return_tensors="pt").to("cuda")
    input_ids = inputs.input_ids

    # Find the position of the mask token
    mask_position = torch.where(input_ids == mask_token_id)[1]
    if mask_position.size(0) == 0:
        print(f"Warning: Mask token not found for original token '{token_str}'. Skipping.")
        continue
    mask_position = mask_position[0].item()

    # Get logits from the Coder model
    with torch.no_grad():
        outputs = coder_model(**inputs)
        logits = outputs.logits
    
    masked_position_logits = logits[0, mask_position, :]

    # Filter out special tokens from predictions
    filtered_logits = masked_position_logits.clone()
    filtered_logits[special_token_ids] = -float('inf')
    
    # Get top 5 predictions
    top_k = 5
    probs = torch.softmax(filtered_logits, dim=-1)
    top_k_probs, top_k_ids = torch.topk(probs, top_k)

    predicted_token_id = top_k_ids[0].item()

    # Calculate relatedness
    original_token_embedding = token_embeddings[token_id]
    predicted_token_embedding = token_embeddings[predicted_token_id]
    relatedness = F.cosine_similarity(
        original_token_embedding.unsqueeze(0),
        predicted_token_embedding.unsqueeze(0)
    )

    print(f"Original Token: '{token_str}'")
    print(f"Top Predicted Token: '{coder_tokenizer.decode([predicted_token_id])}'")
    print(f"Relatedness (Cosine Similarity): {relatedness.item():.4f}")
    
    print(f"--- Top {top_k} Meaningful Predictions ---")
    for j in range(top_k):
        pred_id = top_k_ids[j].item()
        pred_token = coder_tokenizer.decode([pred_id])
        pred_prob = top_k_probs[j].item()
        print(f"{j+1}. '{pred_token}' (Prob: {pred_prob:.3f})")
    
    print("-" * 30)
