import json
import random
from os import getenv
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import base64
from openai import OpenAI as OpenAIClient

load_dotenv()

def encode_image_to_base64(image_path):
    """
    Encode image to base64 string for OpenAI API.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_llm():
    """
    Load the OpenAI client for vision-capable models.
    """
    api_key = getenv("OPENAI_API_KEY")
    client = OpenAIClient(api_key=api_key)
    
    return client

def generate_caption_with_image(client, question, answer_correct, caption_reasoning, answer_redirected, image_path):
    """
    Generate caption hallucination using both image and text.
    
    Args:
        client: OpenAI client
        question: Question text
        answer_correct: Correct answer
        caption_reasoning: Reasoning for correct answer
        answer_redirected: Answer to redirect to (hallucination)
        image_path: Path to image file
    
    Returns:
        Generated caption
    """
    # Encode image
    base64_image = encode_image_to_base64(image_path)
    
    # Create messages with image
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that generates caption hallucinations redirected to a specific answer for a question based on the provided image."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": f"""Question: {question}
Follow structure of this answer:
Answer redirected: {answer_correct}
Caption generated: {caption_reasoning}

Generate a caption for answer redirected: {answer_redirected}.
Only output the caption without any other explanations and caption must be concise and follow the format."""
                }
            ]
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=512
    )
    
    return response.choices[0].message.content.strip()

client = load_llm()

# f_json = "save_caption_outline_Qwen.json"
f_json = "data/aokvqa/dataset/aokvqa_v1p0_val.json"

reader = json.load(open(f_json, "r"))

f_save = open("data/baseline_caption_outline_openai_aokvqa_backup.json", "a+")

for item in tqdm(reader, total=len(reader)):
    
    # DATA CAUSAL_VID
    # question = item["question"]
    # correct_answer_id = item["correct_answer_id"]
    # answer_choice_correct = item[f"answer_choice_{correct_answer_id}"]
    # reasoning = item["reasoning"]
    # outline_answer_id = item["outline_answer_id"]
    # answer_choice_outline = item[f"answer_choice_{outline_answer_id}"]
    
    # DATA AOKVQA
    question = item["question"]
    correct_choice_idx = item["correct_choice_idx"]
    answer_choice_correct = item["choices"][correct_choice_idx]
    reasoning = " ".join(item["rationales"])
    image_path = Path("data/aokvqa/dataset/image/val2017") / f"000000{item['image_id']}.jpg"

    if not image_path.exists():
        print(f"Warning: Image not found: {image_path}")
        continue

    outline_choice_idx = random.randint(0, len(item["choices"]) - 1)
    while outline_choice_idx == correct_choice_idx:
        outline_choice_idx = random.randint(0, len(item["choices"]) - 1)
    answer_choice_outline = item["choices"][outline_choice_idx]
   
    print(f"Processing: {question}")
    print(f"Image: {image_path}")
    print(f"Correct answer: {answer_choice_correct}")
    print(f"Redirected answer: {answer_choice_outline}")

    try:
        caption_outline = generate_caption_with_image(
            client=client,
            question=question,
            answer_correct=answer_choice_correct,
            caption_reasoning=reasoning,
            answer_redirected=answer_choice_outline,
            image_path=str(image_path)
        )
        
        caption_outline = caption_outline.replace("Caption generated:", "").strip()
        
        print(f"Generated caption: {caption_outline}")
        print("-"*50)
        
        item["outline_caption"] = caption_outline
        item["image_path"] = str(image_path)

        f_save.write(json.dumps(item) + "\n")

    except Exception as e:
        print(f"Error generating caption: {e}")
        continue

json.dump(reader, open("data/baseline_caption_outline_openai_aokvqa.json", "w"), indent=4)
print("Done! Saved to data/baseline_caption_outline_openai_aokvqa.json")
