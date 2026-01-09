from csv import reader
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

def generate_caption_with_image(client, question, answer_correct, caption_reasoning, answer_redirected, image_path: str = None):
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
    base64_image = encode_image_to_base64(image_path) if image_path else None
    
    # Create messages with image
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that generates caption hallucinations redirected to a specific answer for a question based on the provided image."
        },
        {
            "role": "user",
            "content": [
                # {
                #     "type": "image_url",
                #     "image_url": {
                #         "url": f"data:image/jpeg;base64,{base64_image}" if base64_image else None
                #     }
                # },
                {
                    "type": "text",
                    "text": f"""Question: {question}.
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
        model="gpt-4.1-mini",
        messages=messages,
        max_tokens=512
    )
    
    return response.choices[0].message.content.strip()


def generate_caption_completely_wrong(client, question, list_answers: list = None):
    """
    Generate caption completely wrong using both image and text.
    
    Args:
        client: OpenAI client
        question: Question text
        answer_correct: Correct answer
        caption_reasoning: Reasoning for correct answer
        answer_redirected: Answer to redirect to (hallucination)
        image_path: Path to image file
    
    Returns:
        Generated caption completely wrong
    """
    list_answers_str = "\n".join(ans for ans in list_answers) if list_answers else ""
    print(list_answers_str)
    # Create messages with image
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that generates short captions ONLY from the text of the user's question. You NEVER see any images, you MUST NOT try to describe or guess the actual image content and caption MUST IGNORE and DONT MENTION to any answer options provided. Your caption should be logically consistent with the question. Always answer with a single concise caption, without any explanation."
        },
        {
            "role": "user",
            "content": [
                # {
                #     "type": "image_url",
                #     "image_url": {
                #         "url": f"data:image/jpeg;base64,{base64_image}" if base64_image else None
                #     }
                # },
                {
                    "type": "text",
                    "text": f"""Question: {question}.
caption MUST IGNORE and DONT MENTION to any answer options: {list_answers_str}.
Write ONE concise caption that answers or fits this question using only the information implied by the question text. Do NOT mention images, photos, or pictures. Output only the caption, with no extra words or explanations."""
                }
            ]
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        max_tokens=512
    )
    
    return response.choices[0].message.content.strip()


def generate_caption_partially_wrong(client, question, list_answers: list, images_path: str = None):
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
    
    print(images_path)
    
    images_url = []
    for image_path in images_path:
        base64_image = encode_image_to_base64(image_path) if image_path else None
        images_url.append({"type": "image_url",
                           "image_url": {"url": f"data:image/jpeg;base64,{base64_image}" if base64_image else None}})
    
    list_answers_str = "\n".join(ans for ans in list_answers)
    # Create messages with image
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that generates short captions for the provided images. Your caption MUST be consistent with what is visually present in the images, but it MUST NOT contain any information that answers or helps answer the user's question and caption MUST IGNORE and DONT MENTION to any answer options provided. If a visual detail might be relevant to answering the question, you should avoid mentioning it. Focus instead on secondary or generic visual aspects (such as general atmosphere, style, background elements) that are unrelated to the question. Never refer to the question in your caption. Always respond with exactly one concise caption and nothing else."
        },
        {
            "role": "user",
            "content": [
                *images_url,
                {
                    "type": "text",
                    "text": f"""Question: {question}.
caption MUST IGNORE and DONT MENTION to any answer options: {list_answers_str}.
The provided images are consecutive keyframes from a video of the same scene.
Write ONE concise caption that correctly describes some visible aspects of the images, but does NOT include any information that answers or helps answer the question above. The caption must:
- Be based only on what is visible in the images.
- Avoid any detail that could be used to answer the question.
- Not mention or refer to the question.
Output only the caption, with no extra explanations."""
                }
            ]
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        max_tokens=512
    )
    
    return response.choices[0].message.content.strip()

def random_outline_answer_id(correct_answer_id, num_choices=5):
    """
    Generate a random outline answer ID different from the correct answer ID.
    
    Args:
        correct_answer_id: The correct answer ID
        num_choices: Total number of answer choices
    """
    
    outline_answer_id = random.randint(0, num_choices - 1)
    while outline_answer_id == correct_answer_id:
        outline_answer_id = random.randint(0, num_choices - 1)
    return outline_answer_id

def load_samples(video_path: Path, question_path: Path, answer_path: Path):
    """
    Load samples from video frames and corresponding Q&A files.
    
    Args:
        video_path: Path to video frames folder
        question_path: Path to questions JSON file
        answer_path: Path to answers JSON file
    """
    
    if not Path(video_path).exists() or not Path(question_path).exists() or not Path(answer_path).exists():
        raise ValueError("One of the provided paths does not exist.")
        
    
    video_id = video_path.stem
    data_question = json.load(open(question_path, "r"))
    data_answer = json.load(open(answer_path, "r"))
    
    return [
        {
            "video_id": video_id,
            "question": data_question["predictive"]["question"],
            "answer_choice_0": data_question["predictive"]["answer"][0],
            "answer_choice_1": data_question["predictive"]["answer"][1],
            "answer_choice_2": data_question["predictive"]["answer"][2],
            "answer_choice_3": data_question["predictive"]["answer"][3],
            "answer_choice_4": data_question["predictive"]["answer"][4],
            "correct_answer_id": data_answer["predictive"]["answer"],
            "reasoning": data_question["predictive"]["reason"][data_answer["predictive"]["reason"]],
            "outline_answer_id": random_outline_answer_id(data_answer["predictive"]["answer"]),
            "outline_caption": ""
        },
        {
            "video_id": video_id,
            "question": data_question["counterfactual"]["question"],
            "answer_choice_0": data_question["counterfactual"]["answer"][0],
            "answer_choice_1": data_question["counterfactual"]["answer"][1],
            "answer_choice_2": data_question["counterfactual"]["answer"][2],
            "answer_choice_3": data_question["counterfactual"]["answer"][3],
            "answer_choice_4": data_question["counterfactual"]["answer"][4],
            "correct_answer_id": data_answer["counterfactual"]["answer"],
            "reasoning": data_question["counterfactual"]["reason"][data_answer["counterfactual"]["reason"]],
            "outline_answer_id": random_outline_answer_id(data_answer["counterfactual"]["answer"]),
            "outline_caption": ""
        }
    ]


def main():
    client = load_llm()
    f_save = open("data/baseline_caption_outline_openai_full_caption_completely_partially_backup.json", "a+")
    
    video_root = Path("/mlcv3/Datasets/Thesis/dataset/")
    answer_root = Path("data/QA")
    video_check = Path("data/keyframes_full").iterdir()
    video_check = [v.stem for v in video_check]
    
    # data_json_generated = json.load(open("data/baseline_caption_outline_openai_full.json", "r"))
    # video_generated = set()
    # for item in data_json_generated:
    #     video_id = item["video_id"]
    #     video_generated.add(video_id)
    
    total_samples = []
    
    # for id in tqdm(video_root.iterdir(), total=len(list(video_root.iterdir()))):
    #     if id.stem not in video_check:
    #         print(f"Skipping video ID: {id.stem}")
    #         continue

        # if id.stem in video_generated:
        #     print(f"Already generated for video ID: {id.stem}, skipping.")
        #     continue

        # samples = load_samples(video_path=Path(list(id.glob("*.mp4"))[0]),
        #              question_path=Path(list(id.glob("*.json"))[0]),
        #              answer_path=Path(answer_root / Path(id.stem) / "answer.json"))
        
    samples = json.load(open("data/data_samples/Causal_Vid/baseline_caption_outline_openai_full_combined.json", "r"))
    
    for item in samples:
        # DATA CAUSAL_VID
        question = item["question"]
        correct_answer_id = item["correct_answer_id"]
        if correct_answer_id not in [0,1,2,3,4]:
            print(f"Skipping invalid correct_answer_id: {correct_answer_id}")
            continue
        answer_choice_correct = item[f"answer_choice_{correct_answer_id}"]
        reasoning = item["reasoning"]
        outline_answer_id = item["outline_answer_id"]
        answer_choice_outline = item[f"answer_choice_{outline_answer_id}"]
    
        print(f"Processing: {question}")
        print(f"Correct answer: {answer_choice_correct}")
        print(f"Reasoning: {reasoning}")
        print(f"Redirected answer: {answer_choice_outline}")

        try:
            # Get keyframes for this video
            keyframes_dir = Path("data/keyframes_full") / item["video_id"]
            image_paths = [str(p) for p in sorted(keyframes_dir.iterdir()) if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
            
            caption_completely_wrong = generate_caption_completely_wrong(
                client=client,
                question=question,
                list_answers=[item[f"answer_choice_{i}"] for i in range(5)]
            )
            
            caption_partially_wrong = generate_caption_partially_wrong(
                client=client,
                question=question,
                list_answers=[item[f"answer_choice_{i}"] for i in range(5)],
                images_path=image_paths
            )
            
            caption_completely_wrong = caption_completely_wrong.replace("Caption generated:", "").strip()
            caption_partially_wrong = caption_partially_wrong.replace("Caption generated:", "").strip()
            
            print(f"Generated caption completely wrong: {caption_completely_wrong}")
            print("-"*50)
            print(f"Generated caption partially wrong: {caption_partially_wrong}")
            print("-"*50)   
            input("Press Enter to continue...")
                                            
            item["caption_completely_wrong"] = caption_completely_wrong
            item["caption_partially_wrong"] = caption_partially_wrong
            
            f_save.write(json.dumps(item) + "\n")

        except Exception as e:
            print(f"Error generating caption: {e}")
            continue
        
        total_samples.append(item)
    
    f_save.close()
    json.dump(total_samples, open("data/baseline_caption_outline_openai_full_caption_completely_partially.json", "w"), indent=4)
    print("Done! Saved to data/baseline_caption_outline_openai_full_caption_completely_partially.json")

    
    # f_json = "save_caption_outline_Qwen.json"
    # f_json = "data/aokvqa/dataset/aokvqa_v1p0_val.json"

    # reader = json.load(open(f_json, "r"))


    # for item in tqdm(reader, total=len(reader)):
        
    #     # DATA CAUSAL_VID
    #     # question = item["question"]
    #     # correct_answer_id = item["correct_answer_id"]
    #     # answer_choice_correct = item[f"answer_choice_{correct_answer_id}"]
    #     # reasoning = item["reasoning"]
    #     # outline_answer_id = item["outline_answer_id"]
    #     # answer_choice_outline = item[f"answer_choice_{outline_answer_id}"]
        
    #     # DATA AOKVQA
    #     question = item["question"]
    #     correct_choice_idx = item["correct_choice_idx"]
    #     answer_choice_correct = item["choices"][correct_choice_idx]
    #     reasoning = " ".join(item["rationales"])
    #     image_path = Path("data/aokvqa/dataset/image/val2017") / f"000000{item['image_id']}.jpg"

    #     if not image_path.exists():
    #         print(f"Warning: Image not found: {image_path}")
    #         continue

    #     outline_choice_idx = random.randint(0, len(item["choices"]) - 1)
    #     while outline_choice_idx == correct_choice_idx:
    #         outline_choice_idx = random.randint(0, len(item["choices"]) - 1)
    #     answer_choice_outline = item["choices"][outline_choice_idx]
    
    #     print(f"Processing: {question}")
    #     print(f"Image: {image_path}")
    #     print(f"Correct answer: {answer_choice_correct}")
    #     print(f"Redirected answer: {answer_choice_outline}")

    #     try:
    #         caption_outline = generate_caption_with_image(
    #             client=client,
    #             question=question,
    #             answer_correct=answer_choice_correct,
    #             caption_reasoning=reasoning,
    #             answer_redirected=answer_choice_outline,
    #             image_path=str(image_path)
    #         )
            
    #         caption_outline = caption_outline.replace("Caption generated:", "").strip()
            
    #         print(f"Generated caption: {caption_outline}")
    #         print("-"*50)
            
    #         item["outline_caption"] = caption_outline
    #         item["image_path"] = str(image_path)

    #         f_save.write(json.dumps(item) + "\n")

    #     except Exception as e:
    #         print(f"Error generating caption: {e}")
    #         continue

    # json.dump(reader, open("data/baseline_caption_outline_openai_aokvqa.json", "w"), indent=4)
    # print("Done! Saved to data/baseline_caption_outline_openai_aokvqa.json")


if __name__ == "__main__":
    main()
    