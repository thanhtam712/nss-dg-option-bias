import sys
import json
import random
import time
import base64
import argparse
from os import getenv
from csv import reader
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
sys.path.append(str(Path(__file__).resolve().parent.parent))
from prompt.prompt_gen_textual import system_redirected, user_redirected, system_completely_wrong, user_completely_wrong, system_partially_wrong, user_paritially_wrong

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Generate textuals using Gemini API")
    parser.add_argument("--images-root", type=str, required=True, help="Path to video frames folder")
    parser.add_argument("--question-path", type=str, required=True, help="Path to questions JSON file")
    parser.add_argument("--json-generated", type=str, default="", required=False, help="Path to previously generated JSONL file to resume from")
    parser.add_argument("--answer-path", type=str, required=True, help="Path to answers JSON file")
    parser.add_argument("--wiki-path", type=str, default="data/wiki/test.txt", required=False, help="Path to wiki TXT file")
    return parser.parse_args()


def encode_image_to_base64(image_path):
    """
    Encode image to base64 string for OpenAI API.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_llm():
    """
    Load the Gemini client for vision-capable models.
    """
    api_key = getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    return client

def generate_textual_redirected(client, question, answer_correct, textual_reasoning, answer_redirected):
    """
    Generate textual hallucination using both image and text.
    
    Args:
        client: Gemini client
        question: Question text
        answer_correct: Correct answer
        textual_reasoning: Reasoning for correct answer
        answer_redirected: Answer to redirect to (hallucination)
        image_path: Path to image file
    
    Returns:
        Generated textual
    """
    prompt = f"{system_redirected}\n\n{user_redirected.format(question=question, answer_correct=answer_correct, textual_reasoning=textual_reasoning, answer_redirected=answer_redirected)}"
    
    contents = []
    contents.append(types.Part(text=prompt))
    
    def generate():
        return client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.7
            )
        )
    
    response = generate()
    
    if response is None or not hasattr(response, 'text') or response.text is None:
        raise Exception("Failed to generate content after retries")
    return response.text.strip()


def generate_textual_completely_wrong(client, question, list_answers: list = None):
    """
    Generate textual completely wrong using both image and text.
    
    Args:
        client: Gemini client
        question: Question text
        list_answers: List of answer choices
    
    Returns:
        Generated textual completely wrong
    """
    list_answers_str = "\n".join(ans for ans in list_answers) if list_answers else ""
    prompt = f"{system_completely_wrong}\n\n{user_completely_wrong.format(question=question, list_answers_str=list_answers_str)}"
    
    def generate():
        return client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7
            )
        )
    
    response = generate()
    
    if response is None or not hasattr(response, 'text') or response.text is None:
        raise Exception("Failed to generate content after retries")
    return response.text.strip()


def generate_textual_partially_wrong(client, question, list_answers: list, images_path: str = None):
    """
    Generate textual hallucination using both image and text.
    
    Args:
        client: Gemini client
        question: Question text
        list_answers: List of answer choices
        images_path: List of image file paths
    
    Returns:
        Generated textual
    """
    print(images_path)
    list_answers_str = "\n".join(ans for ans in list_answers)
    prompt = f"{system_partially_wrong}\n\n{user_paritially_wrong.format(question=question, list_answers_str=list_answers_str)}"
    parts = []

    # Ensure images_path is a list of valid file paths
    valid_image_paths = []
    if images_path:
        if isinstance(images_path, str):
            # If it's a single string, treat as one file path
            if Path(images_path).is_file():
                valid_image_paths = [images_path]
        elif isinstance(images_path, list):
            valid_image_paths = [p for p in images_path if Path(p).is_file()]

    for image_path in valid_image_paths:
        try:
            uploaded_file = client.files.upload(file=image_path)
            parts.append(types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type))
        except Exception as e:
            print(f"Warning: Could not upload image {image_path}: {e}")

    parts.append(types.Part(text=prompt))

    def generate():
        return client.models.generate_content(
            model="gemini-2.5-flash",
            contents=parts,
            config=types.GenerateContentConfig(
                temperature=0.7
            )
        )

    response = generate()
    if response is None or not hasattr(response, 'text') or response.text is None:
        raise Exception("Failed to generate content after retries")
    return response.text.strip()

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
            "outline_textual": ""
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
            "outline_textual": ""
        }
    ]


def load_processed_items(backup_file):
    """Load already processed items from backup file to enable resume."""
    processed = {}
    if Path(backup_file).exists():
        try:
            with open(backup_file, 'r') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        # Create unique key from video_id and question
                        key = f"{item.get('video_id', '')}_{item.get('question', '')[:50]}"
                        processed[key] = item
            print(f"Loaded {len(processed)} already processed items")
        except Exception as e:
            print(f"Error loading processed items: {e}")
    return processed

def filter_samples(dict_samples, item: list):
    video_id = item["video_id"]
    question = item["question"]
    
    if video_id in dict_samples:
        for existing_item in dict_samples[video_id]:
            if existing_item["question"] == question:
                return existing_item
    return False

def samples_generated(file_json: str):
    """Load samples from JSONL format (JSON Lines - one JSON per line)."""
    data_samples_generated = []
    
    if not Path(file_json).exists():
        print(f"Warning: File {file_json} does not exist, returning empty dict")
        return {}
    
    # Read JSONL format (each line is a separate JSON object)
    with open(file_json, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    item = json.loads(line)
                    data_samples_generated.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line: {e}")
                    continue
    
    dict_samples = {}
    for item in data_samples_generated:
        image_id = item.get("image_id")
        if image_id:
            if image_id not in dict_samples:
                dict_samples[image_id] = []
            dict_samples[image_id].append(item)
    
    return dict_samples

def main():
    client = load_llm()
    args = parse_args()
    question_path = args.question_path
    answer_path = args.answer_path
    wiki_path = args.wiki_path
    images_root = args.images_root
    
    Path(answer_path).parent.mkdir(parents=True, exist_ok=True)
    backup_file = answer_path.replace(".json", "_gemini_textual_backup.json")
    f_save = open(backup_file, "a+")

    samples = json.load(open(question_path, "r"))
    data_wiki_irrelevant = open(wiki_path, "r").readlines()
    
    total_samples = []    
    items_processed = 0
    items_skipped = 0
    
    random.shuffle(samples)
    samples = samples[:2500]
    
    dict_generated = samples_generated(args.json_generated) if args.json_generated else {}
    
    for item in tqdm(samples, desc="Generating textuals", total=len(samples)):
        if item["image_id"] in dict_generated:
            continue
        
        question = item["question"]
        
        correct_answer_id = item["correct_answer_id"]
        if correct_answer_id not in [0,1,2,3,4]:
            print(f"Skipping invalid correct_answer_id: {correct_answer_id}")
            continue
        
        answer_choice_correct = item[f"answer_choice_{correct_answer_id}"]
        reasoning = item["reasoning"]
        outline_answer_id = item["outline_answer_id"]
        answer_choice_outline = item[f"answer_choice_{outline_answer_id}"]
        
        num_choices = len([key for key in item.keys() if key.startswith("answer_choice_")])
    
        print(f"\n{'='*60}")
        print(f"Processing: {question}...")
        print(f"Correct answer: {answer_choice_correct}")
        print(f"Reasoning: {reasoning}...")
        print(f"Redirected answer: {answer_choice_outline}")

        try:
            if "video_id" in item:
                keyframes_dir = Path(images_root) / item["video_id"]
                image_paths = [str(p) for p in sorted(keyframes_dir.iterdir()) if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
            else:
                image_paths = str(Path(images_root) / item["image_id"])
                print(f"Using image path: {image_paths}")
            
            textual_redirected = generate_textual_redirected(
                client=client,
                question=question,
                answer_correct=answer_choice_correct,
                textual_reasoning=reasoning,
                answer_redirected=answer_choice_outline,
            )
            
            time.sleep(6)
            
            textual_completely_wrong = generate_textual_completely_wrong(
                client=client,
                question=question,
                list_answers=[item[f"answer_choice_{i}"] for i in range(num_choices)]
            )
            
            time.sleep(6)
            
            textual_partially_wrong = generate_textual_partially_wrong(
                client=client,
                question=question,
                list_answers=[item[f"answer_choice_{i}"] for i in range(num_choices)],
                images_path=image_paths
            )
            
            textual_redirected = textual_redirected.replace("textual generated:", "").strip()
            textual_completely_wrong = textual_completely_wrong.replace("textual generated:", "").strip()
            textual_partially_wrong = textual_partially_wrong.replace("textual generated:", "").strip()
            
            print(f"Generated textual redirected: {textual_redirected}")
            print("-"*50)
            print(f"Generated textual completely wrong: {textual_completely_wrong}")
            print("-"*50)
            print(f"Generated textual partially wrong: {textual_partially_wrong}")
            print("-"*50)   
            
            item["textual_correct"] = item["reasoning"]
            item["textual_redirected"] = textual_redirected
            item["textual_completely_wrong"] = textual_completely_wrong
            item["textual_partially_wrong"] = textual_partially_wrong
            item["textual_irrelevance"] = random.choice(data_wiki_irrelevant).strip()
            
            f_save.write(json.dumps(item) + "\n")
            f_save.flush()
            
        except Exception as e:
            print(f"✗ Error generating textual: {e}")
            print(f"Waiting 60 seconds before continuing to next item...")
            time.sleep(60)
            continue
        
        total_samples.append(item)
    
    f_save.close()
    json.dump(total_samples, open(answer_path, "w"), indent=4)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total items in dataset: {len(samples)}")
    print(f"Items skipped (already processed): {items_skipped}")
    print(f"Items processed in this run: {items_processed}")
    print(f"Total items completed: {len(total_samples)}")
    print("="*60)
    print(f"\nDone! Saved to {answer_path}")

if __name__ == "__main__":
    main()

# python src/preprocess/gemini_gen_textual.py --images-root data/aokvqa/dataset/image/val2017 --question-path data/data_samples/AOK_VQA/preprocessed_aokvqa.json --answer-path data/data_samples/AOK_VQA/baseline_gemini.json 
