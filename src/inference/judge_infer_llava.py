import sys
import json
import random
import time
import argparse
from os import getenv
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import List
sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Generate captions using Gemini API")
    parser.add_argument("--input-result", type=str, required=True, help="Path to input JSON file with answers and question")
    parser.add_argument("--samples", type=str, required=True, help="Path to input JSON file with samples")
    parser.add_argument("--output-answer", type=str, required=True, help="Path to output JSON file to save answer id")
    return parser.parse_args()

prompt_judge_id = """
You are an expert judge in evaluating answers to questions based on given options. Your task is to determine which answer option best corresponds to the provided prediction and **ANSWER ONLY WITH THE ANSWER OPTION ID**.

Question: {question}
Answer Options:
{answers_list}
Prediction: {prediction}

Please identify the answer option that most accurately matches the prediction. Respond only with the answer option ID above between number 0 and {num_indexes}.
Answer Option ID:
If none of the options match, respond with "-1".
"""


def load_llm():
    """
    Load the Gemini client for vision-capable models.
    """
    api_key = getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    return client

def judge_answer_id(client, question: str, answers: List[str], prediction: str) -> str:
    """
    Generate caption hallucination using both image and text.
    
    Args:
        client: Gemini client
        question: Question text
        answer_correct: Correct answer
        caption_reasoning: Reasoning for correct answer
        answer_redirected: Answer to redirect to (hallucination)
        image_path: Path to image file
    
    Returns:
        Generated caption
    """
    prompt = prompt_judge_id.format(
        question=question,
        answers_list="\n".join([f"{i}: {answer}" for i, answer in enumerate(answers)]),
        prediction=prediction,
        num_indexes=len(answers)-1
    )
    
    contents = []
    contents.append(types.Part(text=prompt))
    
    def generate():
        return client.models.generate_content(
            model="gemini-2.5-pro",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.7
            )
        )
    
    response = generate()
    
    if response is None or not hasattr(response, 'text') or response.text is None:
        raise Exception("Failed to generate content after retries")
    return response.text.strip()


def filter_samples(dict_samples, item: list):
    id = item["id"]
    question = item["question"]
    
    if id in dict_samples:
        for existing_item in dict_samples[id]:
            if existing_item["question"] == question:
                return existing_item
    return False

def clean_prediction_id(prediction_id: str) -> str:
    # Clean the prediction_id to extract only the numeric part
    prediction_id = prediction_id.strip().replace(".", "")
    for i in range(-1, 5):
        if f"{i}" in prediction_id and all(f"{j}" not in prediction_id for j in range(-1, 5) if j != i):
            return str(i)
    return prediction_id

def judge_llava(input_path=None, samples=None, answer_path=None):
    client = load_llm()
    
    f_save = open(answer_path, "a+")
    
    samples_data = json.load(open(samples, "r"))
    dict_samples = {}
    for item in samples_data:
        id = item["image_id"] if "image_id" in item else item["video_id"]
        if id not in dict_samples:
            dict_samples[id] = []
        dict_samples[id].append(item)

    # Load JSONL format (one JSON object per line)
    input_data = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                input_data.append(json.loads(line))
    total_samples = []
    correct_caption_true, correct_caption_completely_wrong, correct_caption_partially_wrong, correct_caption_redirected, correct_caption_irrelevance, correct_no_caption = 0, 0, 0, 0, 0, 0
    redirected_caption_redirected = 0

    for item in tqdm(input_data, desc="Generating captions", total=len(input_data)):
        id = item["id"]
        question = item["question"]
        item_sample = filter_samples(dict_samples, item)
        if item_sample is False:
            print(f"Skipping item {id} as no matching sample found.")
            continue
        num_choices = len([key for key in item_sample.keys() if key.startswith("answer_choice_")])
        answers = [item_sample[f"answer_choice_{i}"] for i in range(num_choices)]
        
        correct_answer_id = item["correct_answer_id"]
        outline_answer_id = item["outline_answer_id"]
        
        for prediction_name in ["prediction_caption_true", "prediction_caption_completely_wrong", "prediction_caption_partially_wrong", "prediction_caption_redirected", "prediction_caption_irrelevance", "prediction_no_caption"]:
            prediction = item[prediction_name]
            
            if prediction == "0" or prediction == "1" or prediction == "2" or prediction == "3" or prediction == "4" or prediction == "5":
                print(f"Skipping item {id} as it has already been processed.")
                continue
             
            prediction_id = judge_answer_id(
                client=client,
                question=question,
                answers=answers,
                prediction=prediction
            )
            
            prediction_id = prediction_id.replace("Caption generated:", "").strip()
            prediction_id = clean_prediction_id(prediction_id)
            
            time_traceback = 0
            while prediction_id not in [str(i) for i in range(-1, num_choices)] and time_traceback < 10:
                print(f"Number of choices: {num_choices}")
                print(f"Invalid answer id received: {prediction_id}. Retrying...")
                prediction_id = judge_answer_id(
                    client=client,
                    question=question,
                    answers=answers,
                    prediction=prediction
                )
                prediction_id = prediction_id.replace("Caption generated:", "").strip()
                prediction_id = clean_prediction_id(prediction_id)
                time_traceback += 1
                
            print(f"Answer id for {prediction_name}: {prediction_id}")
            print("-"*50)
            
            item[prediction_name] = prediction_id
            
            if prediction_name == "prediction_caption_true" and prediction_id == str(correct_answer_id):
                correct_caption_true += 1
            if prediction_name == "prediction_caption_completely_wrong" and prediction_id == str(correct_answer_id):
                correct_caption_completely_wrong += 1
            if prediction_name == "prediction_caption_partially_wrong" and prediction_id == str(correct_answer_id):
                correct_caption_partially_wrong += 1
            if prediction_name == "prediction_caption_redirected" and prediction_id == str(correct_answer_id):
                correct_caption_redirected += 1
            if prediction_name == "prediction_caption_irrelevance" and prediction_id == str(correct_answer_id):
                correct_caption_irrelevance += 1
            if prediction_name == "prediction_no_caption" and prediction_id == str(correct_answer_id):
                correct_no_caption += 1 
            if prediction_name == "prediction_caption_redirected" and prediction_id == str(outline_answer_id):
                redirected_caption_redirected += 1
            
        f_save.write(json.dumps(item) + "\n")
        f_save.flush()
            
        total_samples.append(item)
    
    f_save.close()

    print(f"Correct Caption True: {correct_caption_true}/{len(total_samples)}")
    print(f"Correct Caption Completely Wrong: {correct_caption_completely_wrong}/{len(total_samples)}") 
    print(f"Correct Caption Partially Wrong: {correct_caption_partially_wrong}/{len(total_samples)}")
    print(f"Correct Caption Redirected: {correct_caption_redirected}/{len(total_samples)}")
    print(f"Correct Caption Irrelevance: {correct_caption_irrelevance}/{len(total_samples)}")
    print(f"Correct No Caption: {correct_no_caption}/{len(total_samples)}")
    print(f"Redirected Caption Redirected: {redirected_caption_redirected}/{len(total_samples)}")
    
    return correct_caption_true, correct_caption_completely_wrong, correct_caption_partially_wrong, correct_caption_redirected, correct_caption_irrelevance, correct_no_caption, redirected_caption_redirected

if __name__ == "__main__":
    args = parse_args()
    judge_llava(args.input_result, args.samples, args.output_answer)
