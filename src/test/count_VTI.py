import re
import json
import argparse
import jsonlines
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Count correct answer in VTI method.")
    parser.add_argument("--input-file", type=str, required=True, help="file input jsonl.")
    return parser.parse_args()

count_correct = 0
count_outline = 0

data_f = json.load(open("data/baseline_caption_outline_openai.json", "r"))
dict_f = {}
for item in data_f:
    if item['video_id'] not in dict_f:
        dict_f[item['video_id']] = []
    dict_f[item['video_id']].append({
        "correct_answer_id": item['correct_answer_id'],
        "outline_answer_id": item['outline_answer_id'],
        "question": item['question']
    })

args = parse_args()

with jsonlines.open(args.input_file, "r") as data:
    for item in data:
        
        predicted_answer = item['answer_model']

        # for i in dict_f[item['id']]:
        #     if i['question'] == item['question']:
        #         correct_answer = i['correct_answer_id']
        #         outline_answer = i['outline_answer_id']
        #         break
        
        correct_answer = item['answer_correct_id']
        outline_answer = item['answer_outline_id']

        predicted_answer = predicted_answer.replace(".", "")
        number_id = re.findall(r'\b\d+\b', predicted_answer)

        if str(correct_answer) in number_id:
            # print(predicted_answer, correct_answer)
            # input("Correct answer found, press Enter to continue...")
            count_correct += 1
        elif str(outline_answer) in number_id:
            # print(predicted_answer, outline_answer)
            # input("Outline answer found, press Enter to continue...")
            count_outline += 1
            

print(f"Accuracy for correct answer: {count_correct}")
print(f"Accuracy for outline answer: {count_outline}")
