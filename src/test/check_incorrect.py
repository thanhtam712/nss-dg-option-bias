import json
import argparse

def parser_args():
    parser = argparse.ArgumentParser(description="Check incorrect answers.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input JSON file.")
    return parser.parse_args()


args = parser_args()
data = json.load(open(args.input_file, "r"))
count = 0
count_outline = 0
correct_0 = 0
correct_1 = 0
correct_2 = 0
correct_3 = 0
correct_4 = 0
total = len(data)

data_no_duplicates = []
seen = set()
for item in data:
    identifier = (item["question"], item.get("video_id"))
    if identifier not in seen:
        seen.add(identifier)
        data_no_duplicates.append(item)


for item in data_no_duplicates:
    correct_id = item["correct_answer_id"]
    outline_id = item["outline_answer_id"]
    # if "answer_model_4" in item:
    #     answer_case4 = int(item["answer_model_4"])
    #     total += 1
    # else:
    #     continue
    
    # if correct_id == answer_case4:
    #     count += 1
    # if outline_id == answer_case4:
    #     count_outline += 1
    
    if "answer_model_0" in item and int(item["answer_model_0"]) == correct_id:
        correct_0 += 1
    if "answer_model_1" in item and int(item["answer_model_1"]) == correct_id:
        correct_1 += 1
    if "answer_model_2" in item and int(item["answer_model_2"]) == correct_id:
        correct_2 += 1
    if "answer_model_3" in item and int(item["answer_model_3"]) == correct_id:
        correct_3 += 1
    if "answer_model_4" in item and int(item["answer_model_4"]) == correct_id:
        correct_4 += 1
    if "answer_model_4" in item and int(item["answer_model_4"]) == outline_id:
        count_outline += 1
        
print (f"Model 0 correct: {correct_0}/{total} = {correct_0/total:.4f}")
print (f"Model 1 correct: {correct_1}/{total} = {correct_1/total:.4f}")
print (f"Model 2 correct: {correct_2}/{total} = {correct_2/total:.4f}")
print (f"Model 3 correct: {correct_3}/{total} = {correct_3/total:.4f}")
print (f"Model 4 correct: {correct_4}/{total} = {correct_4/total:.4f}")

print(count)
print(count_outline)
print(total)
print(len(data_no_duplicates))
