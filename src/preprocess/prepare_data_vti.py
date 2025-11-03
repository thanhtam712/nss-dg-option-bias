import json
import random
import jsonlines
from tqdm import tqdm
from pathlib import Path

path_data = Path("data/baseline_caption_outline_openai.json")
data = json.load(open(path_data, "r"))

data_vti_new = []

prompt_question="""Thinking and generate the reasoning to choose the correct answer for the below question follow information from images.
Question:
{question}

Answer options:
0: {answer_0}
1: {answer_1}
2: {answer_2}   
3: {answer_3}
4: {answer_4}
"""

prompt_question2="""Thinking and generate the reasoning to choose the correct answer for the below question follow information from images.
Question:
{question}

Answer options:
0: {answer_0}
1: {answer_1}
2: {answer_2}
3: {answer_3}
4: {answer_4}

More information about the video.
{information}
"""


# caption_outline_openai_caption.jsonl
for sample in tqdm(data):
    video_id = sample["video_id"]
    imagess = list((Path("data/keyframes_new") / video_id).iterdir())
    value = sample["reasoning"]
    h_value = sample["outline_caption"]
    # question = prompt_question.format(
    #     question=sample["question"],
    #     answer_0=sample["answer_choice_0"],
    #     answer_1=sample["answer_choice_1"],
    #     answer_2=sample["answer_choice_2"],
    #     answer_3=sample["answer_choice_3"],
    #     answer_4=sample["answer_choice_4"])
    
    question = prompt_question2.format(
        question=sample["question"],
        answer_0=sample["answer_choice_0"],
        answer_1=sample["answer_choice_1"],
        answer_2=sample["answer_choice_2"],
        answer_3=sample["answer_choice_3"],
        answer_4=sample["answer_choice_4"],
        information=sample["outline_caption"])

    answers_choice = [sample[f"answer_choice_{i}"] for i in range(5)]
    correct_id = sample["correct_answer_id"]
    outline_id = sample["outline_answer_id"]
    
    data_vti_new.append({
        "id": video_id,
        "images": [str(image_path.name) for image_path in imagess],
        "value": value,
        "h_value": h_value,
        "question": question,
        "answers_choice": answers_choice,
        "correct_id": correct_id,
        "outline_id": outline_id
    })

with jsonlines.open("VTI/experiments/data/caption_outline_openai_caption2.jsonl", "w") as f:
    f.write(data_vti_new)


# id_answer_outline_vti.jsonl
# prompt_case5 = """Choose the best answer id from the options below.
# Question:
# {question}

# Answer options:
# 0: {answer_0}
# 1: {answer_1}
# 2: {answer_2}
# 3: {answer_3}
# 4: {answer_4}

# More information about the video.
# {information}
# """

# for sample in tqdm(data):
#     video_id = sample["video_id"]
#     imagess = list((Path("data/keyframes_new") / video_id).iterdir())
#     value = str(sample["correct_answer_id"])
#     h_value = str(sample["outline_answer_id"])
#     question = prompt_case5.format(
#         question=sample["question"],
#         answer_0=sample["answer_choice_0"],
#         answer_1=sample["answer_choice_1"],
#         answer_2=sample["answer_choice_2"],
#         answer_3=sample["answer_choice_3"],
#         answer_4=sample["answer_choice_4"],
#         information=sample["outline_caption"])

#     answers_choice = [sample[f"answer_choice_{i}"] for i in range(5)]
    
#     data_vti_new.append({
#         "id": video_id,
#         "images": [str(image_path.name) for image_path in imagess],
#         "value": value,
#         "h_value": h_value,
#         "question": question,
#         "answers_choice": answers_choice,
#         "reasoning": sample["reasoning"],
#         "hallucinate_caption": sample["outline_caption"],
#     })

# with jsonlines.open("VTI/experiments/data/caption_outline_openai.jsonl", "w") as f:
#     f.write(data_vti_new)


# prompt_case5 = """Choose the correct answer id from the options below.
# Question:
# {question}

# Answer options:
# 0: {answer_0}
# 1: {answer_1}
# 2: {answer_2}
# 3: {answer_3}
# 4: {answer_4}
# """

# for sample in tqdm(data):
#     video_id = sample["video_id"]
#     imagess = list((Path("data/keyframes_new") / video_id).iterdir())
#     value = str(sample["correct_answer_id"])
#     list_choices = [0,1,2,3,4]
#     list_choices.remove(sample["outline_answer_id"])
#     h_value = str(random.choice(list_choices))
#     question = prompt_case5.format(
#         question=sample["question"],
#         answer_0=sample["answer_choice_0"],
#         answer_1=sample["answer_choice_1"],
#         answer_2=sample["answer_choice_2"],
#         answer_3=sample["answer_choice_3"],
#         answer_4=sample["answer_choice_4"])

#     answers_choice = [sample[f"answer_choice_{i}"] for i in range(5)]
    
#     data_vti_new.append({
#         "id": video_id,
#         "images": [str(image_path.name) for image_path in imagess],
#         "value": value,
#         "h_value": h_value,
#         "question": question,
#         "answers_choice": answers_choice,
#         "reasoning": sample["reasoning"],
#         "hallucinate_caption": sample["outline_caption"],
#     })

# with jsonlines.open("VTI/experiments/data/case_1_vti.jsonl", "w") as f:
#     f.write(data_vti_new)
