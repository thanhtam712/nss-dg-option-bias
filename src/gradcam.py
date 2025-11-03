# pip install -U transformers torch pillow
import torch, torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
import torch
from pathlib import Path
import copy


model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto",  
    device_map="auto",
    trust_remote_code=True,
    # attn_implementation="flash_attention_2",
    # attn_implementation="eager",
    cache_dir="./cache_dir",
)

# sampling_params = SamplingParams(
#     temperature=0.1,
#     top_p=0.001,
#     repetition_penalty=1.05,
#     max_tokens=256,
#     stop_token_ids=[],
# )

processor = AutoProcessor.from_pretrained(model_name)

imgs_inputs = [{
    "type": "image",
    "image": str(img),
} for img in Path("data/keyframes_new/x-Vmw_iRaeI_000059_000069").iterdir()]

list_questions = ["What is [person_1] going to do ?", "What will happen if [person_1] is sleepy ?"]
list_captions = ["playing ee aligns focusing aligns aligns camera giraffe toy toy toy toy aligns align align", "will scenario of in by of to interaction in in of context of festive for context provided provided"]
list_answers = [{0: "[person_1] will check the result.",
                 1: "[person_1] is going to not continue.",
                 2: "[person_1] is going to continue playing with the cups.",
                 3: "[person_1] will show the bees to [person_1].",
                 4: "[person_1] is going to turn left."},
                {0: "[person_1] will feed him.",
                 1: "Someone will take the books from [person_1]'s bed.",
                 2: "[person_1] will lie on the couch.",
                 3: "[person_1] will go to bed.",
                 4: "[person_1] will go to sleep.",}]

prompt_case5 = """
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


for question, caption, answers in zip(list_questions, list_captions, list_answers):
    question = prompt_case5.format(
        question=question,
        answer_0=answers[0],
        answer_1=answers[1],
        answer_2=answers[2],
        answer_3=answers[3],
        answer_4=answers[4],
        information=caption,
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                # {
                #     "type": "image",
                #     "image": "dogandcat.png",
                # },
                *imgs_inputs,
                {"type": "text", "text": "Choose the **correct ID** answer for the question below, you must response with the **ID** answer, don't include text. \n" + question},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        copy.deepcopy(messages), tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    for k, v in inputs.items():
        print(f"{k}: {v.shape} {v.dtype}")

    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=True)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
