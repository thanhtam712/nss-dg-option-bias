prompt_user = """
Question:
{question}

Answer options:
{options}

Here are some additional information which are text descriptions based on the image to assist you for answering the later question. Note, the information could be irrelevant, missing some information or inaccurate, please use it with caution: {caption}
"""

prompt_user_no_caption = """
Question:
{question}

Answer options:
{options}
"""

prompt_case1 = """
Question:
{question}

Answer options:
0: {answer_0}
1: {answer_1}
2: {answer_2}
3: {answer_3}
4: {answer_4}
"""

prompt_case2 = """
Question:
{question}

Answer options:
0: {answer_0}
1: {answer_1}
2: {answer_2}
3: {answer_3}
4: {answer_4}

Caption generated based on the question and answer choices above.
{caption_generated}
"""

prompt_case3 = """
Question:
{question}

Answer options:
0: {answer_0}
1: {answer_1}
2: {answer_2}
3: {answer_3}
4: {answer_4}

Caption based on the question and answer choices above.
{caption_reasoning}
"""

prompt_case4 = """
Question:
{question}

Answer options:
0: {answer_0}
1: {answer_1}
2: {answer_2}
3: {answer_3}
4: {answer_4}

Transcript the video.
{transcript}
"""

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
