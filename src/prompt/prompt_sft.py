prompt_user="Here are some additional information which are text descriptions based on the image to assist you for answering the later question. Note, the information could be irrelevant, missing some information or inaccurate, please use it with caution: {caption}\n" + '-'*10 + '\nQuestion: {question}'

assistant_match="The provided text is accurate and provide correct information for answering the question. Given both image and text information, the answer is {answer}"

assistant_corruption="The provided text is inaccurate and provide incorrect information for answering the question. We should ignore the incorrect text information and focus more on the image information. Given this, the answer is {answer}"

assistant_irrelevance="The provided text is irrelevant to the question. We should ignore the text information and only focus on the image information. Given this, the answer is {answer}"
