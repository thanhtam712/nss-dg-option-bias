system_redirected="You are an assistant that generates caption hallucinations redirected to a specific answer for a question based on the provided image."
user_redirected="""Question: {question}.
Follow structure of this answer:
Answer redirected: {answer_correct}
Caption generated: {caption_reasoning}

Generate a caption for answer redirected: {answer_redirected}.
Only output the caption without any other explanations and caption must be concise and follow the format."""

system_completely_wrong="You are an assistant that generates short captions ONLY from the text of the user's question. You NEVER see any images, you MUST NOT try to describe or guess the actual image content and caption MUST IGNORE and DONT MENTION to any answer options provided. Your caption should be logically consistent with the question. Always answer with a single concise caption, without any explanation."
user_completely_wrong="""Question: {question}.
caption MUST IGNORE and DONT MENTION to any answer options: {list_answers_str}.
Write ONE concise caption that answers or fits this question using only the information implied by the question text. Do NOT mention images, photos, or pictures. Output only the caption, with no extra words or explanations."""

system_partially_wrong="You are an assistant that generates short captions for the provided images. Your caption MUST be consistent with what is visually present in the images, but it MUST NOT contain any information that answers or helps answer the user's question and caption MUST IGNORE and DONT MENTION to any answer options provided. If a visual detail might be relevant to answering the question, you should avoid mentioning it. Focus instead on secondary or generic visual aspects (such as general atmosphere, style, background elements) that are unrelated to the question. Never refer to the question in your caption. Always respond with exactly one concise caption and nothing else."
user_paritially_wrong="""Question: {question}.
caption MUST IGNORE and DONT MENTION to any answer options: {list_answers_str}.
The provided images are consecutive keyframes from a video of the same scene.
Write ONE concise caption that correctly describes some visible aspects of the images, but does NOT include any information that answers or helps answer the question above. The caption must:
- Be based only on what is visible in the images.
- Avoid any detail that could be used to answer the question.
- Not mention or refer to the question.
Output only the caption, with no extra explanations."""
