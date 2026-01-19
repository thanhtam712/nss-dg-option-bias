system_redirected="You are an assistant that generates text hallucinations redirected to a specific answer for a question based on the provided image."
user_redirected="""Question: {question}.
Follow structure of this answer:
Answer redirected: {answer_correct}
text generated: {text_faithful}

Generate a text for answer redirected: {answer_redirected}.
Only output the text without any other explanations and text must be concise and follow the format."""

system_contradictory="You are an assistant that generates short texts ONLY from the text of the user's question. You NEVER see any images, you MUST NOT try to describe or guess the actual image content and text MUST IGNORE and DONT MENTION to any answer options provided. Your text should be logically consistent with the question. Always answer with a single concise text, without any explanation."
user_contradictory="""Question: {question}.
text MUST IGNORE and DONT MENTION to any answer options: {list_answers_str}.
Write ONE concise text that is on-topic with the question but deliberately introduces a plausible yet incorrect scenario that would contradict the true visual content. The caption must be non-targeted: avoid implying a specific answer choice, avoid providing a strong cue toward any single option, and do not include names/phrases that could match the answer options. Use only general wording derived from the question theme (no external facts). Do NOT mention images, photos, or pictures. Output only the caption."""

system_off_question="You are an assistant that generates short texts for the provided images. Your text MUST be consistent with what is visually present in the images, but it MUST NOT contain any information that answers or helps answer the user's question and text MUST IGNORE and DONT MENTION to any answer options provided. If a visual detail might be relevant to answering the question, you should avoid mentioning it. Focus instead on secondary or generic visual aspects (such as general atmosphere, style, background elements) that are unrelated to the question. Never refer to the question in your text. Always respond with exactly one concise text and nothing else."
user_off_question="""Question: {question}.
text MUST IGNORE and DONT MENTION to any answer options: {list_answers_str}.
The visual inputs are either a single image or consecutive keyframes sampled from the same video scene.
Write ONE concise text that correctly describes some visible aspects of the images, but does NOT include any information that answers or helps answer the question above. The text must:
- Be based only on what is visible in the images.
- Avoid any detail that could be used to answer the question.
- Not mention or refer to the question.
Output only the text, with no extra explanations."""
