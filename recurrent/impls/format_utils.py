import random
import re
import markdown
from bs4 import BeautifulSoup

def make_gap_question_from_html(input_text, window_size=300, num_blanks=3, is_md_input=False):
    """
    Parses markdown text, extracts a random text window, and creates a 
    cloze-style (fill-in-the-blank) question.
    
    Args:
        md_text (str): The input markdown string.
        window_size (int): Approx characters to include in the context.
        num_blanks (int): How many words to 'dig out'.
        
    Returns:
        dict: Contains the 'question', 'answer', and original 'plain_text_segment'.
    """
    
    # Extract text from HTML
    if is_md_input:
        html = markdown.markdown(input_text)
    else:
        html = input_text
    text = BeautifulSoup(html, "html.parser").get_text()
    # Clean up excessive newlines and whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if not text:
        text = input_text

    # --- Step 2: Select a Random Window ---
    text_len = len(text)
    
    if text_len <= window_size:
        # If text is short, use the whole thing
        segment = text
    else:
        # Pick a random start point
        start_idx = random.randint(0, text_len - window_size)
        
        # Adjust start_idx to the beginning of a word (find previous space)
        while start_idx > 0 and text[start_idx] != ' ':
            start_idx -= 1
            
        # define tentative end
        end_idx = start_idx + window_size
        
        # Adjust end_idx to the end of a word (find next space)
        while end_idx < text_len and text[end_idx] != ' ':
            end_idx += 1
            
        segment = text[start_idx:end_idx].strip()

    # --- Step 3: Dig Out Words ---
    words = segment.split()
    
    # Filter candidates: Only pick words longer than 2 chars and alphanumeric
    # (Avoids blanking out "a", "I", or punctuation)
    candidate_indices = [i for i, w in enumerate(words) if w.isalnum() and len(w) > 2]

    if len(candidate_indices) < num_blanks:
        target_indices = candidate_indices
    else:
        target_indices = random.sample(candidate_indices, num_blanks)
        target_indices.sort()

    # Construct Question and Answer
    question_words = words[:]
    answers = []

    for idx in target_indices:
        answers.append(words[idx])
        question_words[idx] = "[______]"
    question_words_str = " ".join(question_words)
    
    question_str = question_words_str + '\n' + """Question: Read the paragraph and fill the gap (_______) based on the image. Give your answer in format of \\boxed{{}} (E.g. \\boxed{xxx, yyy}."""

    return {
        "question": question_str,
        "answer": ", ".join(answers),
        "original_segment": segment
    }

if __name__ == "__main__":
    # --- Example Usage ---
    sample_markdown = """
    # The Apollo Program
    The **Apollo program**, also known as [Project Apollo](https://en.wikipedia.org), 
    was the third United States human spaceflight program carried out by the *NASA*. 
    It succeeded in preparing and landing the first humans on the Moon from 1968 to 1972.
    """

    sample_html = markdown.markdown(sample_markdown)
    result = make_gap_question_from_html(sample_html)

    print(f"Question: {result['question']}")
    print("-" * 20)
    print(f"Answer:   {result['answer']}")